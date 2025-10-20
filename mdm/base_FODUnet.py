import einops
import torch
import torch.nn as nn
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.models.diffusion.ddpm import DDPM
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
import pytorch_lightning as pl

from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, TimestepBlock
from ldm.modules.attention import SpatialTransformer, MyCrossAttention


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, volume_enconding=None, context=None, control=None, **kwargs):
        hs = []
        return_controls = []

        emb = torch.zeros(x.shape[0], self.model_channels * 4, device=x.device, dtype=x.dtype)
        emb = emb + volume_enconding

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            for submodule in module:
                if isinstance(submodule, TimestepBlock):
                    h = submodule(h, emb)
                elif isinstance(submodule, SpatialTransformer):
                    h = submodule(h, context)
                elif isinstance(submodule, AttentionBlock):
                    h = submodule(h)
                    if control is None and i in [0, 3, 6]:
                        return_controls.append(h)
                elif isinstance(submodule, MyCrossAttention):
                    if control is not None and i in [1, 4, 7]:
                        h = submodule(h, torch.concat([h, control.pop(0)], dim=1))
                    else:
                        h = submodule(h, torch.concat([h, h], dim=1))
                else:
                    h = submodule(h)

        h = h.type(x.dtype)
        return self.out(h), return_controls

class FeatExtractNet(nn.Module):
    def __init__(
            self,
            dims=3,
            in_channels=5,
            out_channels=1,
            model_channels=96
    ):
        super().__init__()

        self.dims = dims
        self.in_channels = in_channels # 45 = 1 + 5 + 9 + 13 + 17
        self.out_channels = out_channels

        self.model_channels = model_channels

        self.input_blocks = nn.ModuleList(
            [
                conv_nd(dims, in_channels, out_channels, 1, padding=0, bias=False)
            ]
        )
        
        self.act = nn.SiLU()

        self.order_embeder = nn.Sequential(
            linear(self.model_channels*2, in_channels),
        )

    def forward(self, X, volume_enconding=None, **kwargs):
        assert type(volume_enconding) is not type(None)

        order_emb = torch.concat([timestep_embedding(volume_enconding[:, 0], self.model_channels, repeat_only=False),
                                timestep_embedding(volume_enconding[:, 1], self.model_channels, repeat_only=False)], axis=1)
        
        v_emb = self.order_embeder(order_emb)
        X = X + v_emb[..., None, None, None, None]
        b, tmp_c, c, x, y, z = X.shape
        X = rearrange(X, 'b tmp_c c x y z -> b tmp_c c x (y z)')
        for module in self.input_blocks:
            X = module(X)
        X = self.act(X)
        X = rearrange(X, 'b tmp_c c x (y z) -> b tmp_c c x y z', y=y, z=z)
        return X[:, 0, ...]


class FodUnet(pl.LightningModule):
    """
    training_step(ddpm.DDPM) -> shared_step(self) -> get_input(self) -> forward(self) -> p_losses(self) -> apply_model(self) -> 
    """

    def __init__(self, unet_config, feat_extract_stage_config, dims, mask_key, loss_type, target_key, control_key, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(kwargs)

        self.dims = dims
        self.mask_key = mask_key
        self.loss_type = loss_type
        self.target_key = target_key
        self.control_key = control_key

        self.model = instantiate_from_config(unet_config)
        self.feat_extract_net = instantiate_from_config(feat_extract_stage_config)        

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.target_key)
        loss = self(x, c)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)        
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x = batch[self.target_key]
        if self.dims == 2:
            if len(x.shape) == 3:
                x = x[..., None]
            x = rearrange(x, 'b h w c -> b c h w')
        elif self.dims == 3:
            if len(x.shape) == 4:
                x = x[..., None]
            x = rearrange(x, 'b x y z c -> b c x y z')
        x = x.to(memory_format=torch.contiguous_format).float()

        x = x.to(self.device)

        control = batch[self.control_key]
        mask = None
        if self.mask_key is not None:
            mask = batch[self.mask_key]
        meta = batch["meta"]
        attn_input = batch["attn_input"]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b x y z c -> b c x y z')
        control = control.to(memory_format=torch.contiguous_format).float()
        attn_input = attn_input.to(self.device)
        attn_input = einops.rearrange(attn_input, 'b x y z c -> b c x y z')
        attn_input = attn_input.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[None], c_concat=[control], mask=mask, meta=meta, attn_input=attn_input)

    def forward(self, x, c, *args, **kwargs):
        return self.p_losses(x, c, *args, **kwargs)

    def p_losses(self, target, cond):
        model_output = self.apply_model(cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        mask = cond['mask']
        #print(mask.shape)
        mask_shape = mask.shape
        loss_simple = self.get_loss(
            model_output, target, mean=False) * (0.01 + 0.99 * mask) #0.1 0.9 # * (mask_shape[0] * 32 * 32 * 32 / (torch.sum(mask) + 0.1))
        loss_simple = loss_simple.mean([1, 2, 3, 4])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        loss = loss_simple.mean()

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def upper_branch(self, model, cond, volume_emb):
        vols_5 = []
        conv_in = [1, 5, 9, 13, 17]
        conv_in_accum = [0, 1, 6, 15, 28, 45]
        for i in range(len(conv_in)):
            vols_5.append(torch.mean(cond['c_concat'][0][:, conv_in_accum[i]:conv_in_accum[i+1], ...], dim=1, keepdim=True))
        vols_5 = torch.cat(vols_5, dim=1)

        control_input1 = []
        control_input2 = []
        control_input3 = []
        for i in range(vols_5.shape[1]):
            tmp_input = vols_5[:, i:i+1, ...]
            _, controls = model(x=tmp_input, timesteps=None, volume_enconding=volume_emb, context=None) # TODO: check timesteps = 0 or t
            control_input1.append(controls[0])
            control_input2.append(controls[1])
            control_input3.append(controls[2])
        control_input1 = torch.stack(control_input1, dim=1)
        control_input2 = torch.stack(control_input2, dim=1)
        control_input3 = torch.stack(control_input3, dim=1)
        controls = [control_input1, control_input2, control_input3]
        del control_input1, control_input2, control_input3

        return controls

    def apply_model(self, cond, test_cache=None, *args, **kwargs):
        assert isinstance(cond, dict)

        if len(cond['c_crossattn']) == 1 and cond['c_crossattn'][0] is None:
            cond_txt = None
        else:
            cond_txt = torch.cat(cond['c_crossattn'], 1)

        # one_ch_feat, volume_emb = self.feat_extract_net(cond['c_concat'][0], volume_enconding=cond['meta'])
        volume_emb = torch.concat([timestep_embedding(cond['meta'][:, 0], self.model.model_channels, repeat_only=False),
                                    timestep_embedding(cond['meta'][:, 1], self.model.model_channels, repeat_only=False),
                                    timestep_embedding(cond['meta'][:, 2], self.model.model_channels, repeat_only=False),
                                    timestep_embedding(cond['meta'][:, 3], self.model.model_channels, repeat_only=False)], axis=1)
        
        if test_cache is None:
            controls = self.upper_branch(self.model, cond, volume_emb)
        else:
            controls = test_cache
        controls = [c.detach() for c in controls]

        for i in range(len(controls)):
            controls[i] = self.feat_extract_net(controls[i], volume_enconding=cond['meta'], context=cond_txt)

        inpainted, _ = self.model(x=cond['attn_input'], timesteps=None, volume_enconding=volume_emb, context=cond_txt, control=controls)

        return inpainted

    @torch.no_grad()
    def log_images(self, batch, N=4, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, **kwargs):

        log = dict()
        z, c = self.get_input(batch, self.target_key, bs=N)
        c_cat = c["c_concat"][0][:N]

        c['meta'] = c['meta'][:N]
        c['attn_input'] = c['attn_input'][:N]
        log["input"] = c["attn_input"]
        log["target"] = z[:N]

        if sample:
            log["results"] = self.apply_model(cond={"c_concat": [c_cat], "c_crossattn": [None], "meta": c['meta'], "attn_input": c['attn_input']})

        return log

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def inference(self, batch, cond, ddim_step=50):
        b, c, x, y, z = batch.shape
        assert c == 1, f"c = {c} should be 1"
        
        return self.apply_model(cond)