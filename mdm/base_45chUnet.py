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
import pytorch_lightning as pl


class MultiChUnet(pl.LightningModule):
    """
    training_step(ddpm.DDPM) -> shared_step(self) -> get_input(self) -> forward(self) -> p_losses(self) -> apply_model(self) -> 
    """

    def __init__(self, unet_config, dims, mask_key, loss_type, target_key, control_key, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(kwargs)

        self.dims = dims
        self.mask_key = mask_key
        self.loss_type = loss_type
        self.target_key = target_key
        self.control_key = control_key

        self.model = instantiate_from_config(unet_config)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.target_key)
        loss = self(x, c)
        return loss

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
        mask = mask.to(self.device)
        mask = einops.rearrange(mask, 'b x y z c -> b c x y z')
        mask = mask.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[None], c_concat=[control], mask=mask, meta=meta, attn_input=attn_input)

    def forward(self, x, c, *args, **kwargs):
        return self.p_losses(x, c, *args, **kwargs)

    def p_losses(self, x, cond):
        model_output = self.apply_model(cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        mask = cond['mask']
        #print(mask.shape)
        mask_shape = mask.shape
        target = cond['c_concat'][0]
        # model_output = torch.randn_like(target, requires_grad=True, device=self.device)
        target = torch.randn_like(model_output, device=self.device)
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

    def apply_model(self, cond, test_cache=None, *args, **kwargs):
        assert isinstance(cond, dict)

        x = cond['c_concat'][0] * cond['mask']
        inpainted = self.model(x=x)

        return inpainted

    @torch.no_grad()
    def log_images(self, batch, N=4, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, **kwargs):
        log = dict()
        z, c = self.get_input(batch, self.target_key, bs=N)
        c_cat = c["c_concat"][0][:N]

        c['meta'] = c['meta'][:N]
        c['attn_input'] = c['attn_input'][:N]
        c['mask'] = c['mask'][:N]
        log["input"] = c["attn_input"]
        log["target"] = z[:N]

        if sample:
            log["results"] = self.apply_model(cond={"c_concat": [c_cat], "c_crossattn": [None], "meta": c['meta'], "attn_input": c['attn_input'], "mask": c['mask']})
            # only see L0
            log["results"] = log["results"][:, 0:1, ...]

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
        # ddim_sampler = DDIMSampler(self)
        # b, c, x, y, z = batch.shape

        # shape = (c, x, y, z) # shape of the output instead of the control
        # samples, _ = ddim_sampler.sample(ddim_step, b, shape, cond, verbose=False, log_every_t=self.log_every_t)
        # return samples
    
        b, c, x, y, z = batch.shape
        assert c == 1, f"c = {c} should be 1"
        device = self.betas.device
        
        img = torch.randn((b, c, x, y, z ), device=device)

        # volume_emb = torch.concat([timestep_embedding(cond['meta'][:, 0], self.model.diffusion_model.model_channels, repeat_only=False),
        #                             timestep_embedding(cond['meta'][:, 1], self.model.diffusion_model.model_channels, repeat_only=False),
        #                             timestep_embedding(cond['meta'][:, 2], self.model.diffusion_model.model_channels, repeat_only=False),
        #                             timestep_embedding(cond['meta'][:, 3], self.model.diffusion_model.model_channels, repeat_only=False)], axis=1)
        
        # test_cache = self.upper_branch(self.model.diffusion_model, torch.zeros((b,), device=device, dtype=torch.long), cond, volume_emb)
    
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond=cond,
                                clip_denoised=self.clip_denoised, test_cache=None)
        
        return img