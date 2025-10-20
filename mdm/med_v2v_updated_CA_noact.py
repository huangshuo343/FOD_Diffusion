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

from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock, TimestepBlock
from ldm.modules.attention import SpatialTransformer, MyCrossAttention


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, volume_enconding=None, context=None, control=None, **kwargs):
        hs = []
        return_controls = []

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
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
        
        self.act = nn.ReLU()

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


class v2vLDM(DDPM):
    """
    training_step(ddpm.DDPM) -> shared_step(self) -> get_input(self) -> forward(self) -> p_losses(self) -> apply_model(self) -> 
    """

    def __init__(self, feat_extract_stage_config, control_key, mask_key="", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(kwargs)

        self.feat_extract_net = instantiate_from_config(feat_extract_stage_config)
        self.control_key = control_key
        self.mask_key = mask_key

        self.clip_denoised = False  # adapt from LatentDiffusion

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.target_key)
        loss = self(x, c)
        return loss

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x = super().get_input(batch, self.target_key, *args, **kwargs)
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
        t = torch.randint(0, self.num_timesteps,
                          (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        elif self.parameterization == "score":
            target = self.get_score(noise, t)
        else:
            raise NotImplementedError()

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

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        if len(cond['c_crossattn']) == 1 and cond['c_crossattn'][0] is None:
            cond_txt = None
        else:
            cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, volume_enconding=cond['meta'], context=cond_txt)
        else:
            # one_ch_feat, volume_emb = self.feat_extract_net(cond['c_concat'][0], volume_enconding=cond['meta'])
            # control_input = torch.cat([x_noisy, one_ch_feat], 1)
            volume_emb = torch.concat([timestep_embedding(cond['meta'][:, 0], diffusion_model.model_channels, repeat_only=False),
                                        timestep_embedding(cond['meta'][:, 1], diffusion_model.model_channels, repeat_only=False),
                                        timestep_embedding(cond['meta'][:, 2], diffusion_model.model_channels, repeat_only=False),
                                        timestep_embedding(cond['meta'][:, 3], diffusion_model.model_channels, repeat_only=False)], axis=1)
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
                tmp_input = torch.cat([vols_5[:, i:i+1, ...], vols_5[:, i:i+1, ...]], 1)
                _, controls = diffusion_model(x=tmp_input, timesteps=torch.zeros_like(t), volume_enconding=volume_emb, context=cond_txt) # TODO: check timesteps = 0 or t
                control_input1.append(controls[0])
                control_input2.append(controls[1])
                control_input3.append(controls[2])
            control_input1 = torch.stack(control_input1, dim=1)
            control_input2 = torch.stack(control_input2, dim=1)
            control_input3 = torch.stack(control_input3, dim=1)
            controls = [control_input1, control_input2, control_input3]
            del control_input1, control_input2, control_input3

            # _, controls = diffusion_model(x=control_input, timesteps=t, volume_enconding=volume_emb, context=cond_txt)
            controls = [c.detach() for c in controls]

            for i in range(len(controls)):
                controls[i] = self.feat_extract_net(controls[i], volume_enconding=cond['meta'], context=cond_txt)

            attn_input = cond['attn_input']
            x_noisy = torch.cat([x_noisy, attn_input], 1)
            eps, _ = diffusion_model(x=x_noisy, timesteps=t, volume_enconding=volume_emb, context=cond_txt, control=controls)

        return eps

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
            # get denoise row
            samples, z_denoise_row = self.sample_log(z[:N],
                                                     cond={"c_concat": [c_cat], "c_crossattn": [None], "meta": c['meta'], "attn_input": c['attn_input']},
                                                     batch_size=N,
                                                     ddim_steps=ddim_steps,
                                                     eta=ddim_eta)

            log["results"] = samples

        return log

    @torch.no_grad()
    def sample_log(self, X, cond, batch_size, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, x, y, z = X.shape
        # shape = (self.channels, h // 8, w // 8)
        assert b == batch_size
        shape = (c, x, y, z)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, batch_size, shape, cond, verbose=False, log_every_t=self.log_every_t, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.parameters())

        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def p_mean_variance(self, x, t, cond, clip_denoised: bool):
        """
        Get the distribution p(x_{t-1} | x_t).
        """
        model_out = self.apply_model(x, t, cond=cond)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x[:,0:1,...], t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == "v":
            x_recon = self.predict_start_from_z_and_v(x, t=t, v=model_out)
        elif self.parameterization == "score":
            x_recon = self.predict_start_from_z_and_score(x, t=t, score=model_out)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x[:,0:1,...], t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, X, t, cond, clip_denoised=True, repeat_noise=False):
        """
        Sample from the model p(x_{t-1} | x_t).
        param x: x_t
        """
        b, c, x, y, z = X.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=X, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = torch.randn((b, 1, x, y, z), device=X.device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(X.shape) - 1)))
        # log is for numerical stability, large variance can be not that large in log space, small variance near to zero can be not that small in log space
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

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
        #mask = cond['mask']
    
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond=cond,
                                clip_denoised=self.clip_denoised)
            #img = img * mask + batch * (1 - mask)
        
        return img