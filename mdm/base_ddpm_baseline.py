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


class v2vLDM(DDPM):
    """
    training_step(ddpm.DDPM) -> shared_step(self) -> get_input(self) -> forward(self) -> p_losses(self) -> apply_model(self) -> 
    """

    def __init__(self, control_key, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.control_key = control_key # deprecated
        self.save_hyperparameters(kwargs)

        self.clip_denoised = False  # adapt from LatentDiffusion

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.target_key)
        loss = self(x, c)
        return loss

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x = super().get_input(batch, self.target_key, *args, **kwargs)
        x = x.to(self.device)

        return x, dict(c_crossattn=[None], c_concat=[None])

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

        loss_simple = self.get_loss(model_output, target, mean=False)
        loss_simple = loss_simple.mean([1, 2, 3, 4])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        loss = loss_simple.mean()

        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        diffusion_model = self.model.diffusion_model

        eps = diffusion_model(x=x_noisy, timesteps=t, volume_enconding=None, context=None)
        return eps

    @torch.no_grad()
    def log_images(self, batch, N=4, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, **kwargs):

        log = dict()
        z, c = self.get_input(batch, self.target_key, bs=N)

        log["target"] = z[:N]

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(z[:N],
                                                     cond=None,
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
        ddim_sampler = DDIMSampler(self)
        b, c, x, y, z = batch.shape

        shape = (c, x, y, z) # shape of the output instead of the control
        samples, _ = ddim_sampler.sample(ddim_step, b, shape, cond, verbose=False, log_every_t=self.log_every_t)
        return samples
    
        # b, c, x, y, z = batch.shape
        # assert c == 1, f"c = {c} should be 1"
        # device = self.betas.device
        
        # img = torch.randn((b, c, x, y, z ), device=device)
    
        # for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
        #     c = {"c_concat": [batch], "c_crossattn": [None]}
        #     img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond=c,
        #                         clip_denoised=self.clip_denoised)
        
        # return img

    def inference_inpainting(self, batch, cond):
        b, c, x, y, z = batch.shape
        print('batch shape: ', batch.shape)
        print('mask shape: ', cond['mask'].shape)
        assert c == 1, f"c = {c} should be 1"
        device = self.betas.device
        
        img = torch.randn((b, c, x, y, z ), device=device)
        mask = cond['mask']
    
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            c = {"c_concat": [None], "c_crossattn": [None]}
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond=c, clip_denoised=self.clip_denoised)
            img = img * mask + batch * (1 - mask)
        
        return img