import einops
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.models.diffusion.ddpm import DDPM
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler

class i2iLDM(DDPM):
    """
    training_step(ddpm.DDPM) -> shared_step(self) -> get_input(self) -> forward(self) -> p_losses(self) -> apply_model(self) -> 
    """

    def __init__(self, control_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.control_key = control_key

        self.clip_denoised = False # adapt from LatentDiffusion

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.target_key)
        loss = self(x, c)
        return loss
    
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x = super().get_input(batch, self.target_key, *args, **kwargs)
        x = x.to(self.device)

        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, dict(c_crossattn=[None], c_concat=[control])

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, *args, **kwargs)

    def consistency_loss(self, x, x_t_minus_1, alpha=50):
        # short term consistency loss
        loss = 0.0
        mid = x.shape[1] // 2
        for i in range(x.shape[1]):
            if i == mid:
                continue

            loss += (x[:,i,...] - x_t_minus_1[:,0,...]).abs()
        return torch.mean(loss, dim=(1,2))

    def p_losses(self, x_start, cond, t, noise=None):
        mid = x_start.shape[1] // 2

        noise = default(noise, lambda: torch.randn_like(x_start[:, mid:mid+1, ...]))
        x_noisy = self.q_sample(x_start=x_start[:, mid:mid+1, ...], t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start[:, mid:mid+1, ...]
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start[:, mid:mid+1, ...], noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        
        if mid != 0:
            if self.parameterization == "x0":
                x_0 = model_output
            elif self.parameterization == "eps":
                x_0 = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
                # x_0.clamp_(-1., 1.)
            else:
                raise NotImplementedError()
            loss_consistency = self.consistency_loss(x_start, x_0).mean()
            # t_tmp = t - 1
            # t_tmp[t_tmp < 0] = 0
            # loss_consistency = self.consistency_loss(self.q_sample(x_start=x_start, t=t_tmp, noise=noise), cond['c_concat'][0], x_t_minus_1)
            loss_dict.update({f'{prefix}/loss_consistency': loss_consistency})
        else:
            loss_consistency = 0.0

        loss = loss_simple.mean() + loss_consistency

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
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt)
        else:
            # alternate = False
            # if x_noisy.shape[1] == cond['c_concat'][0].shape[1] and x_noisy.shape[1] > 1:
            #     alternate = True
            x_noisy = torch.cat([x_noisy, cond['c_concat'][0]], 1) # concat in channel dim
            # if alternate:
            #     half_n = x_noisy.shape[1] // 2
            #     indices = torch.tensor([[i, i+half_n] for i in range(half_n)]).flatten()
            #     x_noisy = x_noisy[:, indices, :, :]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt)

        return eps

    @torch.no_grad()
    def log_images(self, batch, N=4, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, **kwargs):

        log = dict()
        z, c = self.get_input(batch, self.target_key, bs=N)
        mid = z.shape[1] // 2
        z = z[:, mid:mid+1, ...]

        c_cat = c["c_concat"][0][:N]

        log["reconstruction"] = z
        log["control"] = c_cat * 2.0 - 1.0 # there will be rescale in logger

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(z[:N], 
                                                     cond={"c_concat": [c_cat], "c_crossattn": [None]},
                                                     batch_size=N, 
                                                     ddim_steps=ddim_steps, 
                                                     eta=ddim_eta)

            log["samples"] = samples
            if plot_denoise_rows:
                from tqdm import tqdm
                
                def _get_denoise_row_from_list(samples, desc=''):
                    denoise_row = []
                    for zd in tqdm(samples, desc=desc):
                        denoise_row.append(zd)
                    n_imgs_per_row = len(denoise_row)
                    denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
                    denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
                    denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
                    denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
                    return denoise_grid
                
                denoise_grid = _get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    @torch.no_grad()
    def sample_log(self, X, cond, batch_size, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = X.shape
        # shape = (self.channels, h // 8, w // 8)
        assert b == batch_size
        shape = (c, h, w)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, log_every_t=self.log_every_t, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.parameters())
        
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def p_mean_variance(self, x, t, clip_denoised: bool):
        """
        Get the distribution p(x_{t-1} | x_t).
        """
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x[:,0:1,...], t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x[:,0:1,...], t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        """
        Sample from the model p(x_{t-1} | x_t).
        param x: x_t
        """
        b, c, h, w = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn((b, 1, h, w), device=x.device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # log is for numerical stability, large variance can be not that large in log space, small variance near to zero can be not that small in log space
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    def inference(self, batch, ddim_step=50, channel=1):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = batch.shape

        shape = (channel, h, w) # shape of the output instead of the control
        samples, _ = ddim_sampler.sample(ddim_step, b, shape, {"c_concat": [batch], "c_crossattn": [None]}, verbose=False, log_every_t=self.log_every_t)
        return samples

        # b, c, h, w = batch.shape
        # device = self.betas.device
        
        # img = torch.randn((b, 1, h, w ), device=device)
        # intermediate = [img]
    
        # for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
        #     img = torch.cat([img, batch], 1)
        #     assert img.shape[1] == c + 1, f"img.shape[1] = {img.shape[1]}, c = {c}"
        #     img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
        #                         clip_denoised=self.clip_denoised)
        #     intermediate.append(img)
        
        
        
        # # save intermediate as grid
        # intermediate = torch.stack(intermediate, dim=0)
        # # only take b=0
        # intermediate = intermediate[:,0,...]
        # intermediate.clamp_(-1., 1.)
        # intermediate = intermediate.detach().cpu()
        # intermediate = (intermediate + 1.0) / 2.0 * 255.0
        # # save intermediate as grid
        # grid = make_grid(intermediate, nrow=30)
        # grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        # grid = grid.numpy().astype(np.uint8)
        # Image.fromarray(grid).save('./trash/intermediate.png')

        # return img