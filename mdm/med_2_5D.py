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
from torch.optim.lr_scheduler import LambdaLR

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

    def consistency_loss(self, x_0):
        # short term consistency loss
        loss = (x_0[:,0,...] - x_0[:,1,...]).abs()
        loss += (x_0[:,2,...] - x_0[:,1,...]).abs()
        return torch.mean(loss, dim=(1,2))

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
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        
        # if self.parameterization == "x0":
        #     x_0 = model_output
        # elif self.parameterization == "eps":
        #     x_0 = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)
        #     # x_0.clamp_(-1., 1.)
        # else:
        #     raise NotImplementedError()
        
        # loss_consistency = self.consistency_loss(x_0).mean()
        # loss_dict.update({f'{prefix}/loss_consistency': loss_consistency})

        loss = loss_simple.mean()# + loss_consistency

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
            x_noisy = torch.cat([x_noisy, cond['c_concat'][0]], 1) # concat in channel dim
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt)

        return eps

    @torch.no_grad()
    def log_images(self, batch, N=4, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, **kwargs):

        log = dict()
        z, c = self.get_input(batch, self.target_key, bs=N)
        
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
        # return opt

        # Define the lambda function for the learning rate scheduler
        warm_up_steps = 10000
        
        def lr_lambda(current_step):
            # During warm-up
            if current_step < warm_up_steps:
                return 1.e-6 + (1 - 1.e-6) * (current_step / warm_up_steps)
            # After warm-up
            return 1.0  # Keep the learning rate constant
        
        scheduler = LambdaLR(opt, lr_lambda)
        
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def inference(self, batch, ddim_step=50, channel=1, x0=None):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = batch.shape

        shape = (channel, h, w) # shape of the output instead of the control
        samples, intermediates = ddim_sampler.sample(ddim_step, b, shape, {"c_concat": [batch], "c_crossattn": [None]}, verbose=False, log_every_t=1, hack_with_x0={'x0': x0} if x0 is not None else None)

        # List to collect grids for each batch
        grid_list = []
        intermediate = torch.stack(intermediates['x0'], dim=0)

        # Loop through all batches
        for b in range(intermediate.size(1)):
            intermediate_batch = intermediate[:, b, ...]
            intermediate_batch.clamp_(-1., 1.)
            intermediate_batch = intermediate_batch.detach().cpu()
            intermediate_batch = (intermediate_batch + 1.0) / 2.0 * 255.0
            # save intermediate as grid
            grid = make_grid(intermediate_batch, nrow=30)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy().astype(np.uint8)
            grid_list.append(grid)

        # Concatenate all grids vertically
        concatenated_grid = np.vstack(grid_list)

        # Save the concatenated grid
        Image.fromarray(concatenated_grid).save('./intermediate_x0.png')

        # List to collect grids for each batch
        grid_list = []
        intermediate = torch.stack(intermediates['minus_v'], dim=0)

        # Loop through all batches
        for b in range(intermediate.size(1)):
            intermediate_batch = intermediate[:, b, ...]
            intermediate_batch.clamp_(-1., 1.)
            intermediate_batch = intermediate_batch.detach().cpu()
            intermediate_batch = (intermediate_batch + 1.0) / 2.0 * 255.0
            # save intermediate as grid
            grid = make_grid(intermediate_batch, nrow=30)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy().astype(np.uint8)
            grid_list.append(grid)

        # Concatenate all grids vertically
        concatenated_grid = np.vstack(grid_list)

        # Save the concatenated grid
        Image.fromarray(concatenated_grid).save('./intermediate_minus_v.png')

        return samples