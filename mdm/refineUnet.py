import einops
import torch

from einops import rearrange, repeat
import pytorch_lightning as pl
from ldm.models.diffusion.ddpm import DiffusionWrapper

class RefineUnet(pl.LightningModule):
    """
    training_step(ddpm.DDPM) -> shared_step(self) -> get_input(self) -> forward(self) -> p_losses(self) -> apply_model(self) -> 
    """

    def __init__(self, unet_config, target_key, control_key, dims=2, loss_type='l1'):
        super().__init__()
        self.save_hyperparameters()

        self.dims = dims
        self.loss_type = loss_type
        self.control_key = control_key
        self.target_key = target_key

        self.refine_net = DiffusionWrapper(unet_config)

        self.clip_denoised = False # adapt from LatentDiffusion

    def shared_step(self, batch, **kwargs):
        ct, sct = self.get_input(batch, self.target_key)
        loss = self(ct, sct)
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
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        return x, control

    def forward(self, ct, sct, *args, **kwargs):
        return self.p_losses(ct, sct, *args, **kwargs)

    def p_losses(self, ct, sct):
        ct_pred = self.apply_model(self.refine_net, sct)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # simple loss
        lambda_simple = 1.0

        loss_sct2ct = lambda_simple * self.get_loss(ct_pred, ct, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_sct2ct': loss_sct2ct.mean()})

        # total loss
        loss = loss_sct2ct.mean()

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

    def apply_model(self, model, input_, *args, **kwargs):
        diffusion_model = model.diffusion_model

        ct_pred = diffusion_model(x=input_, context=None)

        return ct_pred

    @torch.no_grad()
    def log_images(self, batch, N=4, sample=False, ddim_steps=50, ddim_eta=0.0, plot_denoise_rows=False, **kwargs):

        log = dict()
        ct, sct = self.get_input(batch, self.target_key, bs=N)

        log["ct"] = ct * 2.0 - 1.0 # there will be rescale in logger
        log["sct"] = sct[:,:3,...] * 2.0 - 1.0 # there will be rescale in logger

        generated_ct = self.refine_net.diffusion_model(x=sct, context=None)
        
        log["pred"] = generated_ct * 2.0 - 1.0

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
        opt = torch.optim.AdamW(self.refine_net.diffusion_model.parameters(), lr=lr)
        return opt
    
    def inference(self, batch):
        ct_pred = self.apply_model(self.refine_net, batch)
        return ct_pred