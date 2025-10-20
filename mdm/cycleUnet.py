import einops
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from einops import rearrange, repeat
import pytorch_lightning as pl
from ldm.models.diffusion.ddpm import DiffusionWrapper

class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        
        # Use the VGG16 model pre-trained on ImageNet
        vgg_pretrained = models.vgg16(pretrained=True)
        
        # Use the relu2_2 layer for perceptual loss computation
        self.vgg = nn.Sequential(*list(vgg_pretrained.features)[:9])
        
        # We only need to do a forward pass, so we don't want to compute gradients
        for param in self.vgg.parameters():
            param.requires_grad = requires_grad
        
        # Move to the appropriate device
        self.vgg = self.vgg.cuda()
        
        # L1 loss for comparing feature maps
        self.loss = nn.L1Loss()

        imagenet_mean = [0.456]
        imagenet_std = [0.224]
        self.normalize = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)

        x = self.normalize(x)
        y = self.normalize(y)

        # Forward pass on the VGG network
        x_features = self.vgg(x)
        y_features = self.vgg(y)
        
        # Compute the L1 loss between the feature maps
        return self.loss(x_features, y_features)

class CycleUnet(pl.LightningModule):
    """
    training_step(ddpm.DDPM) -> shared_step(self) -> get_input(self) -> forward(self) -> p_losses(self) -> apply_model(self) -> 
    """

    def __init__(self, unet_config, target_key, control_key, dims=2, loss_type='l1'):
        super().__init__()
        self.save_hyperparameters({
            'unet_config': unet_config,
            'target_key': target_key,
            'control_key': control_key,
            'dims': dims,
            'loss_type': loss_type
        })

        self.dims = dims
        self.loss_type = loss_type
        self.control_key = control_key
        self.target_key = target_key

        self.mr2ct = DiffusionWrapper(unet_config)
        self.ct2mr = DiffusionWrapper(unet_config)

        self.perceptual_loss = PerceptualLoss()

        self.clip_denoised = False # adapt from LatentDiffusion

    def shared_step(self, batch, **kwargs):
        ct, mr = self.get_input(batch, self.target_key)
        loss = self(ct, mr)
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

    def forward(self, ct, mr, *args, **kwargs):
        return self.p_losses(ct, mr, *args, **kwargs)

    def p_losses(self, ct, mr):
        ct_pred = self.apply_model(self.mr2ct, mr)
        mr_pred = self.apply_model(self.ct2mr, ct)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # simple loss
        lambda_simple = 1.0

        loss_mr2ct = lambda_simple * self.get_loss(ct_pred, ct, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_mr2ct': loss_mr2ct.mean()})
        loss_ct2mr = lambda_simple * self.get_loss(mr_pred, mr, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_ct2mr': loss_ct2mr.mean()})

        # cycle loss
        lambda_cycle = 1.0

        loss_mr_recon = lambda_cycle * self.get_loss(self.apply_model(self.ct2mr, ct_pred), mr, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_mr_recon': loss_mr_recon.mean()})
        loss_ct_recon = lambda_cycle * self.get_loss(self.apply_model(self.mr2ct, mr_pred), ct, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_ct_recon': loss_ct_recon.mean()})

        # # consistency loss
        # lambda_consistency = 10.0

        # loss_pred_mr_consistency = lambda_consistency * self.get_loss(mr_pred[:,0:1,...], mr_pred[:,1:2,...], mean=False).mean([1, 2, 3])
        # loss_pred_mr_consistency += lambda_consistency * self.get_loss(mr_pred[:,2:3,...], mr_pred[:,1:2,...], mean=False).mean([1, 2, 3])
        # loss_dict.update({f'{prefix}/loss_mr_consis': loss_pred_mr_consistency.mean()})

        # loss_pred_ct_consistency = lambda_consistency * self.get_loss(ct_pred[:,0:1,...], ct_pred[:,1:2,...], mean=False).mean([1, 2, 3])
        # loss_pred_ct_consistency += lambda_consistency * self.get_loss(ct_pred[:,2:3,...], ct_pred[:,1:2,...], mean=False).mean([1, 2, 3])
        # loss_dict.update({f'{prefix}/loss_ct_consis': loss_pred_ct_consistency.mean()})

        # perceptual loss
        lambda_perceptual = 1.0
        ct_perceptual = lambda_perceptual * self.perceptual_loss(ct_pred, ct)
        loss_dict.update({f'{prefix}/loss_ct_perc': ct_perceptual})
        mr_perceptual = lambda_perceptual * self.perceptual_loss(mr_pred, mr)
        loss_dict.update({f'{prefix}/loss_mr_perc': mr_perceptual})

        # total loss
        loss = loss_mr2ct.mean() + loss_ct2mr.mean() + loss_mr_recon.mean() + loss_ct_recon.mean()# + ct_perceptual + mr_perceptual# + loss_pred_mr_consistency.mean() + loss_pred_ct_consistency.mean()

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
        ct, mr = self.get_input(batch, self.target_key, bs=N)

        log["reconstruction"] = ct * 2.0 - 1.0 # there will be rescale in logger
        log["control"] = mr * 2.0 - 1.0 # there will be rescale in logger

        generated_ct = self.mr2ct.diffusion_model(x=mr, context=None)
        generated_mr = self.ct2mr.diffusion_model(x=ct, context=None)
        
        log["samples_ct"] = generated_ct * 2.0 - 1.0
        log["samples_mr"] = generated_mr * 2.0 - 1.0

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
        params = list(self.mr2ct.diffusion_model.parameters()) + list(self.ct2mr.diffusion_model.parameters())
        
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    
    def inference(self, batch, ddim_step=50, channel=1):
        ct, mr = self.get_input(batch, self.target_key, bs=1)
        ct_pred = self.apply_model(self.mr2ct, mr)
        return ct_pred