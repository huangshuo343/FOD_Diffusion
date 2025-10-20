import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from ldm.modules.diffusionmodules.model import Encoder, Decoder, Encoder3D, Decoder3D

from ldm.util import instantiate_from_config

class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 learn_logvar=False
                 ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+postfix)

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val"+postfix)

        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
            self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list,
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec

        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class AutoencoderKL3D(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 encoder_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 learn_logvar=False,
                 use_discriminator=False
                 ):
        super().__init__()
        self.learn_logvar = learn_logvar
        self.image_key = image_key
        if encoder_config is None:
            self.encoder = Encoder3D(**ddconfig)
            assert ddconfig["double_z"]
            self.quant_conv = torch.nn.Conv3d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        else:
            self.encoder = instantiate_from_config(encoder_config)
            self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], 2*embed_dim, 1)
        self.decoder = Decoder3D(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.use_discriminator = use_discriminator
        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        # print("Encoder output shape: {}".format(h.shape))
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # print("Decoder output shape: {}".format(dec.shape))
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 4:
            x = x[..., None]
        
        x = x.permute(0, 4, 1, 2, 3).to(memory_format=torch.contiguous_format).float()

        return x

    def training_step(self, batch, batch_idx, optimizer_idx=-1): # optimizer_idx=-1 is for the case of using only one optimizer
        # print("optimizer_idx: {}".format(optimizer_idx))
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        
        if self.use_discriminator:
            if optimizer_idx == 0:
                # train encoder+decoder+logvar
                aeloss = 0
                log_dict_ae = {}
                for i in range(reconstructions.shape[-1]):
                    cur_aeloss, cur_log_dict_ae = self.loss(inputs[...,i], reconstructions[...,i], posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                    aeloss += cur_aeloss
                    for k in cur_log_dict_ae.keys():
                        if k not in log_dict_ae.keys():
                            log_dict_ae[k] = 0
                        log_dict_ae[k] += cur_log_dict_ae[k]
                aeloss /= reconstructions.shape[-1]
                for k in log_dict_ae.keys():
                    log_dict_ae[k] /= reconstructions.shape[-1]
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                return aeloss

            if optimizer_idx == 1:
                # train the discriminator
                discloss = 0
                log_dict_disc = {}
                for i in range(reconstructions.shape[-1]):
                    cur_discloss, cur_log_dict_disc = self.loss(inputs[...,i], reconstructions[...,i], posterior, optimizer_idx, self.global_step,
                                                        last_layer=self.get_last_layer(), split="train")
                    discloss += cur_discloss
                    for k in cur_log_dict_disc.keys():
                        if k not in log_dict_disc.keys():
                            log_dict_disc[k] = 0
                        log_dict_disc[k] += cur_log_dict_disc[k]
                discloss /= reconstructions.shape[-1]
                for k in log_dict_disc.keys():
                    log_dict_disc[k] /= reconstructions.shape[-1]

                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                return discloss
        else:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

    def validation_step(self, batch, batch_idx):
        # print("validation_step")
        log_dict = self._validation_step(batch, batch_idx)
        # with self.ema_scope():
        #     log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix=""):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if self.use_discriminator:
            aeloss = 0
            log_dict_ae = {}
            discloss = 0
            log_dict_disc = {}
            for i in range(reconstructions.shape[-1]):
                cur_aeloss, cur_log_dict_ae = self.loss(inputs[...,i], reconstructions[...,i], posterior, 0, self.global_step,
                                                    last_layer=self.get_last_layer(), split="val"+postfix)
                cur_discloss, cur_log_dict_disc = self.loss(inputs[...,i], reconstructions[...,i], posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val"+postfix)
                aeloss += cur_aeloss
                discloss += cur_discloss
                for k in cur_log_dict_ae.keys():
                    if k not in log_dict_ae.keys():
                        log_dict_ae[k] = 0
                    log_dict_ae[k] += cur_log_dict_ae[k]
                for k in cur_log_dict_disc.keys():
                    if k not in log_dict_disc.keys():
                        log_dict_disc[k] = 0
                    log_dict_disc[k] += cur_log_dict_disc[k]
            aeloss /= reconstructions.shape[-1]
            discloss /= reconstructions.shape[-1]
            for k in log_dict_ae.keys():
                log_dict_ae[k] /= reconstructions.shape[-1]
            for k in log_dict_disc.keys():
                log_dict_disc[k] /= reconstructions.shape[-1]

            self.log(f"val{postfix}/loss", aeloss)
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return self.log_dict
        else:
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, self.global_step,
                                                last_layer=self.get_last_layer(), split="val"+postfix)
            self.log(f"val{postfix}/loss", aeloss)
            self.log_dict(log_dict_ae)
            return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
            self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: Learning logvar")
            ae_params_list.append(self.loss.logvar)
        opt_ae = torch.optim.Adam(ae_params_list,
                                  lr=lr, betas=(0.5, 0.9))
        
        if self.use_discriminator:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
            return [opt_ae, opt_disc], [] # if only use one optimizer there would be no error on discrminator parameters not used problem
        else:
            return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            # if x.shape[1] > 1:
            #     # grayize with random projection
            #     assert xrec.shape[1] > 1
            #     x = self.to_gray(x)
            #     xrec = self.to_gray(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))[:4]
            log["reconstructions"] = xrec[:4]

        log["inputs"] = x[:4]
        # normalize to [0,1] for visualization
        for k in log.keys():
            log[k] = (log[k] - log[k].min()) / (log[k].max() - log[k].min())
        return log

    def to_gray(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "grayize"):
            self.register_buffer("grayize", torch.randn(1, x.shape[1], 1, 1, 1).to(x))
        x = F.conv2d(x, weight=self.grayize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    def inference(self, x):
        posterior = self.encode(x)
        z = posterior.mode()
        dec = self.decode(z)
        return z, dec


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


import numpy as np
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean