import os

import numpy as np
import torch
import torchvision
from PIL import Image
import json
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
from einops import rearrange

from datetime import datetime


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, dataset_config='2d',
                 log_images_kwargs=None, **kwargs):
        super().__init__()

        assert dataset_config in ['2d', '3d', '3d_3ch', '3d_45ch', '2.5d', '2.5d3ch',
                                  'refine', 'refine2'], 'Unknown dataset config: {}'.format(dataset_config)

        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

        self.start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.split_name = "train_" + self.start_time
        self.dataset_config = dataset_config

    @rank_zero_only
    def log_local(self, save_dir, name, version, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, name, version, "image_log")
        for k in images:
            if self.dataset_config == '3d':
                data = []
                # in each dimension, extract 16 slices evenly spaced between 0 to 64
                data.append(rearrange(images[k][:, :, np.linspace(
                    1, 33, 8, endpoint=False, dtype=int), :, :], 'b c n h w -> (b n) c h w'))
                #data.append(rearrange(images[k][:, :, np.linspace(
                #    1, 33, 8, endpoint=False, dtype=int), :, :], 'b c (n p) h w -> (b c n) p h w', p=1))
                # data.append(rearrange(images[k][:, :, :, np.linspace(
                #     32, 256, 8, endpoint=False, dtype=int), :], 'b c h n w -> (b n) c h w'))
                # data.append(rearrange(images[k][:, :, :, :, np.linspace(
                #     32, 256, 8, endpoint=False, dtype=int)], 'b c h w n -> (b n) c h w'))
                data = torch.cat(data, dim=0)
                nrow = 8
            elif self.dataset_config == '3d_3ch':
                data = []
                data.append(rearrange(images[k], 'b c n h w -> (b c) n h w', c=1))
                data = torch.cat(data, dim=0)
                nrow = 8
            elif self.dataset_config == '3d_45ch':
                data = rearrange(images[k][:, :, np.linspace(
                    1, 33, 8, endpoint=False, dtype=int), :, :], 'b c n h w -> (b n c) h w')
                data = data[:, None, ...]
                nrow = 45
            else:
                data = images[k]
                nrow = 4

            grid = torchvision.utils.make_grid(data, nrow=nrow)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)

            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            # filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
            #     k, global_step, current_epoch, batch_idx)
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(global_step, current_epoch, batch_idx, k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                # this is the function that defined in TauPETAD2D.py
                images = pl_module.log_images(batch, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            # pl_module.logger.name == trainer.logger.name, pl_module.logger.version == trainer.logger.version
            self.log_local(pl_module.logger.save_dir, pl_module.logger.name, pl_module.logger.version, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx)


class SaveAdditionalContentCallback(Callback):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        save_dir = trainer.logger.save_dir

        if save_dir:
            version_path = os.path.join(save_dir, trainer.logger.name, trainer.logger.version)

            # Create or save the content you wish
            cfg_path = os.path.join(version_path, 'cfg.yaml')

            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            with open(cfg_path, 'w') as f:
                f.write(json.dumps(cfg_dict, indent=4))

            model_path = os.path.join(version_path, f'model.txt')
            with open(model_path, 'w') as f:
                f.write(str(pl_module))

            print(f"Saved additional content to: {cfg_path}")
        else:
            print("Logger's save directory is not defined.")
