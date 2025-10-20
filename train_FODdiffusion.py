# import os
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

from share import *

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mdm.logger import ImageLogger, SaveAdditionalContentCallback
from mdm.model import create_model, create_dataset, load_state_dict, load_config
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
# import torch

# CUDA_VISIBLE_DEVICES=3 python train_FODdiffusion.py --cfg cfgs/train/train_AE3D_orderwise_originalrange.yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MedDiffusion Training Script")
    parser.add_argument("--cfg", type=str, required=True, help="training config path")
    args = parser.parse_args()

    cfg = load_config(args.cfg)

    # Training settings
    resume_path = cfg.training.resume_path
    batch_size = cfg.training.batch_size
    learning_rate = cfg.training.learning_rate

    gpus = cfg.training.gpus
    max_steps = cfg.training.max_steps
    accumulate_grad_batches = cfg.training.accumulate_grad_batches

    # Create model: First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(cfg.model, resume_path).cpu()
    # torch.save(model.state_dict(), '/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/distortion_painting/lightning_logs/version_23/model.pt')
    # exit(1)
    model.learning_rate = learning_rate

    # Create dataset and dataloader
    dataset = create_dataset(cfg.dataset)
    if cfg.val_dataset is not None:
        val_dataset = create_dataset(cfg.val_dataset)

    # Create logger parameters
    logger_params = cfg.logger

    # Create logger
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    if cfg.val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False)
    
    imgLogger = ImageLogger(log_images_kwargs={'sample': True, 'N': min(cfg.training.batch_size, 4)}, **logger_params)
    cfgLogger = SaveAdditionalContentCallback(cfg=cfg)
    logger = TensorBoardLogger("logs", name=cfg.logger.experiment_name, version=imgLogger.split_name)

    best_model_checkpoint = ModelCheckpoint(
        monitor='val/loss',
        #dirpath='basefodunet/checkpoints/',
        filename='best-model-epoch={epoch}-step={global_step}',
        save_top_k=1,  # Only save the top 1 model
        mode='min',  # `min` for metrics like loss, `max` for metrics like accuracy
        verbose=True,
    )

    # Callback to save the latest model at the end of every epoch
    latest_model_checkpoint = ModelCheckpoint(
        #dirpath='basefodunet/checkpoints/',
        filename='latest-model-epoch={epoch}-step={global_step}',
        save_top_k=1,  # Set to -1 to save all epochs
        every_n_epochs=1,  # Save every epoch
        save_on_train_epoch_end=True,  # Save at the end of the training epoch
    )

    trainer = pl.Trainer(
        gpus=gpus, 
        precision=32, 
        callbacks=[imgLogger, cfgLogger, best_model_checkpoint, latest_model_checkpoint], 
        max_steps=max_steps, 
        accumulate_grad_batches=accumulate_grad_batches, 
        plugins=DDPPlugin(find_unused_parameters=False) if gpus > 1 else None,
        logger=logger
    )

    # Train!
    trainer.fit(model, dataloader, val_dataloader if cfg.val_dataset is not None else None)
