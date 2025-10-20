import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def load_config(path):
    cfg = OmegaConf.load(path)
    if '_base_' in cfg:
        for base in cfg['_base_']:
            cfg = OmegaConf.merge(load_config(base), cfg)
            print(f'Loaded base config from [{base}]')
        del cfg['_base_']
    return cfg


def create_dataset(cfg_dataset):
    dataset = instantiate_from_config(cfg_dataset)
    return dataset


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(cfg_model, resume_path):
    model = instantiate_from_config(cfg_model).cpu()
    if len(resume_path) > 0:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    return model
