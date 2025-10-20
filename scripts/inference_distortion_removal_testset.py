# %%
import os
import time
import torch
import argparse
import einops
import nibabel as nib
from torch.utils.data import DataLoader
from mdm.model import create_model, create_dataset, load_config
import utils.utils as utils
import numpy as np

if __name__ == '__main__':
    # test_model: CUDA_VISIBLE_DEVICES=2 python -m scripts.inference_distortion_removal_testset --model /ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/distortion_painting/logs/FOD_distortion_removal/train_2024-01-19_16-46-57/checkpoints/epoch=192-step=97465.ckpt --config /ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/distortion_painting/cfgs/train/train_distortion_painting.yaml
    
    parser = argparse.ArgumentParser(description="FOD distortion removal inference")
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--config", type=str, required=True, help="config path")

    args = parser.parse_args()

    resume_path = args.model
    model_config = args.config
    device = 'cuda'

    assert device in ['cpu', 'cuda'], 'Unknown device: {}'.format(device)
    device = torch.device(device)

    model_config = load_config(model_config)

    dataset = create_dataset(model_config.test_dataset)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=15, shuffle=False)

    model = create_model(model_config.model, resume_path).cpu()#.half()
    model = model.to(device)
    model.eval()

    #print(dataloader)

    # load data
    volume_idx = 640 * 45 #36 * 45 0 600 * 45 350 * 45 262 * 45 0 320 * 45
    for batch in dataloader:        
        start = time.time()
        # x is target, only shape is used during inference
        # mask is only used during training for loss calculation
        x, control, mask, meta, attn_input = batch['jpg'], batch['hint'], batch['mask'], batch['meta'], batch['attn_input']
        x = x.to(device)
        x = einops.rearrange(x, 'b x y z c -> b c x y z')
        x = x.to(memory_format=torch.contiguous_format).float()

        meta = meta.to(device)

        control = control.to(device)
        control = einops.rearrange(control, 'b x y z c -> b c x y z')
        control = control.to(memory_format=torch.contiguous_format).float()

        attn_input = attn_input.to(device)
        attn_input = einops.rearrange(attn_input, 'b x y z c -> b c x y z')
        attn_input = attn_input.to(memory_format=torch.contiguous_format).float()

        #mask = mask.to(device)
        #mask = einops.rearrange(mask, 'b x y z c -> b c x y z')
        #mask = mask.to(memory_format=torch.contiguous_format).float()

        cond = {"c_concat": [control], "c_crossattn": [None], "meta": meta, "attn_input": attn_input}#, 'mask': mask
        # run 10 times to get the average
        with torch.no_grad():
            pred = model.inference(x, cond, 150)
    
        end = time.time()
        print('LDM inference time: ', end - start)

        print('pred shape: ', pred.shape)

        pred = einops.rearrange(pred, 'b c x y z -> b x y z c')
        x = einops.rearrange(x, 'b c x y z -> b x y z c')
        
        bs = pred.shape[0]
        for i in range(bs):
            output_path = f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/distortion_brainsteam/UKBiobank_anotherregion/results_front_evaluate/data{volume_idx//45}_vol{volume_idx%45}_pred.nii.gz'
            reconstucted = pred[i].cpu().numpy().astype('float32')
            reconstucted = nib.Nifti1Image(reconstucted, affine=np.eye(4))
            nib.save(reconstucted, output_path)

            output_path = f'/ifs/loni/faculty/shi/spectrum/Student_2020/huangshuo/distortion_brainsteam/UKBiobank_anotherregion/results_front_evaluate/data{volume_idx//45}_vol{volume_idx%45}_ori.nii.gz'
            ori = x[i].cpu().numpy().astype('float32')
            ori = nib.Nifti1Image(ori, affine=np.eye(4))
            nib.save(ori, output_path)

            volume_idx += 45#1