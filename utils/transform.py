# for dataset transform, see MedDiffusion/dataset.py and MedDiffusion/cfgs/dataset/dataset.yaml
import numpy as np


def add_channel_at_last_dim(img):
    return img[..., np.newaxis]


def put_height_at_first_dim(img):
    return np.moveaxis(img, -1, 0)

def put_last_channel_at_first_dim(img):
    return np.moveaxis(img, -1, 0)

def _0_255_to_0_1(img):
    return img / 255.0

def _minus7_7_to_minus1_1(img):
    return img / 7.0

def _minus10_10_to_minus1_1(img):
    return img / 10.0

def _0_255_to_minus1_1(img):
    return img / 127.5 - 1.0


def _0_1_to_minus1_1(img):
    return img * 2.0 - 1.0

def cut_between_0_1(img):
    return np.clip(img, 0, 1)


def cut_between_minus1_1(img):
    return np.clip(img, -1, 1)


def identity_transform(img):
    return img


def cp_midlle_ch_to_side(img):
    return np.concatenate((img[..., 1:2], img[..., 1:2], img[..., 1:2]), axis=-1)


def cp_upper_ch_to_other(img):
    return np.concatenate((img[..., 0:1], img[..., 0:1], img[..., 0:1]), axis=-1)


def cp_lower_ch_to_other(img):
    return np.concatenate((img[..., 2:3], img[..., 2:3], img[..., 2:3]), axis=-1)

# composite transform


def threeD_src_transform(img):
    img = add_channel_at_last_dim(img)
    img = _0_255_to_0_1(img)
    return img


def threeD_tgt_transform(img):
    img = add_channel_at_last_dim(img)
    img = _0_255_to_minus1_1(img)
    return img


def take_middle_channel_transform(img):
    return img[..., 1:2]
