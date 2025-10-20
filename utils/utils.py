from skimage.util import view_as_windows
from einops import rearrange
import numpy as np

def split(img_data, vol_size, step_size):
    """
    Split the image into patches of size vol_size with step_size stride.
    Args:
        img_data: 3D numpy array, shape (H, W, C)
        vol_size: tuple of 3 ints, (h1, w1, c1)
        step_size: tuple of 3 ints, (h2, w2, c2)
    Returns:
        volumes: 4D numpy array, shape (N, h1, w1, c1)
        volumes_shape: tuple of 6 ints, (N1, N2, N3, h1, w1, c1)
    """
    assert len(img_data.shape) == len(vol_size) == len(step_size), 'The dimension of img_data, vol_size and step_size should be the same.'
    
    volumes = view_as_windows(img_data, vol_size, step_size)
    volumes_shape = volumes.shape

    volumes = volumes.reshape(-1, volumes_shape[3], volumes_shape[4], volumes_shape[5])

    return volumes, volumes_shape

def reconstruct(preds, img_data, vol_size, step_size, volumes_shape, output_shape):
    """
    Reconstruct the image from the patches.
    Args:
        preds: 4D numpy array, shape (N, h3, w3, c3)
        img_data: 3D numpy array, shape (H, W, C)
        vol_size: tuple of 3 ints, (h1, w1, c1)
        step_size: tuple of 3 ints, (h2, w2, c2)
        volumes_shape: tuple of 6 ints, (N1, N2, N3, h1, w1, c1)
        output_shape: tuple of 3 ints, (h3, w3, c3)
    Returns:
        reconstucted: 3D numpy array, shape (H, W, C)
        count: 3D numpy array, shape (H, W, C)
    """
    preds = rearrange(preds, '(b1 b2 b3) h w c -> b1 b2 b3 h w c', b1=volumes_shape[0], b2=volumes_shape[1], b3=volumes_shape[2])
    reconstucted = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]), dtype=np.float32)
    count = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2]), dtype=np.float32)

    dx, dy, dz = (vol_size[0] - output_shape[0]) // 2, (vol_size[1] - output_shape[1]) // 2, (vol_size[2] - output_shape[2]) // 2
    
    for i in range(volumes_shape[0]):
        for j in range(volumes_shape[1]):
            for k in range(volumes_shape[2]):
                x = i * step_size[0] + dx
                y = j * step_size[1] + dy
                z = k * step_size[2] + dz

                reconstucted[x:x+output_shape[0], y:y+output_shape[1], z:z+output_shape[2]] += preds[i,j,k]
                count[x:x+output_shape[0], y:y+output_shape[1], z:z+output_shape[2]] += 1
    
    return reconstucted, count