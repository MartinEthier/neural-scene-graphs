import numpy as np
import torch


def tensor_to_array(tensor):
    """
    Converts a torch tensor image to opencv numpy format.
    """
    # Convert to 0-255 uint8 numpy array
    img = tensor.numpy()
    img -= img.min()
    img *= 255/img.max()
    img = img.astype(np.uint8)

    # Transpose to HWC
    img = np.moveaxis(img, 0, -1)

    return img

def display_path(img, path):
    """

    img: (H, W, 3) numpy array
    path: (N, 2) numpy array
    """
    return None

