import numpy as np
import torch
import cv2

from constants import AXLE_HEIGHT, TR_REARAXLE_CAM2PLANE, IMG_SIZE


def read_calib_file(file_path):
    data = {}
    with file_path.open('r') as f:
        for line in f.readlines():
            l = line.split()
            name = l[0].replace(':', '')
            arr = np.array([float(x) for x in l[1:]])

            # Reshape matrices to (4, 4)
            if arr.shape[0] == 9:
                # For R_rect, first need to pad with zero column
                arr = arr.reshape((3, 3))
                arr = np.pad(arr, [(0, 0), (0, 1)], mode='constant')
            else:
                arr = arr.reshape((3, 4))
            # Add zero row and set last element to 1
            arr = np.pad(arr, [(0, 1), (0, 0)], mode='constant')
            arr[-1, -1] = 1.0

            data[name] = arr

    return data

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

def batch_tensor_to_array(tensor):
    """
    Converts a torch tensor image to opencv numpy format.
    """
    # Convert to 0-255 uint8 numpy array
    img = tensor.numpy()
    img -= img.min(axis=(1, 2, 3), keepdims=True)
    img *= 255 / img.max(axis=(1, 2, 3), keepdims=True)
    img = img.astype(np.uint8)

    # Transpose to BHWC
    img = np.moveaxis(img, 1, -1)

    return img

def display_path(img, path, color=(255, 0, 0)):
    """
    img: (H, W, 3) numpy array
    path: (N, 2) numpy array (relative to center of rear axle coordinate frame)
    """
    # img needs to be original size to be compatible with projection matrix
    if img.shape[:2] != IMG_SIZE:
        img = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))

    # Add z dimension to path assuming flat world
    # Add 1 vector to be compatible with 4x4 transformation matrix
    N = path.shape[0]
    path_3d = np.concatenate((path, -AXLE_HEIGHT * np.ones((N, 1)), np.ones((N, 1))), axis=1)

    # Map points from rearaxle frame to camera 2 plane
    img_path = (TR_REARAXLE_CAM2PLANE @ path_3d.T).T

    # Normalize by z to get (u,v,1)
    img_path[:, :2] /= img_path[:, 2:3]

    for p in img_path:
        loc = (int(p[0]), int(p[1]))
        cv2.drawMarker(img, loc, color, markerType=cv2.MARKER_STAR, markerSize=10, thickness=1, line_type=cv2.LINE_AA)

    return img
