from pathlib import Path
import json
import random

import numpy as np
import torch
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

import kitti_trajectory as kt
import utils


class KITTIDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, horizon, num_frames, img_transforms=None, full_transforms=None):
        self.root_dir = Path(root_dir).expanduser()
        self.horizon = horizon
        self.num_frames = num_frames
        self.img_transforms = img_transforms
        self.full_transforms = full_transforms
        self.to_tensor = ToTensor()

        # Load all valid path locations
        self.dataset = []
        split_file = self.root_dir / f"{split}.json"
        with split_file.open('r') as f:
            scene_dict = json.load(f)
            for scene, valid_t0 in scene_dict.items():
                self.dataset.extend([(scene, t0) for t0 in valid_t0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load frames
        scene = self.dataset[idx][0]
        t0 = self.dataset[idx][1]
        scene_path = self.root_dir / scene
        frames = []
        for frame_id in range(t0 + 1 - self.num_frames, t0 + 1):
            img_path = scene_path / f"{frame_id:06}.png"
            frame = Image.open(img_path)
            frames.append(self.to_tensor(frame))
        
        # Stack frames into single tensor of shape (T, C, H, W)
        frames = torch.stack(frames)
        
        # torchvision applies the same transform to a batch of images
        if self.img_transforms is not None:
            frames = self.img_transforms(frames)

        # Get relative GT path over horizon
        scene_id = img_path.parents[0].stem
        split = img_path.parents[2].stem
        oxts_path = self.root_dir / split / "oxts" / f"{scene_id}.txt"
        poses = kt.load_oxts_poses(oxts_path, t0, self.horizon)
        x = np.array([pose[0, 3] for pose in poses])
        y = np.array([pose[1, 3] for pose in poses])
        positions = np.stack((x, y), axis=1)[1:]

        sample = {
            "frames": frames,
            "label_path": torch.from_numpy(positions).float()
        }

        if self.full_transforms is not None:
            sample = self.full_transforms(sample)

        # Concat the frames together into the channel dimension
        s = sample['frames'].shape
        sample['frames'] = torch.reshape(sample['frames'], (s[0]*s[1], s[2], s[3]))

        return sample


if __name__=="__main__":
    from torchvision import transforms as tf
    from custom_transforms import Denormalize
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    img_tf = tf.Compose([
        tf.Resize([256, 256]),
        tf.Normalize(mean=norm_mean, std=norm_std)
    ])
    train_set = KITTIDataset("~/repos/neural-scene-graphs/data/kitti", "train", 50, 2, img_transforms=img_tf)
    print(len(train_set))
    val_set = KITTIDataset("~/repos/neural-scene-graphs/data/kitti", "val", 50, 2, img_transforms=img_tf)
    print(len(val_set))

    # Visualize a sample
    sample = train_set[6000]
    print(sample['frames'].shape)
    print(sample['label_path'].shape)
    frame = sample['frames'][-3:]
    frame = Denormalize()(frame)
    img = utils.tensor_to_array(frame)
    label_path = sample['label_path'].numpy()
    img = utils.display_path(img, label_path)

    plt.imshow(img)
    plt.savefig("test_images/dataset_sample.png")
