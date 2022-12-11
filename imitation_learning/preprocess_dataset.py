from pathlib import Path
import argparse
import json
import random
random.seed(1)

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import median_abs_deviation

from kitti_trajectory import load_oxts_poses


GRAD_SPIKE_THRESH = 0.06
DEBUG_PLOT = False


def get_valid_paths(positions, horizon):
    """
    Returns all indices where there are no trajectory glitches within the
    horizon starting at that index.

    positions: (N, 2) array of (x, y) positions
    horizon: int corresponding to horizon window
    """
    # To find glitches in the path, calculate the gradient along the path and
    # and find locations where there is a sudden change in the gradient
    grads = np.mean(np.gradient(positions, axis=0), axis=1)
    grad_diffs = np.abs(np.diff(grads))
    grad_diffs = np.insert(grad_diffs, 0, 0) # Add leading zero so sizes match
    spike_mask = grad_diffs > GRAD_SPIKE_THRESH

    if DEBUG_PLOT:
        plt.plot(grad_diffs, label="grad_diffs")
        plt.axhline(GRAD_SPIKE_THRESH, color='r')
        plt.savefig("grad_diffs.png")

    # Find all paths with no glitches in horizon
    valid_t0 = []
    for t0 in range(1, positions.shape[0] - horizon):
        if not np.any(spike_mask[t0 : t0 + horizon]):
            valid_t0.append(t0)

    return valid_t0

def main(root_dir, train_ratio, horizon):
    # Collect all scenes in both train and test folders
    train_dir = root_dir / "training"
    test_dir = root_dir / "testing"
    scene_dict = {}
    # For each scene, calculate all the GT paths and verify for GPS glitches
    for split_dir in [train_dir, test_dir]:
        for img_dir in split_dir.glob("image_02/*"):
            scene_id = img_dir.stem
            oxts_path = img_dir.parents[1] / "oxts" / f"{scene_id}.txt"
            poses = load_oxts_poses(oxts_path)

            # Extract xy positions from pose
            x = np.array([pose[0, 3] for pose in poses])
            y = np.array([pose[1, 3] for pose in poses])
            positions = np.stack((x, y), axis=1)

            # Get all starting points where the path has no glitches
            valid_t0 = get_valid_paths(positions, horizon)
            
            scene_dict[str(img_dir.relative_to(root_dir))] = valid_t0

    # Randomly split scenes into new train/val
    scenes = list(scene_dict.keys())
    random.shuffle(scenes)
    train_idx = int(train_ratio * len(scenes))
    train_scenes = scenes[:train_idx]
    val_scenes = scenes[train_idx:]
    train_scene_dict = dict((scene, scene_dict[scene]) for scene in train_scenes)
    val_scene_dict = dict((scene, scene_dict[scene]) for scene in val_scenes)

    # Save new split + valid t0 in root directory
    for filename, data in [("train.json", train_scene_dict), ("val.json", val_scene_dict)]:
        fpath = root_dir / filename
        with fpath.open('w') as f:
            f.write(json.dumps(data))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=Path,
        help="Root path of the KITTI tracking dataset.")
    parser.add_argument("--train_ratio", type=float, default=0.8,
        help="Ratio of training to validation scenes.")
    parser.add_argument("--horizon", type=int, default=50,
        help="Length of ground truth trajectories.")
    args = parser.parse_args()

    if args.train_ratio < 0.0 or 1.0 < args.train_ratio:
        raise ValueError("train_ratio argument must be between 0 and 1.")
    if args.horizon <= 0:
        raise ValueError("horizon argument must be greater than 0.")

    main(args.root_dir, args.train_ratio, args.horizon)
