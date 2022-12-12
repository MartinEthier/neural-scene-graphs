"""
Calculate the average calibration matrices to avoid needing to load individual calibrations at runtime.
"""
from pathlib import Path

import numpy as np

from utils import read_calib_file


def main(root_path):
    splits = ["training", "testing"]
    calibs = {}
    for split in splits:
        calib_dir = root_path / split / "calib"
        for calib_path in calib_dir.glob("*.txt"):
            calib = read_calib_file(calib_path)
            for key, arr in calib.items():
                if key not in calibs:
                    calibs[key] = []
                calibs[key].append(arr)

    for key, arr_list in calibs.items():
        print(key)
        calibs_arr = np.stack(arr_list, axis=0)
        mean_arr = calibs_arr.mean(0)
        print(mean_arr)

if __name__=="__main__":
    root_path = Path("/home/methier/repos/neural-scene-graphs/data/kitti")
    main(root_path)
