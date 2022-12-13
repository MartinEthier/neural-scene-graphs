# Imitation Learning

This folder contains all the code needed to train a model to predict trajectories from input images on the KITTI tracking dataset.

## Environment

Setup the pytorch conda environment by running:

```bash
conda env create -f environment.yml
```

## Dataset
Setup the KITTI tracking dataset as described in the base README for the neural-scene-graphs repo. Then, run the pre-processing script to remove outliers and split the dataset:

```bash
python preprocess_dataset.py <path_to_kitti_folder>
```

## Training

To train an end-to-end trajectory planning model, edit the "root_dir" value in the config directory at the bottom of the train.py script and run:

```bash
python train.py
```

## MPC Recovery Trajectory

To test out the MPC recovery trajectory calculation, you can play with the settings in the mpc_recovery.py script and then run:

```bash
python mpc_recovery.py
```
