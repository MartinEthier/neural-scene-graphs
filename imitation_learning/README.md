# Imitation Learning

This folder contains all the code needed to train a model to predict trajectories from input images on the KITTI tracking dataset.

## Dataset
Need a split file under the data/kitti folder that says which scenes are in train/val

Run the preprocess_dataset file to pre-calculate the poses, remove outliers (gps glitches), and split into train/val

Ex of gps glitch in /home/methier/repos/neural-scene-graphs/data/kitti/training/image_02/0019/000509.png

Try setting up simulator with the warp transform instead of the NeRF:
https://stackoverflow.com/questions/45811421/python-create-image-with-new-camera-position
