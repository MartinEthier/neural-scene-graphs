import numpy as np


# Sensor setup constants (all distances in meters)
WHEELBASE = 2.71
AXLE_HEIGHT = 0.3
CAM_HEIGHT = 1.65 
IMU_HEIGHT = 0.93
FPS = 10 # Hz
T_STEP = 1 / FPS # seconds

# Original image size
IMG_SIZE = (375, 1242) # (H, W)

# Average transformation matrices (run calculate_calibration.py to get them)
P2 = np.array([
    [7.16721350e+02, 0.00000000e+00, 6.07304470e+02, 4.51871514e+01],
    [0.00000000e+00, 7.16721350e+02, 1.76000546e+02, 1.60868480e-02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 3.52550930e-03]])
R_RECT = np.array([
    [ 0.99992234,  0.00978564, -0.00762135,  0.0],
    [-0.00981796,  0.99994261, -0.00424717,  0.0],
    [ 0.00757937,  0.00432166,  0.99996129,  0.0],
    [ 0.0,         0.0,         0.0,         1.0]])
TR_VELO_CAM = np.array([
    [ 0.00736142, -0.99997163, -0.00121094, -0.01025698],
    [ 0.00879806,  0.00127745, -0.99993035, -0.0711034 ],
    [ 0.99990448,  0.00735609,  0.00880698, -0.29135691],
    [ 0.0,         0.0,         0.0,         1.0       ]])
TR_IMU_VELO = np.array([
    [ 9.999976e-01,  7.553071e-04, -2.035826e-03, -8.086759e-01],
    [-7.854027e-04,  9.998898e-01, -1.482298e-02,  3.195559e-01],
    [ 2.024406e-03,  1.482454e-02,  9.998881e-01, -7.997231e-01],
    [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
TR_IMU_REARAXLE = np.array([
    [1.0, 0.0, 0.0, -0.05],
    [0.0, 1.0, 0.0, 0.32],
    [0.0, 0.0, 1.0, 0.63],
    [0.0, 0.0, 0.0, 1.0]
])

# Calculate rearaxle -> camera 2 plane transformation for visualizations
TR_REARAXLE_CAM2PLANE = P2 @ R_RECT @ TR_VELO_CAM @ TR_IMU_VELO @ np.linalg.inv(TR_IMU_REARAXLE)
