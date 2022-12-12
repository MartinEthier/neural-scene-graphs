"""
This module is based on a module from the pykitti library.
"""
from collections import namedtuple
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from constants import TR_IMU_REARAXLE


# Per dataformat.txt
OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([
        [c,  0,  s],
        [0,  1,  0],
        [-s, 0,  c]])

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1]])

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t

def load_oxts_poses(oxts_path, t0=None, horizon=None):
    """Generator to read OXTS ground truth data.
        Poses are given in an East-North-Up coordinate system 
        whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Inverse of origin pose of the global coordinate system (first GPS position)
    T_origin_inv = None

    #
    if t0 is None:
        t0 = 0
    if horizon is None:
        t_final = -1
    else:
        t_final = t0 + horizon + 1

    oxts = []
    poses = []

    with oxts_path.open('r') as f:
        # Just process timesteps within horizon
        lines = f.readlines()
        lines = lines[t0:t_final]
        for line in lines:
            line = line.split()
            # Last five entries are flags and counts
            line[:-5] = [float(x) for x in line[:-5]]
            line[-5:] = [int(float(x)) for x in line[-5:]]

            packet = OxtsPacket(*line)

            if scale is None:
                scale = np.cos(packet.lat * np.pi / 180.)

            R, t = pose_from_oxts_packet(packet, scale)

            # Current pose is in IMU frame, convert to middle of rear axle
            t_vec = np.append(t, 1.0)[:, np.newaxis]
            t = (TR_IMU_REARAXLE @ t_vec)[:3, 0]

            if T_origin_inv is None:
                T_origin_inv = np.linalg.inv(transform_from_rot_trans(R, t))

            # Calculate pose relative to origin
            rel_pose = T_origin_inv @ transform_from_rot_trans(R, t)

            poses.append(rel_pose)

    return poses


if __name__=="__main__":
    # Load a sample sequence and extract poses
    oxts_path = Path("~/repos/neural-scene-graphs/data/kitti/testing/oxts/0014.txt").expanduser()
    poses = load_oxts_poses(oxts_path)
    print(len(poses))
    print(poses[0])
    x = np.array([pose[0, 3] for pose in poses])
    y = np.array([pose[1, 3] for pose in poses])
    plt.plot(x, y)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.savefig("test_images/kitti_trajectory_testing_0014.png")
