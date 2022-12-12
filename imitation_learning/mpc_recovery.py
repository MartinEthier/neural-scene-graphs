import numpy as np
from matplotlib import pyplot as plt
import do_mpc

from constants import WHEELBASE, T_STEP


# MPC parameters
N_ROBUST = 0
R_A = 1e-1
R_PHI = 1e-1
C_X = 1.0
C_Y = 1.0
MAX_ACCEL = 1.0 # m/s^2
MAX_STEER_ANGLE = 0.52 # ~30 deg


class MPC():
    def __init__(self, horizon):
        self.horizon = horizon
        self.model = None
        self.mpc = None
        self.simulator = None

    def setup_model(self):
        """
        Setup a simple bicycle kinematics model.
        """
        self.model = do_mpc.model.Model('discrete')

        # State
        x = self.model.set_variable('_x',  'pos_x')
        y = self.model.set_variable('_x',  'pos_y')
        v = self.model.set_variable('_x', 'velocity')
        delta = self.model.set_variable('_x',  'steering_angle')
        theta = self.model.set_variable('_x',  'heading_angle')

        # Input
        a = self.model.set_variable('_u',  'acceleration')
        phi = self.model.set_variable('_u',  'steering_rate')

        # Reference trajectory
        x_ref = self.model.set_variable('_tvp',  'pos_x')
        y_ref = self.model.set_variable('_tvp',  'pos_y')
        
        # Define update equations
        x_next = x + T_STEP * v * np.cos(theta)
        y_next = y + T_STEP * v * np.sin(theta)
        v_next = v + T_STEP * a
        delta_next = delta + T_STEP * phi
        theta_next = theta + T_STEP * v / WHEELBASE * np.tan(delta)

        # Trick to wrap angle to [-pi, pi] range
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))

        # Set update equations in model
        self.model.set_rhs('pos_x', x_next)
        self.model.set_rhs('pos_y', y_next)
        self.model.set_rhs('velocity', v_next)
        self.model.set_rhs('steering_angle', delta_next)
        self.model.set_rhs('heading_angle', theta_next)

        self.model.setup()

    def setup_mpc(self, ref_traj):
        # Setup controller with params
        self.mpc = do_mpc.controller.MPC(self.model)
        mpc_params = {
            'n_horizon': self.horizon,
            'n_robust': N_ROBUST,
            't_step': T_STEP,
            'state_discretization': 'discrete',
            'store_full_solution': True
        }
        self.mpc.set_param(**mpc_params)
        
        # Configure objective function
        stage_cost = C_Y * (self.model.tvp['pos_y'] - self.model.x['pos_y'])**2 + \
                    C_X * (self.model.tvp['pos_x'] - self.model.x['pos_x'])**2
        terminal_cost = stage_cost
        self.mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)
        self.mpc.set_rterm(
            acceleration=R_A,
            steering_rate=R_PHI
        )

        # Set reference trajectory
        self.ref_traj = ref_traj
        tvp_template = self.mpc.get_tvp_template()
        def tvp_fun(t_now):
            for k in range(self.horizon+1):
                tvp_template['_tvp', k, 'pos_x'] = self.ref_traj['pos_x'][k]
                tvp_template['_tvp', k, 'pos_y'] = self.ref_traj['pos_y'][k]
            return tvp_template
        self.mpc.set_tvp_fun(tvp_fun)

        # Set bounds
        self.mpc.bounds['lower','_x', 'steering_angle'] = -MAX_STEER_ANGLE
        self.mpc.bounds['upper','_x', 'steering_angle'] = MAX_STEER_ANGLE
        self.mpc.bounds['lower','_u', 'acceleration'] = -MAX_ACCEL
        self.mpc.bounds['upper','_u', 'acceleration'] = MAX_ACCEL

        self.mpc.setup()

    def setup_estimator(self):
        self.estimator = do_mpc.estimator.StateFeedback(self.model)

    def setup_simulator(self):
        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.simulator.set_param(t_step=T_STEP)

        # Need a tvp template so just use default
        tvp_template = self.simulator.get_tvp_template()
        def tvp_fun(t_now):
            return tvp_template
        self.simulator.set_tvp_fun(tvp_fun)

        self.simulator.setup()

    def setup(self, ref_traj):
        self.setup_model()
        self.setup_mpc(ref_traj)
        self.setup_estimator()
        self.setup_simulator()
    
    def set_initial(self, x0):
        self.x0 = x0
        self.mpc.x0 = self.x0
        self.simulator.x0 = self.x0
        self.estimator.x0 = self.x0
        self.mpc.set_initial_guess()
    
    def plot_trajectory(self):
        # Calculates predictions over the horizon
        u0 = self.mpc.make_step(self.x0)
        
        # Plot reference trajectory
        plt.plot(self.ref_traj['pos_x'], self.ref_traj['pos_y'], color='green', label='ref_traj')
        
        # Plot the predicted trajectory
        pos_x_pred = self.mpc.data.prediction(('_x', 'pos_x'))[0,:,0]
        pos_y_pred = self.mpc.data.prediction(('_x', 'pos_y'))[0,:,0]
        plt.plot(pos_x_pred, pos_y_pred, color='red', label='pred_traj')

        plt.legend(loc='best')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.savefig("test_images/mpc_trajectory.png")

if __name__=="__main__":
    # Load a KITTI trajectory to test
    from pathlib import Path
    from kitti_trajectory import load_oxts_poses
    oxts_path = Path("~/repos/neural-scene-graphs/data/kitti/testing/oxts/0014.txt").expanduser()
    t0 = 100
    horizon = 50
    poses = load_oxts_poses(oxts_path, t0, horizon)
    gt_x = np.array([pose[0, 3] for pose in poses])
    gt_y = np.array([pose[1, 3] for pose in poses])
    gt_trajectory = {
        'pos_x': gt_x,
        'pos_y': gt_y,
    }

    # Apply disturbance to starting pose
    y_offset = 0.5
    steering_offset = -0.087 # 5 deg in rad
    heading_offset = -0.087 # 5 deg in rad
    y_new = gt_y[0] + y_offset

    # Set initial velocity to be the same as path
    v0 = ((gt_x[1]-gt_x[0])**2 + (gt_y[1]-gt_y[0])**2)**0.5 / T_STEP

    x0 = np.array([0.0, y_new, v0, steering_offset, heading_offset]) # [x, y, v, delta, theta]

    # Run MPC and plot predicted trajectory
    mpc = MPC(horizon)
    mpc.setup(gt_trajectory)
    mpc.set_initial(x0)
    mpc.plot_trajectory()
