"""
Takes trajectory taken by the ego and applies a random left/right disturbance at
the chosen timestep. Using a simple controller, calculate a recovery trajectory 
that brings the car from the disturbed state back to the ground truth trajectory.

The nvidia e2e paper uses a pure pursuit controller but this only works because we
predict steering angles instead of trajectories. An MPC controller is needed to
produce full trajectories.

Note: Trajectory poses from dataset are the camera poses. To properly use MPC, 
transform reference trajectory poses to rear/front axle and run MPC using bicycle
model with front/rear axle point, then transform output trajectory back to camera.

From https://www.cvlibs.net/datasets/kitti/setup.php, can see that L=2.71 meters 
and can get the transformation between camera and front/rear axle.
"""
import numpy as np
from matplotlib import pyplot as plt
import do_mpc


### Parameters
# Model
L = 2.71 # meters
T_STEP = 0.1 # seconds (KITTI runs at 10 fps)
# MPC settings
N_ROBUST = 0
R_A = 1e-1
R_PHI = 1e-1
C_X = 1.0
C_Y = 1.0
C_V = 0.5
# Trajectory
L_TRAJ = 50 # length of gt trajectory
MAX_Y_DISTURB = 1.0 # meters
MAX_STEERING_DISTURB = 0.175 # ~10 deg in rad
MAX_HEADING_DISTURB = 0.175 # ~10 deg in rad
# Plotting
arrow_scale = 50
arrow_width = 3e-3

class MPC():
    def __init__(self):
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
        v_ref = self.model.set_variable('_tvp',  'velocity')
        theta_ref = self.model.set_variable('_tvp',  'heading_angle')
        
        # Define update equations
        x_next = x + T_STEP * v * np.cos(theta)
        y_next = y + T_STEP * v * np.sin(theta)
        v_next = v + T_STEP * a
        delta_next = delta + T_STEP * phi
        theta_next = theta + T_STEP * v / L * np.tan(delta)

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
            'n_horizon': L_TRAJ-1,
            'n_robust': N_ROBUST,
            't_step': T_STEP,
            'state_discretization': 'discrete',
            'store_full_solution': True
        }
        self.mpc.set_param(**mpc_params)
        
        # Configure objective function
        stage_cost = C_Y * (self.model.tvp['pos_y'] - self.model.x['pos_y'])**2 + \
                    C_X * (self.model.tvp['pos_x'] - self.model.x['pos_x'])**2 + \
                    C_V * (self.model.tvp['velocity'] - self.model.x['velocity'])**2
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
            for k in range(L_TRAJ):
                tvp_template['_tvp', k, 'pos_x'] = self.ref_traj['pos_x'][k]
                tvp_template['_tvp', k, 'pos_y'] = self.ref_traj['pos_y'][k]
                tvp_template['_tvp', k, 'velocity'] = self.ref_traj['velocity'][k]
            return tvp_template
        self.mpc.set_tvp_fun(tvp_fun)

        # Set bounds
        self.mpc.bounds['lower','_x', 'steering_angle'] = -0.52 # ~-30 deg
        self.mpc.bounds['upper','_x', 'steering_angle'] = 0.52 # ~30 deg
        self.mpc.bounds['lower','_u', 'acceleration'] = -1.0
        self.mpc.bounds['upper','_u', 'acceleration'] = 1.0

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
        plt.quiver(
            self.ref_traj['pos_x'],
            self.ref_traj['pos_y'],
            np.ones(L_TRAJ),
            np.ones(L_TRAJ),
            angles=self.ref_traj['heading_angle']*180/np.pi,
            color='green',
            scale=arrow_scale,
            width=arrow_width,
            label="ref_traj"
        )
        
        # Plot the predicted trajectory
        pos_x_pred = self.mpc.data.prediction(('_x', 'pos_x'))[0,:,0]
        pos_y_pred = self.mpc.data.prediction(('_x', 'pos_y'))[0,:,0]
        heading_angle_pred = self.mpc.data.prediction(('_x', 'heading_angle'))[0,:,0]
        plt.quiver(
            pos_x_pred,
            pos_y_pred,
            np.ones(L_TRAJ),
            np.ones(L_TRAJ),
            angles=heading_angle_pred*180/np.pi,
            color='red',
            scale=arrow_scale,
            width=arrow_width,
            label='pred_traj'
        )

        plt.legend(loc='best')
        plt.savefig("mpc_trajectory.png")

if __name__=="__main__":
    # Generate the reference trajectory
    # Trajectory is relative to pose a t=0 so pose at t=0 should be x=0,y=0,yaw=0
    gt_x = np.arange(L_TRAJ)
    w = 0.15
    C = 0.4
    gt_y = C * (np.cos(w * gt_x) - 1)
    #gt_y = np.zeros(L_TRAJ)
    gt_v = 1/T_STEP * np.ones(L_TRAJ)
    gt_heading = -w * C * np.sin(w * gt_x) # derivative of gt_y
    gt_trajectory = {
        'pos_x': gt_x,
        'pos_y': gt_y,
        'velocity': gt_v,
        'heading_angle': gt_heading
    }

    # Sample a new random starting pose
    y_new = gt_y[0] + np.random.uniform(-MAX_Y_DISTURB, MAX_Y_DISTURB)
    delta_new = np.random.uniform(-MAX_STEERING_DISTURB, MAX_STEERING_DISTURB)
    theta_new = np.random.uniform(-MAX_HEADING_DISTURB, MAX_HEADING_DISTURB)
    x0 = np.array([0.0, y_new, 1/T_STEP, delta_new, theta_new]) # [x, y, v, delta, theta]

    # Run MPC and plot predicted trajectory
    mpc = MPC()
    mpc.setup(gt_trajectory)
    mpc.set_initial(x0)
    mpc.plot_trajectory()
