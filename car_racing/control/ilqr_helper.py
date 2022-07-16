import numpy as np
from utils.constants import *

def get_cost_derivation(
    ctrl_U, 
    dX, 
    matrix_Q, 
    matrix_R, 
    num_horizon, 
    xvar, 
    obs_traj, 
    lap_length,
    num_cycle_ego,
    l_agent,
    w_agent,
    l_obs,
    w_obs
    ):
    # define control cost derivation
    l_u = np.zeros((U_DIM, num_horizon))
    l_uu = np.zeros((U_DIM, U_DIM, num_horizon))
    l_x = np.zeros((X_DIM, num_horizon))
    l_xx = np.zeros((X_DIM, X_DIM, num_horizon))
    # obstacle avoidance
    safety_margin = 0.15
    q1 = 2.5
    q2 = 2.5
    for i in range(num_horizon):
        l_u[:, i] = 2 * matrix_R @ ctrl_U[:, i]
        l_uu[:, :, i] = 2 * matrix_R
        l_x_i = 2 * matrix_Q @ dX[:, i]
        l_xx_i = 2 * matrix_Q
        # calculate control barrier functions for each obstacle at timestep
        degree = 2
        num_cycle_obs = int(obs_traj[4, 0] / lap_length)
        diffs = xvar[4, i] - obs_traj[4, i] - \
            (num_cycle_ego - num_cycle_obs) * lap_length
        diffey = xvar[5, i] - obs_traj[5, i]
        matrix_P1 = np.diag([0, 0, 0, 0, 1/((l_agent + l_obs) ** degree), 1/((w_agent + w_obs) ** degree)])
        diff = np.array([0, 0, 0, 0, diffs, diffey]).reshape(-1,1)
        h = 1 + safety_margin - diff.T @ matrix_P1 @ diff
        h_dot = -2 * matrix_P1 @ diff
        _, b_dot_obs, b_ddot_obs = repelling_cost_function(q1, q2, h, h_dot)
        l_x_i += b_dot_obs.squeeze()
        l_xx_i += b_ddot_obs
        l_xx[:, :, i] = l_xx_i
        l_x[:, i] = l_x_i
    return l_u, l_uu, l_x, l_xx


def repelling_cost_function(q1, q2, c, c_dot):
    b = q1*np.exp(q2*c)
    b_dot = q1*q2*np.exp(q2*c)*c_dot
    b_ddot = q1*(q2**2)*np.exp(q2*c)*np.matmul(c_dot, c_dot.T)
    return b, b_dot, b_ddot