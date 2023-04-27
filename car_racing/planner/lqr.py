import datetime

import numpy as np
import scipy.linalg as la

from planner.base import PlannerBase
from racing_env import U_DIM, X_DIM, SystemParam


class LQRTrackingParam:
    """ Collection of tunable LQR parameters
    """
    def __init__(
        self,
        matrix_A: np.ndarray = np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=","),
        matrix_B: np.ndarray = np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=","),
        matrix_Q: np.ndarray = np.diag([10.0, 0.0, 0.0, 4.0, 0.0, 40.0]),
        matrix_R: np.ndarray = np.diag([0.1, 0.1]),
        vt: float = 0.6,
        eyt: float = 0.0,
        max_iter: int = 50,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.vt = vt
        self.eyt = eyt
        self.max_iter = max_iter


class iLQRRacingParam:
    """ Collection of tunable iLQR parameters
    """
    def __init__(
        self,
        matrix_A: np.ndarray = np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=","),
        matrix_B: np.ndarray = np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=","),
        matrix_Q: np.ndarray = np.diag([10.0, 0.0, 0.0, 4.0, 0.0, 40.0]),
        matrix_R: np.ndarray = np.diag([0.1, 0.1]),
        vt: float = 0.6,
        eyt: float = 0.0,
        max_iter: int = 150,
        num_horizon: int = 50,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.vt = vt
        self.eyt = eyt
        self.max_iter = max_iter
        self.num_horizon = num_horizon


class LQRTracking(PlannerBase):
    """LQR as the planner"""
    def __init__(self, lqr_param: LQRTrackingParam, system_param: SystemParam):
        PlannerBase.__init__(self)
        self.set_target_speed(lqr_param.vt)
        self.set_target_deviation(lqr_param.eyt)
        self.system_param = system_param
        self.lqr_param = lqr_param

    def _lqr(self, xtarget: np.ndarray):
        """Run the LQR towards the target `xtarget`.
        
        Params
        ------
        xtarget: the target state

        Returns
        -------
        the planned control inputs
        """
        vt = xtarget[0]
        eyt = xtarget[5]
        A = self.lqr_param.matrix_A
        B = self.lqr_param.matrix_B
        max_iter = self.lqr_param.max_iter
        start_timer = datetime.datetime.now()
        # define variables
        xvar = np.zeros(X_DIM).reshape(X_DIM, 1)
        xvar[:, 0] = self.x
        u_next = np.zeros((U_DIM,))
        # solve a discrete time Algebraic Riccati equation
        R = self.lqr_param.matrix_R
        Q = self.lqr_param.matrix_Q
        P = Q
        P_iter = Q
        eps = 0.01
        # achieve convergence
        for i in range(max_iter):
            P_iter = A.T @ P @ A - A.T @ P @ B @ \
                    la.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
            if abs(P_iter - P).max() < eps:
                break
            P = P_iter
        # compute the gain K
        K = la.inv(B.T @ P @ B + R) @ B.T @ P @ A
        uvar = - K @ (xvar - xtarget)
        # Optimal control input
        u_next[0] = uvar[0, 0]
        u_next[1] = uvar[1, 0]
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("solver time: {}".format(solver_time))
        return u_next

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(X_DIM, 1)
        self.u = self._lqr(xtarget, self.lqr_param)
        if self.agent_name == "ego":
            if self.realtime_flag == False:
                vehicles = self.racing_env.vehicles
            else:
                vehicles = self.vehicles
            vehicles["ego"].local_trajs.append(None)
            vehicles["ego"].vehicles_interest.append(None)
            vehicles["ego"].splines.append(None)
            vehicles["ego"].all_splines.append(None)
            vehicles["ego"].all_local_trajs.append(None)
            vehicles["ego"].lmpc_prediction.append(None)
            vehicles["ego"].mpc_cbf_prediction.append(None)
        self.time += self.timestep


class iLQRRacing(PlannerBase):
    """iLQR as planner"""
    def __init__(self, ilqr_param, system_param):
        PlannerBase.__init__(self)
        self.set_target_speed(ilqr_param.vt)
        self.set_target_deviation(ilqr_param.eyt)
        self.system_param = system_param
        self.ilqr_param = ilqr_param

    # ---------------- SOME HELPER FUNCTIONS ----------------
    @classmethod
    def _repelling_cost_function(cls, q1, q2, c, c_dot):
        b = q1*np.exp(q2*c)
        b_dot = q1*q2*np.exp(q2*c)*c_dot
        b_ddot = q1*(q2**2)*np.exp(q2*c)*np.matmul(c_dot, c_dot.T)
        return b, b_dot, b_ddot

    @classmethod
    def _get_cost_derivation(cls,
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
            _, b_dot_obs, b_ddot_obs = cls._repelling_cost_function(q1, q2, h, h_dot)
            l_x_i += b_dot_obs.squeeze()
            l_xx_i += b_ddot_obs
            l_xx[:, :, i] = l_xx_i
            l_x[:, i] = l_x_i
        return l_u, l_uu, l_x, l_xx

    def _ilqr(self, xtarget) -> np.ndarray:
        """ Core of the iLQR alorithm, find the control inputs
        to drive the system towards a target

        Params
        ------
        xtarget: the target state (in local representation)

        Returns
        -------
        Control inputs
        """
        matrix_A = self.ilqr_param.matrix_A
        matrix_B = self.ilqr_param.matrix_B
        max_iter = self.ilqr_param.max_iter
        num_horizon = self.ilqr_param.num_horizon
        eps = 0.01
        lamb = 1
        lamb_factor = 10
        max_lamb = 1000
        start_timer = datetime.datetime.now()
        # define variables
        uvar = np.zeros((U_DIM, num_horizon))
        xvar = np.zeros((X_DIM, num_horizon+1))
        xvar[:, 0] = self.x
        matrix_Q = self.ilqr_param.matrix_Q
        matrix_R = self.ilqr_param.matrix_R
        dX = np.zeros((X_DIM, num_horizon+1))
        dX[:, 0] = xvar[:, 0] - xtarget
        #get other vehicles' state estimations
        safety_time = 2.0
        dist_margin_front = self.x[0] * safety_time
        dist_margin_behind = self.x[0] * safety_time
        num_cycle_ego = int(self.x[4] / self.racing_env.track.lap_length)
        dist_ego = self.x[4] - num_cycle_ego * self.racing_env.track.lap_length
        obs_infos = {}
        for name in list(self.racing_env.vehicles):
            if name != self.agent_name:
                # get predictions from other vehicles
                obs_traj, _ = self.racing_env.vehicles[name].get_trajectory_nsteps(
                    self.time, self.timestep, self.ilqr_param.num_horizon + 1
                )
        # get ego agent and obstacles' dimensions
        l_agent = self.racing_env.vehicles[self.agent_name].param.length / 2
        w_agent = self.racing_env.vehicles[self.agent_name].param.width / 2
        l_obs = self.racing_env.vehicles["car1"].param.length / 2
        w_obs = self.racing_env.vehicles["car1"].param.width / 2
        for i in range(max_iter):
            # Forward simulation
            cost = 0
            for k in range(num_horizon):
                xvar[:, k+1] = matrix_A @ xvar[:, k] + matrix_B @ uvar[:, k]
                dX[:, k+1] = xvar[:, k+1] - xtarget.T
                l_state = (xvar[:, k] - xtarget).T @ matrix_Q @ (xvar[:, k] - xtarget)
                l_ctrl = uvar[:, k].T @ matrix_R @ uvar[:, k]
                cost = cost + l_state + l_ctrl
            l_state_final = (xvar[:, num_horizon] - xtarget).T @ \
                matrix_Q @ (xvar[:, num_horizon] - xtarget)
            cost = cost + l_state_final
            # Backward pass
            # System derivation
            f_x = matrix_A
            f_u = matrix_B
            # cost derivation
            l_u, l_uu, l_x, l_xx = self._get_cost_derivation(
                uvar, 
                dX, 
                matrix_Q, 
                matrix_R, 
                num_horizon,xvar, 
                obs_traj, 
                self.racing_env.track.lap_length,
                num_cycle_ego,
                l_agent,
                w_agent,
                l_obs,
                w_obs
            )
            # Value function at last timestep
            matrix_Vx = l_x[:, -1]
            matrix_Vxx = l_xx[:, :, -1]
            # define control modification k and K
            K = np.zeros((U_DIM, X_DIM, num_horizon))
            k = np.zeros((U_DIM, num_horizon))
            for i in range(num_horizon-1, -1, -1):
                matrix_Qx = l_x[:,i] + f_x.T @ matrix_Vx
                matrix_Qu = l_u[:,i] + f_u.T @ matrix_Vx
                matrix_Qxx = l_xx[:,:,i] + f_x.T @ matrix_Vxx @ f_x
                matrix_Quu = l_uu[:,:,i] + f_u.T @ matrix_Vxx @ f_u
                matrix_Qux = f_u.T @ matrix_Vxx @ f_x
                # Improved Regularization
                Q_uu_evals, Q_uu_evecs = np.linalg.eig(matrix_Quu)
                Q_uu_evals[Q_uu_evals < 0] = 0.0
                Q_uu_evals += lamb
                matrix_Quu_inv = np.dot(Q_uu_evecs,np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))
                # Calculate feedforward and feedback terms
                k[:,i] = -matrix_Quu_inv @ matrix_Qu
                K[:,:,i] = -matrix_Quu_inv @ matrix_Qux
                # Update value function for next time step
                matrix_Vx = matrix_Qx - K[:,:,i].T @ matrix_Quu @ k[:,i]
                matrix_Vxx = matrix_Qxx - K[:,:,i].T @ matrix_Quu @ K[:,:,i]
            # Forward pass
            xvar_new = np.zeros((X_DIM, num_horizon + 1))
            xvar_new[:, 0] = self.x
            uvar_new = np.zeros((U_DIM, num_horizon))
            cost_new = 0
            for i in range(num_horizon):
                uvar_new[:, i] = uvar[:, i] + k[:, i] + K[:, :, i] @ \
                    (xvar_new[:, i] - xvar[:, i])    
                xvar_new[:, i+1] = matrix_A @ xvar_new[:, i] + matrix_B @ uvar_new[:, i]
                l_state_new = (xvar_new[:, i] - xtarget).T @ \
                    matrix_Q @ (xvar_new[:, i] - xtarget)        
                l_ctrl_new = uvar_new[:, i].T @ matrix_R @ uvar_new[:, i]
                cost_new = cost_new + l_state_new + l_ctrl_new
            l_state_final_new = (xvar_new[:, num_horizon] - xtarget).T @ \
                matrix_Q @ (xvar_new[:, num_horizon] - xtarget)
            cost_new = cost_new + l_state_final_new
            if cost_new < cost:
                xvar = xvar_new
                uvar = uvar_new
                lamb /= lamb_factor
                if abs((cost_new - cost)/cost) < eps:
                    print("Convergence achieved")
                    break
            else:
                lamb *= lamb_factor
                if lamb > max_lamb:
                    break
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("solver time: {}".format(solver_time))
        return uvar[:, 0]

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt])  # .reshape(X_DIM, 1)
        self.u = self._ilqr(
            self.x,
            xtarget,
            self.ilqr_param,
            self.racing_env.vehicles,
            self.agent_name,
            self.racing_env.track.lap_length,
            self.time,
            self.timestep,
            self.track,
            self.system_param,
        )
        if self.agent_name == "ego":
            if self.realtime_flag == False:
                vehicles = self.racing_env.vehicles
            else:
                vehicles = self.vehicles
            vehicles["ego"].local_trajs.append(None)
            vehicles["ego"].vehicles_interest.append(None)
            vehicles["ego"].splines.append(None)
            vehicles["ego"].all_splines.append(None)
            vehicles["ego"].all_local_trajs.append(None)
            vehicles["ego"].lmpc_prediction.append(None)
            vehicles["ego"].mpc_cbf_prediction.append(None)
        self.time += self.timestep
