import datetime

import numpy as np
import casadi as ca

from planner.base import PlannerBase
from racing_env.params import X_DIM, U_DIM

class MPCTrackingParam:
    def __init__(
        self,
        matrix_A=np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=","),
        matrix_B=np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=","),
        matrix_Q=np.diag([10.0, 0.0, 0.0, 4.0, 0.0, 40.0]),
        matrix_R=np.diag([0.1, 0.1]),
        vt=0.6,
        eyt=0.0,
        num_horizon=10,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.vt = vt
        self.eyt = eyt
        self.num_horizon = num_horizon


class MPCTracking(PlannerBase):
    def __init__(self, mpc_lti_param, system_param):
        PlannerBase.__init__(self)
        self.set_target_speed(mpc_lti_param.vt)
        self.set_target_deviation(mpc_lti_param.eyt)
        self.mpc_lti_param = mpc_lti_param
        self.system_param = system_param

    def _mpc_lti(self, xtarget):
        vt = xtarget[0]
        eyt = xtarget[5]
        num_horizon = self.mpc_lti_param.num_horizon
        start_timer = datetime.datetime.now()
        opti = ca.Opti()
        # define variables
        xvar = opti.variable(X_DIM, num_horizon + 1)
        uvar = opti.variable(U_DIM, num_horizon)
        cost = 0
        opti.subject_to(xvar[:, 0] == self.x)
        # dynamics + state/input constraints
        for i in range(num_horizon):
            # system dynamics
            opti.subject_to(
                xvar[:, i + 1]
                == ca.mtimes(self.mpc_lti_param.matrix_A, xvar[:, i])
                + ca.mtimes(self.mpc_lti_param.matrix_B, uvar[:, i])
            )
            # min and max of delta
            opti.subject_to(-self.system_param.delta_max <= uvar[0, i])
            opti.subject_to(uvar[0, i] <= self.system_param.delta_max)
            # min and max of a
            opti.subject_to(-self.system_param.a_max <= uvar[1, i])
            opti.subject_to(uvar[1, i] <= self.system_param.a_max)
            # input cost
            cost += ca.mtimes(uvar[:, i].T,
                            ca.mtimes(self.mpc_lti_param.matrix_R, uvar[:, i]))
        for i in range(num_horizon + 1):
            # speed vx upper bound
            opti.subject_to(xvar[0, i] <= self.system_param.v_max)
            opti.subject_to(xvar[0, i] >= self.system_param.v_min)
            # min and max of ey
            opti.subject_to(xvar[5, i] <= self.track.width)
            opti.subject_to(-self.track.width <= xvar[5, i])
            # state cost
            cost += ca.mtimes(
                (xvar[:, i] - xtarget).T,
                ca.mtimes(self.mpc_lti_param.matrix_Q, xvar[:, i] - xtarget),
            )
        # setup solver
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.minimize(cost)
        opti.solver("ipopt", option)
        sol = opti.solve()
        x_pred = sol.value(xvar).T
        u_pred = sol.value(uvar).T
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("solver time: {}".format(solver_time))
        return u_pred[0, :]

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(X_DIM, 1)
        self.u = self._mpc_lti(xtarget)
        if self.agent_name == "ego":
            if self.realtime_flag == False:
                vehicles = self.racing_sim.vehicles
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


class MPCCBFRacingParam:
    def __init__(
        self,
        matrix_A=np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=","),
        matrix_B=np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=","),
        matrix_Q=np.diag([10.0, 0.0, 0.0, 4.0, 0.0, 40.0]),
        matrix_R=np.diag([0.1, 0.1]),
        vt=0.6,
        eyt=0.0,
        num_horizon=10,
        alpha=0.6,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.vt = vt
        self.eyt = eyt
        self.num_horizon = num_horizon
        self.alpha = alpha


class MPCCBFRacing(PlannerBase):
    def __init__(self, mpc_cbf_param, system_param):
        PlannerBase.__init__(self)
        self.set_target_speed(mpc_cbf_param.vt)
        self.set_target_deviation(mpc_cbf_param.eyt)
        self.realtime_flag = None
        self.mpc_cbf_param = mpc_cbf_param
        self.system_param = system_param

    def _mpccbf(self, xtarget):
        vt = xtarget[0]
        eyt = xtarget[5]
        start_timer = datetime.datetime.now()
        opti = ca.Opti()
        # define variables
        xvar = opti.variable(X_DIM, self.mpc_cbf_param.num_horizon + 1)
        uvar = opti.variable(U_DIM, self.mpc_cbf_param.num_horizon)
        cost = 0
        opti.subject_to(xvar[:, 0] == self.x)
        # get other vehicles' state estimations
        safety_time = 2.0
        dist_margin_front = self.x[0] * safety_time
        dist_margin_behind = self.x[0] * safety_time
        num_cycle_ego = int(self.x[4] / self.racing_sim.track.lap_length)
        dist_ego = self.x[4] - num_cycle_ego * self.racing_sim.track.lap_length
        obs_infos = {}
        for name in list(self.vehicles):
            if name != self.agent_name:
                # get predictions from other vehicles
                if self.realtime_flag == False:
                    obs_traj, _ = self.vehicles[name].get_trajectory_nsteps(
                        self.time, self.timestep, self.mpc_cbf_param.num_horizon + 1
                    )
                elif self.realtime_flag == True:
                    obs_traj, _ = self.vehicles[name].get_trajectory_nsteps(
                        self.mpc_cbf_param.num_horizon + 1)
                else:
                    pass
                # check whether the obstacle is nearby, not consider it if not
                num_cycle_obs = int(obs_traj[4, 0] / self.racing_sim.track.lap_length)
                dist_obs = obs_traj[4, 0] - num_cycle_obs * self.racing_sim.track.lap_length
                if (dist_ego > dist_obs - dist_margin_front) & (
                    dist_ego < dist_obs + dist_margin_behind
                ):
                    obs_infos[name] = obs_traj
        # slack variables for control barrier functions
        cbf_slack = opti.variable(len(obs_infos), self.mpc_cbf_param.num_horizon + 1)
        # obstacle avoidance
        safety_margin = 0.15
        degree = 6  # 2, 4, 6, 8
        for count, obs_name in enumerate(obs_infos):
            obs_traj = obs_infos[obs_name]
            # get ego agent and obstacles' dimensions
            l_agent = self.vehicles[self.agent_name].param.length / 2
            w_agent = self.vehicles[self.agent_name].param.width / 2
            l_obs = self.vehicles[obs_name].param.length / 2
            w_obs = self.vehicles[obs_name].param.width / 2
            # calculate control barrier functions for each obstacle at timestep
            for i in range(self.mpc_cbf_param.num_horizon):
                num_cycle_obs = int(obs_traj[4, 0] / self.racing_sim.track.lap_length)
                diffs = xvar[4, i] - obs_traj[4, i] - \
                    (num_cycle_ego - num_cycle_obs) * self.racing_sim.track.lap_length
                diffey = xvar[5, i] - obs_traj[5, i]
                diffs_next = xvar[4, i + 1] - obs_traj[4, i + 1]
                diffey_next = xvar[5, i + 1] - obs_traj[5, i + 1]
                h = (
                    diffs ** degree / ((l_agent + l_obs) ** degree)
                    + diffey ** degree / ((w_agent + w_obs) ** degree)
                    - 1
                    - safety_margin
                    - cbf_slack[count, i]
                )
                h_next = (
                    diffs_next ** degree / ((l_agent + l_obs) ** degree)
                    + diffey_next ** degree / ((w_agent + w_obs) ** degree)
                    - 1
                    - safety_margin
                    - cbf_slack[count, i + 1]
                )
                opti.subject_to(h_next - h >= -self.mpc_cbf_param.alpha * h)
                opti.subject_to(cbf_slack[count, i] >= 0)
                cost += 10000 * cbf_slack[count, i]
            opti.subject_to(cbf_slack[count, i + 1] >= 0)
            cost += 10000 * cbf_slack[count, i + 1]
        # dynamics + state/input constraints
        for i in range(self.mpc_cbf_param.num_horizon):
            # system dynamics
            opti.subject_to(
                xvar[:, i + 1]
                == ca.mtimes(self.mpc_cbf_param.matrix_A, xvar[:, i])
                + ca.mtimes(self.mpc_cbf_param.matrix_B, uvar[:, i])
            )
            # min and max of delta
            opti.subject_to(-self.system_param.delta_max <= uvar[0, i])
            opti.subject_to(uvar[0, i] <= self.system_param.delta_max)
            # min and max of a
            opti.subject_to(-self.system_param.a_max <= uvar[1, i])
            opti.subject_to(uvar[1, i] <= self.system_param.a_max)
            # input cost
            cost += ca.mtimes(uvar[:, i].T,
                            ca.mtimes(self.mpc_cbf_param.matrix_R, uvar[:, i]))
        for i in range(self.mpc_cbf_param.num_horizon + 1):
            # speed vx upper bound
            opti.subject_to(xvar[0, i] <= self.system_param.v_max)
            opti.subject_to(xvar[0, i] >= self.system_param.v_min)
            # min and max of ey
            opti.subject_to(xvar[5, i] <= self.track.width)
            opti.subject_to(-self.track.width <= xvar[5, i])
            # state cost
            cost += ca.mtimes(
                (xvar[:, i] - xtarget).T,
                ca.mtimes(self.mpc_cbf_param.matrix_Q, xvar[:, i] - xtarget),
            )
        # setup solver
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.minimize(cost)
        opti.solver("ipopt", option)
        sol = opti.solve()
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("solver time: {}".format(solver_time))
        x_pred = sol.value(xvar).T
        u_pred = sol.value(uvar).T
        return u_pred[0, :]

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(X_DIM, 1)
        # determine if it is a real-time simulator
        if self.realtime_flag == False:
            self.u = self._mpccbf(
                self.x,
                xtarget,
                self.mpc_cbf_param,
                self.racing_sim.vehicles,
                self.agent_name,
                self.racing_sim.track.lap_length,
                self.time,
                self.timestep,
                self.realtime_flag,
                self.track,
                self.system_param,
            )
        elif self.realtime_flag == True:
            self.u = self._mpccbf(xtarget)
        else:
            pass
        if self.agent_name == "ego":
            if self.realtime_flag == False:
                vehicles = self.racing_sim.vehicles
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

