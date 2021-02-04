import datetime
import numpy as np
import casadi as ca


class ControlPolicyBase:
    def __init__(self):
        self.agent_name = None
        self.xdim = 6
        self.udim = 2
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.u = None

    def set_racing_sim(self, racing_sim):
        self.racing_sim = racing_sim

    def set_timestep(self, timestep):
        self.timestep = timestep

    def set_target_speed(self, vt):
        self.vt = vt

    def set_target_deviation(self, eyt):
        self.eyt = eyt

    def set_state(self, x0):
        self.x = x0

    def calc_input(self):
        pass

    def get_input(self):
        return self.u


class PIDTracking(ControlPolicyBase):
    def __init__(self, vt=0.6, eyt=0.0):
        ControlPolicyBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)

    def calc_input(self):
        """Computes control action
        Arguments:
            x0: current state position
        """
        u_next = np.zeros(self.udim)
        u_next[0] = (
            -0.6 * (self.x[5] - self.eyt)
            - 0.9 * self.x[3]  # + np.maximum(-0.9, np.minimum(np.random.randn() * 0.25, 0.9))
        )
        u_next[1] = 1.5 * (self.vt - self.x[0])  # + np.maximum(-0.8, np.minimum(np.random.randn() * 0.80, 0.8))
        self.u = u_next
        self.time += self.timestep


class MPCTracking(ControlPolicyBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        ControlPolicyBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)
        self.num_of_horizon = 10
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R

    def calc_input(self):
        xt = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(self.xdim, 1)
        start_timer = datetime.datetime.now()
        opti = ca.Opti()
        # define variables
        xvar = opti.variable(self.xdim, self.num_of_horizon + 1)
        uvar = opti.variable(self.udim, self.num_of_horizon)
        cost = 0
        opti.subject_to(xvar[:, 0] == self.x)
        # dynamics + state/input constraints
        for i in range(self.num_of_horizon):
            # system dynamics
            opti.subject_to(
                xvar[:, i + 1] == ca.mtimes(self.matrix_A, xvar[:, i]) + ca.mtimes(self.matrix_B, uvar[:, i])
            )
            # min and max of delta
            opti.subject_to(-0.5 <= uvar[0, i])
            opti.subject_to(uvar[0, i] <= 0.5)
            # min and max of a
            opti.subject_to(-1.0 <= uvar[1, i])
            opti.subject_to(uvar[1, i] <= 1.0)
            # input cost
            cost += ca.mtimes(uvar[:, i].T, ca.mtimes(self.matrix_R, uvar[:, i]))
        for i in range(self.num_of_horizon + 1):
            # speed vx upper bound
            opti.subject_to(xvar[0, i] <= 10.0)
            opti.subject_to(xvar[0, i] >= 0.0)
            # min and max of ey
            opti.subject_to(xvar[5, i] <= 2.0)
            opti.subject_to(-2.0 <= xvar[5, i])
            # state cost
            cost += ca.mtimes((xvar[:, i] - xt).T, ca.mtimes(self.matrix_Q, xvar[:, i] - xt))
        # setup solver
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.minimize(cost)
        opti.solver("ipopt", option)
        sol = opti.solve()
        end_timer = datetime.datetime.now()
        solver_time = end_timer - start_timer
        print("solver time: ", (end_timer - start_timer).total_seconds())
        self.x_pred = sol.value(xvar).T
        self.u_pred = sol.value(uvar).T
        self.u = self.u_pred[0, :]
        self.time += self.timestep


class MPCCBFRacing(ControlPolicyBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        ControlPolicyBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)
        self.num_of_horizon = 10
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.alpha = 0.6

    def calc_input(self):
        xt = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(self.xdim, 1)
        start_timer = datetime.datetime.now()
        opti = ca.Opti()
        # define variables
        xvar = opti.variable(self.xdim, self.num_of_horizon + 1)
        uvar = opti.variable(self.udim, self.num_of_horizon)
        cost = 0
        opti.subject_to(xvar[:, 0] == self.x)
        # get other vehicles' state estimations
        safety_time = 2.0
        dist_margin_front = self.x[0] * safety_time
        dist_margin_behind = self.x[0] * safety_time
        num_cycle_ego = int(self.x[4] / self.racing_sim.track.lap_length)
        dist_ego = self.x[4] - num_cycle_ego * self.racing_sim.track.lap_length
        obs_infos = {}
        for name in self.racing_sim.vehicles:
            if name != self.agent_name:
                # get predictions from other vehicles
                obs_traj, _ = self.racing_sim.vehicles[name].get_trajectory_nsteps(
                    self.time, self.timestep, self.num_of_horizon + 1
                )
                # check whether the obstacle is nearby, not consider it if not
                num_cycle_obs = int(obs_traj[4, 0] / self.racing_sim.track.lap_length)
                dist_obs = obs_traj[4, 0] - num_cycle_obs * self.racing_sim.track.lap_length
                if (dist_ego > dist_obs - dist_margin_front) & (dist_ego < dist_obs + dist_margin_behind):
                    obs_infos[name] = obs_traj
        # slack variables for control barrier functions
        cbf_slack = opti.variable(len(obs_infos), self.num_of_horizon + 1)
        # obstacle avoidance
        degree = 2
        for count, obs_name in enumerate(obs_infos):
            obs_traj = obs_infos[obs_name]
            # get ego agent and obstacles' dimensions
            obs_sdim = (
                self.racing_sim.vehicles[self.agent_name].param.length / 2
                + self.racing_sim.vehicles[obs_name].param.length / 2
            )
            obs_eydim = (
                self.racing_sim.vehicles[self.agent_name].param.width / 2
                + self.racing_sim.vehicles[obs_name].param.width / 2
            )
            # calculate control barrier functions for each obstacle at timestep
            for i in range(self.num_of_horizon):
                diffs = xvar[4, i] - obs_traj[4, i]
                diffey = xvar[5, i] - obs_traj[5, i]
                diffs_next = xvar[4, i + 1] - obs_traj[4, i + 1]
                diffey_next = xvar[5, i + 1] - obs_traj[5, i + 1]
                h = (
                    diffs ** degree / (obs_sdim ** degree)
                    + diffey ** degree / (obs_eydim ** degree)
                    - 1
                    - cbf_slack[count, i]
                )
                h_next = (
                    diffs ** degree / (obs_sdim ** degree)
                    + diffey ** degree / (obs_eydim ** degree)
                    - 1
                    - cbf_slack[count, i + 1]
                )
                opti.subject_to(h_next - h >= -self.alpha * h)
                opti.subject_to(cbf_slack[count, i] >= 0)
                cost += 10000 * cbf_slack[count, i]
            opti.subject_to(cbf_slack[count, i + 1] >= 0)
            cost += 10000 * cbf_slack[count, i + 1]
        # dynamics + state/input constraints
        for i in range(self.num_of_horizon):
            # system dynamics
            opti.subject_to(
                xvar[:, i + 1] == ca.mtimes(self.matrix_A, xvar[:, i]) + ca.mtimes(self.matrix_B, uvar[:, i])
            )
            # min and max of delta
            opti.subject_to(-0.5 <= uvar[0, i])
            opti.subject_to(uvar[0, i] <= 0.5)
            # min and max of a
            opti.subject_to(-1.0 <= uvar[1, i])
            opti.subject_to(uvar[1, i] <= 1.0)
            # input cost
            cost += ca.mtimes(uvar[:, i].T, ca.mtimes(self.matrix_R, uvar[:, i]))
        for i in range(self.num_of_horizon + 1):
            # speed vx upper bound
            opti.subject_to(xvar[0, i] <= 10.0)
            opti.subject_to(xvar[0, i] >= 0.0)
            # min and max of ey
            opti.subject_to(xvar[5, i] <= 2.0)
            opti.subject_to(-2.0 <= xvar[5, i])
            # state cost
            cost += ca.mtimes((xvar[:, i] - xt).T, ca.mtimes(self.matrix_Q, xvar[:, i] - xt))
        # setup solver
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.minimize(cost)
        opti.solver("ipopt", option)
        sol = opti.solve()
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("solver time: ", solver_time)
        self.x_pred = sol.value(xvar).T
        self.u_pred = sol.value(uvar).T
        self.u = self.u_pred[0, :]
        self.time += self.timestep
