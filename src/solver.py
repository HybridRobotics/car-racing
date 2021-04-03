import datetime
import numpy as np
import casadi as ca


def pid(xcurv, xtarget, udim):
    start_timer = datetime.datetime.now()
    u_next = np.zeros(udim)
    vt = xtarget[0]
    eyt = xtarget[5]
    u_next[0] = (
        -0.6 * (xcurv[5] - eyt)
        - 0.9 * xcurv[3]  # + np.maximum(-0.9, np.minimum(np.random.randn() * 0.25, 0.9))
    )
    u_next[1] = 1.5 * (vt - xcurv[0])  # + np.maximum(-0.8, np.minimum(np.random.randn() * 0.80, 0.8))
    end_timer = datetime.datetime.now()
    solver_time = (end_timer - start_timer).total_seconds()
    print("solver time: {}".format(solver_time))
    return u_next


def mpc(xcurv, xtarget, udim, num_of_horizon, matrix_A, matrix_B, matrix_Q, matrix_R):
    vt = xtarget[0]
    eyt = xtarget[5]
    start_timer = datetime.datetime.now()
    opti = ca.Opti()
    # define variables
    xvar = opti.variable(len(xcurv), num_of_horizon + 1)
    uvar = opti.variable(udim, num_of_horizon)
    cost = 0
    opti.subject_to(xvar[:, 0] == xcurv)
    # dynamics + state/input constraints
    for i in range(num_of_horizon):
        # system dynamics
        opti.subject_to(
            xvar[:, i + 1] == ca.mtimes(matrix_A, xvar[:, i]) + ca.mtimes(matrix_B, uvar[:, i])
        )
        # min and max of delta
        opti.subject_to(-0.5 <= uvar[0, i])
        opti.subject_to(uvar[0, i] <= 0.5)
        # min and max of a
        opti.subject_to(-1.0 <= uvar[1, i])
        opti.subject_to(uvar[1, i] <= 1.0)
        # input cost
        cost += ca.mtimes(uvar[:, i].T, ca.mtimes(matrix_R, uvar[:, i]))
    for i in range(num_of_horizon + 1):
        # speed vx upper bound
        opti.subject_to(xvar[0, i] <= 10.0)
        opti.subject_to(xvar[0, i] >= 0.0)
        # min and max of ey
        opti.subject_to(xvar[5, i] <= 2.0)
        opti.subject_to(-2.0 <= xvar[5, i])
        # state cost
        cost += ca.mtimes((xvar[:, i] - xtarget).T, ca.mtimes(matrix_Q, xvar[:, i] - xtarget))
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


def mpccbf(xcurv, xtarget, udim, num_of_horizon, matrix_A, matrix_B, matrix_Q, matrix_R, vehicles, agent_name, lap_length, time, timestep, alpha):
    vt = xtarget[0]
    eyt = xtarget[5]
    start_timer = datetime.datetime.now()
    opti = ca.Opti()
    # define variables
    xvar = opti.variable(len(xcurv), num_of_horizon + 1)
    uvar = opti.variable(udim, num_of_horizon)
    cost = 0
    opti.subject_to(xvar[:, 0] == xcurv)
    # get other vehicles' state estimations
    safety_time = 2.0
    dist_margin_front = xcurv[0] * safety_time
    dist_margin_behind = xcurv[0] * safety_time
    num_cycle_ego = int(xcurv[4] / lap_length)
    dist_ego = xcurv[4] - num_cycle_ego * lap_length
    obs_infos = {}
    for name in vehicles:
        if name != agent_name:
            # get predictions from other vehicles
            obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                time, timestep, num_of_horizon + 1
            )
            # check whether the obstacle is nearby, not consider it if not
            num_cycle_obs = int(obs_traj[4, 0] / lap_length)
            dist_obs = obs_traj[4, 0] - num_cycle_obs * lap_length
            if (dist_ego > dist_obs - dist_margin_front) & (dist_ego < dist_obs + dist_margin_behind):
                obs_infos[name] = obs_traj
    # slack variables for control barrier functions
    cbf_slack = opti.variable(len(obs_infos), num_of_horizon + 1)
    # obstacle avoidance
    safety_margin = 0.15
    degree = 6  # 2, 4, 6, 8
    for count, obs_name in enumerate(obs_infos):
        obs_traj = obs_infos[obs_name]
        # get ego agent and obstacles' dimensions
        l_agent = vehicles[agent_name].param.length / 2
        w_agent = vehicles[agent_name].param.width / 2
        l_obs = vehicles[obs_name].param.length / 2
        w_obs = vehicles[obs_name].param.width / 2
        # calculate control barrier functions for each obstacle at timestep
        for i in range(num_of_horizon):
            num_cycle_obs = int(obs_traj[4, 0] / lap_length)
            diffs = xvar[4, i] - obs_traj[4, i] - (num_cycle_ego - num_cycle_obs) * lap_length
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
            opti.subject_to(h_next - h >= -alpha * h)
            opti.subject_to(cbf_slack[count, i] >= 0)
            cost += 10000 * cbf_slack[count, i]
        opti.subject_to(cbf_slack[count, i + 1] >= 0)
        cost += 10000 * cbf_slack[count, i + 1]
    # dynamics + state/input constraints
    for i in range(num_of_horizon):
        # system dynamics
        opti.subject_to(
            xvar[:, i + 1] == ca.mtimes(matrix_A, xvar[:, i]) + ca.mtimes(matrix_B, uvar[:, i])
        )
        # min and max of delta
        opti.subject_to(-0.5 <= uvar[0, i])
        opti.subject_to(uvar[0, i] <= 0.5)
        # min and max of a
        opti.subject_to(-1.0 <= uvar[1, i])
        opti.subject_to(uvar[1, i] <= 1.0)
        # input cost
        cost += ca.mtimes(uvar[:, i].T, ca.mtimes(matrix_R, uvar[:, i]))
    for i in range(num_of_horizon + 1):
        # speed vx upper bound
        opti.subject_to(xvar[0, i] <= 10.0)
        opti.subject_to(xvar[0, i] >= 0.0)
        # min and max of ey
        opti.subject_to(xvar[5, i] <= 2.0)
        opti.subject_to(-2.0 <= xvar[5, i])
        # state cost
        cost += ca.mtimes((xvar[:, i] - xtarget).T, ca.mtimes(matrix_Q, xvar[:, i] - xtarget))
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