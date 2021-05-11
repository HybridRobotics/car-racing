import datetime
import numpy as np
import casadi as ca
from utils import lmpc_helper
from casadi import *
from scipy import sparse
from scipy.sparse import vstack


def pid(xcurv, xtarget, udim):
    start_timer = datetime.datetime.now()
    u_next = np.zeros(udim)
    vt = xtarget[0]
    eyt = xtarget[5]
    u_next[0] = (-0.6 * (xcurv[5] - eyt) - 0.9 * xcurv[3])
    u_next[1] = 1.5 * (vt - xcurv[0])
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
            xvar[:, i + 1] == ca.mtimes(matrix_A, xvar[:, i]) +
            ca.mtimes(matrix_B, uvar[:, i])
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
        cost += ca.mtimes((xvar[:, i] - xtarget).T,
                          ca.mtimes(matrix_Q, xvar[:, i] - xtarget))
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


def mpccbf(xcurv, xtarget, udim, num_of_horizon, matrix_A, matrix_B, matrix_Q, matrix_R, vehicles, agent_name, lap_length, time, timestep, alpha, realtime_flag):
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
    for name in list(vehicles):
        if name != agent_name:
            # get predictions from other vehicles
            if realtime_flag == False:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                    time, timestep, num_of_horizon + 1
                )
            elif realtime_flag == True:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                    num_of_horizon + 1)
            else:
                pass
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
            diffs = xvar[4, i] - obs_traj[4, i] - \
                (num_cycle_ego - num_cycle_obs) * lap_length
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
            xvar[:, i + 1] == ca.mtimes(matrix_A, xvar[:, i]) +
            ca.mtimes(matrix_B, uvar[:, i])
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
        cost += ca.mtimes((xvar[:, i] - xtarget).T,
                          ca.mtimes(matrix_Q, xvar[:, i] - xtarget))
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


def lmpc(xcurv, matrix_Atv, matrix_Btv, matrix_Ctv, matrix_R_LMPC, matrix_Q_LMPC, matrix_dR_LMPC, matrix_Qslack, xdim, udim, N, num_ss_iter, ss_curv, Qfun, iter, num_ss_points, shift, lap_length, lap_width, u_old):
    start_timer = datetime.datetime.now()
    ss_point_selected_tot = np.empty((xdim, 0))
    Qfun_selected_tot = np.empty((0))
    for jj in range(0, num_ss_iter):
        ss_point_selected, Qfun_selected = lmpc_helper.select_points(
            ss_curv, Qfun, iter - jj - 1, xcurv, num_ss_points / num_ss_iter, shift)
        ss_point_selected_tot = np.append(
            ss_point_selected_tot, ss_point_selected, axis=1)
        Qfun_selected_tot = np.append(Qfun_selected_tot, Qfun_selected, axis=0)
    # initialize the problem
    opti = ca.Opti()
    # define variables
    x = opti.variable(xdim, N+1)
    u = opti.variable(udim, N)
    lambd = opti.variable(Qfun_selected_tot.shape[0])
    slack = opti.variable(xdim)
    cost_mpc = 0
    cost_learning = 0
    # add constraints and cost function
    x_track = np.array([2.0, 0, 0, 0, 0, 0]).reshape(xdim, 1)
    opti.subject_to(x[:, 0] == xcurv)
    for i in range(N):
        opti.subject_to(x[:, i+1] == mtimes(matrix_Atv[i], x[:, i]) +
                        mtimes(matrix_Btv[i], u[:, i]) + matrix_Ctv[i])
        # min and max of ey
        opti.subject_to(x[0, i] <= 3.0)
        opti.subject_to(x[5, i] <= lap_width)
        opti.subject_to(-lap_width <= x[5, i])
        # min and max of delta
        opti.subject_to(-0.5 <= u[0, i])
        opti.subject_to(u[0, i] <= 0.5)
        # min and max of a
        opti.subject_to(-1.0 <= u[1, i])
        opti.subject_to(u[1, i] <= 1.0)
        # quadratic cost
        cost_mpc += mtimes((x[:, i]-x_track).T,
                           mtimes(matrix_Q_LMPC, x[:, i]-x_track))
        cost_mpc += mtimes(u[:, i].T, mtimes(matrix_R_LMPC, u[:, i]))
        if i == 0:
            cost_mpc += mtimes((u[:, i]-u_old.T).T,
                               mtimes(matrix_dR_LMPC, u[:, i]-u_old.T))
        else:
            cost_mpc += mtimes((u[:, i]-u[:, i-1]).T,
                               mtimes(matrix_dR_LMPC, u[:, i]-u[:, i-1]))
    # convex hull for LMPC
    cost_mpc += mtimes((x[:, N]-x_track).T,
                       mtimes(matrix_Q_LMPC, x[:, N]-x_track))
    cost_learning += mtimes(slack.T, mtimes(matrix_Qslack, slack))
    opti.subject_to(lambd >= np.zeros(lambd.shape[0]))
    opti.subject_to(x[:, N] == mtimes(ss_point_selected_tot, lambd))
    opti.subject_to(mtimes(np.ones((1, lambd.shape[0])), lambd) == 1)
    opti.subject_to(mtimes(np.diag([1, 1, 1, 1, 1, 1]), slack) == np.zeros(6))
    cost_learning += mtimes(np.array([Qfun_selected_tot]), lambd)
    cost = cost_mpc + cost_learning
    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.minimize(cost)
    opti.solver("ipopt", option)
    sol = opti.solve()
    end_timer = datetime.datetime.now()
    solver_time = (end_timer - start_timer).total_seconds()
    print("solver time: {}".format(solver_time))
    lin_points = np.concatenate(
        (sol.value(x).T[1:, :], np.array([sol.value(x).T[-1, :]])), axis=0)
    x_pred = sol.value(x).T
    u_pred = sol.value(u).T
    lin_input = np.vstack((u_pred[1:, :], u_pred[-1, :]))
    return u_pred, x_pred, ss_point_selected_tot, Qfun_selected_tot, lin_points, lin_input
