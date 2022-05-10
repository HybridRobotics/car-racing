import datetime
import numpy as np
import casadi as ca
from control import lmpc_helper
from planning.planner_helper import *
from utils.constants import *
from casadi import *
from scipy.sparse import vstack
from scipy.interpolate import interp1d


def pid(xcurv, xtarget):
    start_timer = datetime.datetime.now()
    u_next = np.zeros((U_DIM,))
    vt = xtarget[0]
    eyt = xtarget[5]
    u_next[0] = -0.6 * (xcurv[5] - eyt) - 0.9 * xcurv[3]
    u_next[1] = 1.5 * (vt - xcurv[0])
    end_timer = datetime.datetime.now()
    solver_time = (end_timer - start_timer).total_seconds()
    print("solver time: {}".format(solver_time))
    return u_next


def mpc_lti(xcurv, xtarget, mpc_lti_param, system_param, track):
    vt = xtarget[0]
    eyt = xtarget[5]
    num_horizon = mpc_lti_param.num_horizon
    start_timer = datetime.datetime.now()
    opti = ca.Opti()
    # define variables
    xvar = opti.variable(X_DIM, num_horizon + 1)
    uvar = opti.variable(U_DIM, num_horizon)
    cost = 0
    opti.subject_to(xvar[:, 0] == xcurv)
    # dynamics + state/input constraints
    for i in range(num_horizon):
        # system dynamics
        opti.subject_to(
            xvar[:, i + 1]
            == ca.mtimes(mpc_lti_param.matrix_A, xvar[:, i])
            + ca.mtimes(mpc_lti_param.matrix_B, uvar[:, i])
        )
        # min and max of delta
        opti.subject_to(-system_param.delta_max <= uvar[0, i])
        opti.subject_to(uvar[0, i] <= system_param.delta_max)
        # min and max of a
        opti.subject_to(-system_param.a_max <= uvar[1, i])
        opti.subject_to(uvar[1, i] <= system_param.a_max)
        # input cost
        cost += ca.mtimes(uvar[:, i].T,
                          ca.mtimes(mpc_lti_param.matrix_R, uvar[:, i]))
    for i in range(num_horizon + 1):
        # speed vx upper bound
        opti.subject_to(xvar[0, i] <= system_param.v_max)
        opti.subject_to(xvar[0, i] >= system_param.v_min)
        # min and max of ey
        opti.subject_to(xvar[5, i] <= track.width)
        opti.subject_to(-track.width <= xvar[5, i])
        # state cost
        cost += ca.mtimes(
            (xvar[:, i] - xtarget).T,
            ca.mtimes(mpc_lti_param.matrix_Q, xvar[:, i] - xtarget),
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


def mpc_multi_agents(
    xcurv,
    mpc_lti_param,
    track,
    matrix_Atv,
    matrix_Btv,
    matrix_Ctv,
    system_param,
    target_traj_xcurv=None,
    vehicles=None,
    agent_name=None,
    direction_flag=None,
    target_traj_xglob=None,
    sorted_vehicles=None,
    time=None,
):
    print("overtaking")
    num_horizon = mpc_lti_param.num_horizon_ctrl
    start_timer = datetime.datetime.now()
    opti = ca.Opti()
    # define variables
    xvar = opti.variable(X_DIM, num_horizon + 1)
    uvar = opti.variable(U_DIM, num_horizon)
    cost = 0
    opti.subject_to(xvar[:, 0] == xcurv)
    vx = xcurv[0]
    f_traj = interp1d(target_traj_xcurv[:, 4], target_traj_xcurv[:, 5])
    veh_len = vehicles["ego"].param.length
    veh_width = vehicles["ego"].param.width
    # CBF parameter
    CBF_Flag = True
    ABC_Flag = True
    if CBF_Flag:
        safety_time = 2.0
        alpha = 0.6
        dist_margin_front = xcurv[0] * safety_time
        dist_margin_behind = xcurv[0] * safety_time
        realtime_flag = False
        obs_infos = {}
        timestep = 0.1
        num_cycle_ego = int(xcurv[4] / track.lap_length)
        dist_ego = xcurv[4] - num_cycle_ego * track.lap_length
        for name in sorted_vehicles:
            if name != agent_name:
                if realtime_flag == False:
                    obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                        time, timestep, num_horizon + 1
                    )
                elif realtime_flag == True:
                    obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                        num_horizon + 1)
                else:
                    pass
                num_cycle_obs = int(obs_traj[4, 0] / track.lap_length)
                dist_obs = obs_traj[4, 0] - num_cycle_obs * track.lap_length
                if (dist_ego > dist_obs - dist_margin_front) & (
                    dist_ego < dist_obs + dist_margin_behind
                ):
                    obs_infos[name] = obs_traj
        cbf_slack = opti.variable(len(obs_infos), num_horizon + 1)
        safety_margin = 0.15
        degree = 6  # 2, 4, 6, 8
        for count, obs_name in enumerate(obs_infos):
            obs_traj = obs_infos[obs_name]
            # get ego agent and obstacles' dimensions
            l_agent = veh_len / 2
            w_agent = veh_width / 2
            l_obs = veh_len / 2
            w_obs = veh_width / 2
            # calculate control barrier functions for each obstacle at timestep
            for i in range(num_horizon):
                num_cycle_obs = int(obs_traj[4, 0] / track.lap_length)
                diffs = (
                    xvar[4, i] - obs_traj[4, i] -
                        (num_cycle_ego - num_cycle_obs) * track.lap_length
                )
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
    for i in range(num_horizon):
        # system dynamics
        opti.subject_to(
            xvar[:, i + 1]
            == ca.mtimes(mpc_lti_param.matrix_A, xvar[:, i])
            + ca.mtimes(mpc_lti_param.matrix_B, uvar[:, i])
        )
        # min and max of delta
        opti.subject_to(-system_param.delta_max <= uvar[0, i])
        opti.subject_to(uvar[0, i] <= system_param.delta_max)
        # min and max of a
        opti.subject_to(-system_param.a_max <= uvar[1, i])
        opti.subject_to(uvar[1, i] <= system_param.a_max)
        # input cost
        cost += ca.mtimes(uvar[:, i].T,
                          ca.mtimes(mpc_lti_param.matrix_R, uvar[:, i]))
    for i in range(num_horizon + 1):
        # speed vx upper bound
        opti.subject_to(xvar[0, i] <= system_param.v_max)
        opti.subject_to(xvar[0, i] >= system_param.v_min)
        # min and max of ey
        opti.subject_to(xvar[5, i] <= track.width)
        opti.subject_to(-track.width <= xvar[5, i])
        s_tmp = vx * 0.1 * i + xcurv[4]
        if s_tmp < target_traj_xcurv[0, 4]:
            s_tmp = target_traj_xcurv[0, 4]
        if s_tmp >= target_traj_xcurv[-1, 4]:
            s_tmp = target_traj_xcurv[-1, 4]
        xtarget = np.array([vx, 0, 0, 0, 0, f_traj(s_tmp)])
        cost += ca.mtimes(
            (xvar[:, i] - xtarget).T,
            ca.mtimes(mpc_lti_param.matrix_Q, xvar[:, i] - xtarget),
        )
    for i in range(num_horizon + 1):
        # constraint on the left, first line is the track boundary
        s_tmp = vx * 0.1 * i + xcurv[4]
        if direction_flag == 0:
            pass
        else:
            name = sorted_vehicles[direction_flag - 1]
            epsi_agent = vehicles[name].xcurv[3]
            s_agent = vehicles[name].xcurv[4]
            while s_agent > track.lap_length:
                s_agent = s_agent - track.lap_length
            s_veh = s_agent
            epsi_veh = epsi_agent
            ey_veh = vehicles[name].xcurv[5]
            ey_veh_max, ey_veh_min, s_veh_max, s_veh_min = get_agent_range(
                s_veh, ey_veh, epsi_veh, veh_len, veh_width
            )
            ey_ego_max, ey_ego_min, s_ego_max, s_ego_min = get_agent_range(
                s_tmp, xcurv[5], xcurv[3], veh_len, veh_width
            )
            ego_agent_overlap_flag = ego_agent_overlap_checker(
                s_ego_min, s_ego_max, s_veh_min, s_veh_max, track.lap_length
            )
            if ego_agent_overlap_flag:
                if CBF_Flag:
                    pass
                else:
                    opti.subject_to(
                        xvar[5, i]
                        + 0.5 * veh_len * np.sin(xvar[3, i])
                        + 0.5 * veh_width * np.cos(xvar[3, i])
                        <= 1.2 * ey_veh_min
                    )
        if direction_flag == np.size(sorted_vehicles):
            pass
        else:
            name = sorted_vehicles[direction_flag]
            epsi_agent = vehicles[name].xcurv[3]
            s_agent = vehicles[name].xcurv[4]
            while s_agent > track.lap_length:
                s_agent = s_agent - track.lap_length
            s_veh = s_agent
            epsi_veh = epsi_agent
            ey_veh = vehicles[name].xcurv[5]
            ey_veh_max, ey_veh_min, s_veh_max, s_veh_min = get_agent_range(
                s_veh, ey_veh, epsi_veh, veh_len, veh_width
            )
            ey_ego_max, ey_ego_min, s_ego_max, s_ego_min = get_agent_range(
                s_tmp, xcurv[5], xcurv[3], veh_len, veh_width
            )
            ego_agent_overlap_flag = ego_agent_overlap_checker(
                s_ego_min, s_ego_max, s_veh_min, s_veh_max, track.lap_length
            )
            if ego_agent_overlap_flag:
                if CBF_Flag:
                    pass
                else:
                    opti.subject_to(
                        xvar[5, i]
                        - 0.5 * veh_len * np.sin(xvar[3, i])
                        - 0.5 * veh_width * np.cos(xvar[3, i])
                        >= 1.2 * ey_veh_max
                    )
    # setup solver
    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.minimize(cost)
    opti.solver("ipopt", option)
    try:
        sol = opti.solve()
        x_pred = sol.value(xvar).T
        u_pred = sol.value(uvar).T
        lin_points = np.concatenate(
            (sol.value(xvar).T[1:, :], np.array([sol.value(xvar).T[-1, :]])), axis=0
        )
        lin_input = np.vstack((u_pred[1:, :], u_pred[-1, :]))
    except RuntimeError:
        print("solver fail")
        lin_points = np.concatenate(
            (
                opti.debug.value(xvar).T[1:, :],
                np.array([opti.debug.value(xvar).T[-1, :]]),
            ),
            axis=0,
        )
        x_pred = opti.debug.value(xvar).T
        u_pred = opti.debug.value(uvar).T
        lin_input = np.vstack((u_pred[1:, :], u_pred[-1, :]))
    end_timer = datetime.datetime.now()
    solver_time = (end_timer - start_timer).total_seconds()
    print("solver time: {}".format(solver_time))
    return u_pred[0, :], x_pred


def mpccbf(
    xcurv,
    xtarget,
    mpc_cbf_param,
    vehicles,
    agent_name,
    lap_length,
    time,
    timestep,
    realtime_flag,
    track,
    system_param,
):
    vt = xtarget[0]
    eyt = xtarget[5]
    start_timer = datetime.datetime.now()
    opti = ca.Opti()
    # define variables
    xvar = opti.variable(X_DIM, mpc_cbf_param.num_horizon + 1)
    uvar = opti.variable(U_DIM, mpc_cbf_param.num_horizon)
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
                    time, timestep, mpc_cbf_param.num_horizon + 1
                )
            elif realtime_flag == True:
                obs_traj, _ = vehicles[name].get_trajectory_nsteps(
                    mpc_cbf_param.num_horizon + 1)
            else:
                pass
            # check whether the obstacle is nearby, not consider it if not
            num_cycle_obs = int(obs_traj[4, 0] / lap_length)
            dist_obs = obs_traj[4, 0] - num_cycle_obs * lap_length
            if (dist_ego > dist_obs - dist_margin_front) & (
                dist_ego < dist_obs + dist_margin_behind
            ):
                obs_infos[name] = obs_traj
    # slack variables for control barrier functions
    cbf_slack = opti.variable(len(obs_infos), mpc_cbf_param.num_horizon + 1)
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
        for i in range(mpc_cbf_param.num_horizon):
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
            opti.subject_to(h_next - h >= -mpc_cbf_param.alpha * h)
            opti.subject_to(cbf_slack[count, i] >= 0)
            cost += 10000 * cbf_slack[count, i]
        opti.subject_to(cbf_slack[count, i + 1] >= 0)
        cost += 10000 * cbf_slack[count, i + 1]
    # dynamics + state/input constraints
    for i in range(mpc_cbf_param.num_horizon):
        # system dynamics
        opti.subject_to(
            xvar[:, i + 1]
            == ca.mtimes(mpc_cbf_param.matrix_A, xvar[:, i])
            + ca.mtimes(mpc_cbf_param.matrix_B, uvar[:, i])
        )
        # min and max of delta
        opti.subject_to(-system_param.delta_max <= uvar[0, i])
        opti.subject_to(uvar[0, i] <= system_param.delta_max)
        # min and max of a
        opti.subject_to(-system_param.a_max <= uvar[1, i])
        opti.subject_to(uvar[1, i] <= system_param.a_max)
        # input cost
        cost += ca.mtimes(uvar[:, i].T,
                          ca.mtimes(mpc_cbf_param.matrix_R, uvar[:, i]))
    for i in range(mpc_cbf_param.num_horizon + 1):
        # speed vx upper bound
        opti.subject_to(xvar[0, i] <= system_param.v_max)
        opti.subject_to(xvar[0, i] >= system_param.v_min)
        # min and max of ey
        opti.subject_to(xvar[5, i] <= track.width)
        opti.subject_to(-track.width <= xvar[5, i])
        # state cost
        cost += ca.mtimes(
            (xvar[:, i] - xtarget).T,
            ca.mtimes(mpc_cbf_param.matrix_Q, xvar[:, i] - xtarget),
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


def lmpc(
    xcurv,
    lmpc_param,
    matrix_Atv,
    matrix_Btv,
    matrix_Ctv,
    ss_curv,
    Qfun,
    iter,
    lap_length,
    lap_width,
    u_old,
    system_param,
):
    start_timer = datetime.datetime.now()
    ss_point_selected_tot = np.empty((X_DIM, 0))
    Qfun_selected_tot = np.empty((0))
    for jj in range(0, lmpc_param.num_ss_iter):
        ss_point_selected, Qfun_selected = lmpc_helper.select_points(
            ss_curv,
            Qfun,
            iter - jj - 1,
            xcurv,
            lmpc_param.num_ss_points / lmpc_param.num_ss_iter,
            lmpc_param.shift,
        )
        ss_point_selected_tot = np.append(
            ss_point_selected_tot, ss_point_selected, axis=1)
        Qfun_selected_tot = np.append(Qfun_selected_tot, Qfun_selected, axis=0)
    # initialize the problem
    opti = ca.Opti()
    # define variables
    x = opti.variable(X_DIM, lmpc_param.num_horizon + 1)
    u = opti.variable(U_DIM, lmpc_param.num_horizon)
    lambd = opti.variable(Qfun_selected_tot.shape[0])
    slack = opti.variable(X_DIM)
    cost_mpc = 0
    cost_learning = 0
    # add constraints and cost function
    x_track = np.array([5.0, 0, 0, 0, 0, 0]).reshape(X_DIM, 1)
    opti.subject_to(x[:, 0] == xcurv)
    # state/input constraints
    for i in range(lmpc_param.num_horizon):
        opti.subject_to(
            x[:, i + 1]
            == mtimes(matrix_Atv[i], x[:, i]) + mtimes(matrix_Btv[i], u[:, i]) + matrix_Ctv[i]
        )
        # min and max of ey
        opti.subject_to(x[0, i] <= system_param.v_max)
        opti.subject_to(x[5, i] <= lap_width)
        opti.subject_to(-lap_width <= x[5, i])
        # min and max of delta
        opti.subject_to(-system_param.delta_max <= u[0, i])
        opti.subject_to(u[0, i] <= system_param.delta_max)
        # min and max of a
        opti.subject_to(-system_param.a_max <= u[1, i])
        opti.subject_to(u[1, i] <= system_param.a_max)
        # quadratic cost
        cost_mpc += mtimes(
            (x[:, i] - x_track).T,
            mtimes(lmpc_param.matrix_Q, x[:, i] - x_track),
        )
        cost_mpc += mtimes(u[:, i].T, mtimes(lmpc_param.matrix_R, u[:, i]))
        if i == 0:
            cost_mpc += mtimes(
                (u[:, i] - u_old.T).T,
                mtimes(lmpc_param.matrix_dR, u[:, i] - u_old.T),
            )
        else:
            cost_mpc += mtimes(
                (u[:, i] - u[:, i - 1]).T,
                mtimes(lmpc_param.matrix_dR, u[:, i] - u[:, i - 1]),
            )
    # convex hull for LMPC
    cost_mpc += mtimes(
        (x[:, lmpc_param.num_horizon] - x_track).T,
        mtimes(lmpc_param.matrix_Q, x[:, lmpc_param.num_horizon] - x_track),
    )
    cost_learning += mtimes(slack.T, mtimes(lmpc_param.matrix_Qslack, slack))
    opti.subject_to(lambd >= np.zeros(lambd.shape[0]))
    opti.subject_to(x[:, lmpc_param.num_horizon] ==
                    mtimes(ss_point_selected_tot, lambd))
    opti.subject_to(mtimes(np.ones((1, lambd.shape[0])), lambd) == 1)
    opti.subject_to(
        mtimes(np.diag([1, 1, 1, 1, 1, 1]), slack) == np.zeros(X_DIM))
    cost_learning += mtimes(np.array([Qfun_selected_tot]), lambd)
    cost = cost_mpc + cost_learning
    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.minimize(cost)
    opti.solver("ipopt", option)
    try:
        sol = opti.solve()
        lin_points = np.concatenate(
            (sol.value(x).T[1:, :], np.array([sol.value(x).T[-1, :]])), axis=0
        )
        x_pred = sol.value(x).T
        u_pred = sol.value(u).T
        lin_input = np.vstack((u_pred[1:, :], u_pred[-1, :]))
    except RuntimeError:
        lin_points = np.concatenate(
            (
                opti.debug.value(x).T[1:, :],
                np.array([opti.debug.value(x).T[-1, :]]),
            ),
            axis=0,
        )
        x_pred = opti.debug.value(x).T
        u_pred = opti.debug.value(u).T
        lin_input = np.vstack((u_pred[1:, :], u_pred[-1, :]))
        print("solver fail to find the solution, the non-converged solution is used")
    end_timer = datetime.datetime.now()
    solver_time = (end_timer - start_timer).total_seconds()
    print("solver time: {}".format(solver_time))
    return (
        u_pred,
        x_pred,
        ss_point_selected_tot,
        Qfun_selected_tot,
        lin_points,
        lin_input,
    )
