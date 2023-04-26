import copy
import datetime
from pathos.multiprocessing import ProcessingPool as Pool

import casadi as ca
from cvxopt.solvers import qp
from cvxopt import matrix
import numpy as np
from numpy import linalg as la

from planner.base import PlannerBase, get_agent_range, ego_agent_overlap_checker
from planner.overtake import OvertakePathPlanner, OvertakeTrajPlanner
from racing_env import X_DIM, U_DIM, get_curvature


def _compute_cost(xcurv, u, lap_length):
    # The cost has the same elements of the vector x --> time +1
    cost = 10000 * np.ones((xcurv.shape[0]))
    # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
    # We start from the last element of the vector x and we sum the running cost
    for i in range(0, xcurv.shape[0]):
        if i == 0:  # Note that for i = 0 --> pick the latest element of the vector x
            cost[xcurv.shape[0] - 1 - i] = 0
        elif xcurv[xcurv.shape[0] - 1 - i, 4] < lap_length:
            cost[xcurv.shape[0] - 1 - i] = cost[xcurv.shape[0] - 1 - i + 1] + 1
        else:
            cost[xcurv.shape[0] - 1 - i] = 0
    return cost


def _regression_and_linearization(
    lin_points,
    lin_input,
    used_iter,
    ss_xcurv,
    u_ss,
    time_ss,
    max_num_point,
    qp,
    matrix,
    point_and_tangent,
    dt,
    i,
):
    x0 = lin_points[i, :]
    Ai = np.zeros((X_DIM, X_DIM))
    Bi = np.zeros((X_DIM, U_DIM))
    Ci = np.zeros((X_DIM, 1))
    # Compute Index to use
    h = 5
    lamb = 0.0
    state_features = [0, 1, 2]
    consider_input = True
    if consider_input == True:
        scaling = np.array(
            [
                [0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        x_lin = np.hstack((lin_points[i, state_features], lin_input[i, :]))
    else:
        scaling = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        x_lin = lin_points[i, state_features]
    index_selected = []
    K = []
    for i in used_iter:
        index_selected_i, K_i = _compute_index(
            h,
            ss_xcurv,
            u_ss,
            time_ss,
            i,
            x_lin,
            state_features,
            scaling,
            max_num_point,
            consider_input,
        )
        index_selected.append(index_selected_i)
        K.append(K_i)
    # =========================
    # ====== Identify vx ======
    input_features = [1]
    Q_vx, M_vx = _compute_Q_M(
        ss_xcurv,
        u_ss,
        index_selected,
        state_features,
        input_features,
        used_iter,
        np,
        matrix,
        lamb,
        K,
    )
    y_index = 0
    b = compute_b(ss_xcurv, y_index, used_iter, matrix, M_vx, index_selected, K)
    (
        Ai[y_index, state_features],
        Bi[y_index, input_features],
        Ci[y_index],
    ) = _lmpc_loc_lin_reg(Q_vx, b, state_features, input_features, qp)
    # =======================================
    # ====== Identify Lateral Dynamics ======
    input_features = [0]
    Q_lat, M_lat = _compute_Q_M(
        ss_xcurv,
        u_ss,
        index_selected,
        state_features,
        input_features,
        used_iter,
        np,
        matrix,
        lamb,
        K,
    )
    y_index = 1  # vy
    b = compute_b(ss_xcurv, y_index, used_iter, matrix, M_lat, index_selected, K)
    (
        Ai[y_index, state_features],
        Bi[y_index, input_features],
        Ci[y_index],
    ) = _lmpc_loc_lin_reg(Q_lat, b, state_features, input_features, qp)
    y_index = 2  # wz
    b = compute_b(ss_xcurv, y_index, used_iter, matrix, M_lat, index_selected, K)
    (
        Ai[y_index, state_features],
        Bi[y_index, input_features],
        Ci[y_index],
    ) = _lmpc_loc_lin_reg(Q_lat, b, state_features, input_features, qp)
    # ===========================
    # ===== Linearization =======
    vx = x0[0]
    vy = x0[1]
    wz = x0[2]
    epsi = x0[3]
    s = x0[4]
    ey = x0[5]
    if s < 0:
        print("s is negative, here the state: \n", lin_points)
    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    cur = get_curvature(
        point_and_tangent[-1, 3] + point_and_tangent[-1, 4],
        point_and_tangent,
        s,
    )
    den = 1 - cur * ey
    # ===========================
    # ===== Linearize epsi ======
    # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
    depsi_vx = -dt * np.cos(epsi) / den * cur
    depsi_vy = dt * np.sin(epsi) / den * cur
    depsi_wz = dt
    depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
    depsi_s = 0  # Because cur = constant
    depsi_ey = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den ** 2) * cur * (-cur)
    Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
    Ci[3] = (
        epsi
        + dt * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur)
        - np.dot(Ai[3, :], x0)
    )
    # ===========================
    # ===== Linearize s =========
    # s_{k+1} = s    + dt * ( (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) )
    ds_vx = dt * (np.cos(epsi) / den)
    ds_vy = -dt * (np.sin(epsi) / den)
    ds_wz = 0
    ds_epsi = dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den
    # + Ts * (Vx * cos(epsi) - Vy * sin(epsi)) / (1 - ey * rho) ^ 2 * (-ey * drho);
    ds_s = 1
    ds_ey = -dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / (den * 2) * (-cur)
    Ai[4, :] = [ds_vx, ds_vy, ds_wz, ds_epsi, ds_s, ds_ey]
    Ci[4] = (
        s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey)) - np.dot(Ai[4, :], x0)
    )
    # ===========================
    # ===== Linearize ey ========
    # ey_{k+1} = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi))
    dey_vx = dt * np.sin(epsi)
    dey_vy = dt * np.cos(epsi)
    dey_wz = 0
    dey_epsi = dt * (vx * np.cos(epsi) - vy * np.sin(epsi))
    dey_s = 0
    dey_ey = 1
    Ai[5, :] = [dey_vx, dey_vy, dey_wz, dey_epsi, dey_s, dey_ey]
    Ci[5] = ey + dt * (vx * np.sin(epsi) + vy * np.cos(epsi)) - np.dot(Ai[5, :], x0)
    deltaTimer_tv = datetime.datetime.now() - startTimer
    return Ai, Bi, Ci, index_selected


def _compute_index(
    h,
    ss_xcurv,
    u_ss,
    time_ss,
    iter,
    x0,
    state_features,
    scaling,
    max_num_point,
    consider_input,
):
    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    # What to learn a model such that: x_{k+1} = A x_k  + B u_k + C
    one_vec = np.ones((ss_xcurv[0 : time_ss[iter], :, iter].shape[0] - 1, 1))
    x0_vec = (np.dot(np.array([x0]).T, one_vec.T)).T
    if consider_input == True:
        DataMatrix = np.hstack(
            (
                ss_xcurv[0 : time_ss[iter] - 1, state_features, iter],
                u_ss[0 : time_ss[iter] - 1, :, iter],
            )
        )
    else:
        DataMatrix = ss_xcurv[0 : time_ss[iter] - 1, state_features, iter]
    diff = np.dot((DataMatrix - x0_vec), scaling)
    norm = la.norm(diff, 1, axis=1)
    index_tot = np.squeeze(np.where(norm < h))
    if index_tot.shape[0] >= max_num_point:
        index = np.argsort(norm)[0:max_num_point]
    else:
        index = index_tot

    K = (1 - (norm[index] / h) ** 2) * 3 / 4
    return index, K


def _compute_Q_M(
    ss_xcurv,
    u_ss,
    index_selected,
    state_features,
    input_features,
    used_iter,
    np,
    matrix,
    lamb,
    K,
):
    counter = 0
    iter = 1
    X0 = np.empty((0, len(state_features) + len(input_features)))
    Ktot = np.empty((0))
    for iter in used_iter:
        X0 = np.append(
            X0,
            np.hstack(
                (
                    np.squeeze(ss_xcurv[np.ix_(index_selected[counter], state_features, [iter])]),
                    np.squeeze(
                        u_ss[np.ix_(index_selected[counter], input_features, [iter])],
                        axis=2,
                    ),
                )
            ),
            axis=0,
        )
        Ktot = np.append(Ktot, K[counter])
        counter = counter + 1
    M = np.hstack((X0, np.ones((X0.shape[0], 1))))
    Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
    Q = matrix(Q0 + lamb * np.eye(Q0.shape[0]))
    return Q, M


def _select_points(ss_xcurv, Qfun, iter, x0, num_ss_points, shift):
    # selects the closest point in the safe set to x0
    # returns a subset of the safe set which contains a range of points ahead of this point
    xcurv = ss_xcurv[:, :, iter]
    one_vec = np.ones((xcurv.shape[0], 1))
    x0_vec = (np.dot(np.array([x0]).T, one_vec.T)).T
    diff = xcurv - x0_vec
    norm = la.norm(diff, 1, axis=1)
    min_norm = np.argmin(norm)
    if min_norm + shift >= 0:
        ss_points = xcurv[int(shift + min_norm) : int(shift + min_norm + num_ss_points), :].T
        sel_Qfun = Qfun[int(shift + min_norm) : int(shift + min_norm + num_ss_points), iter]
    else:
        ss_points = xcurv[int(min_norm) : int(min_norm + num_ss_points), :].T
        sel_Qfun = Qfun[int(min_norm) : int(min_norm + num_ss_points), iter]
    return ss_points, sel_Qfun


def compute_b(ss_xcurv, y_index, used_iter, matrix, M, index_selected, K):
    counter = 0
    y = np.empty((0))
    Ktot = np.empty((0))
    for iter in used_iter:
        y = np.append(
            y,
            np.squeeze(ss_xcurv[np.ix_(index_selected[counter] + 1, [y_index], [iter])]),
        )
        Ktot = np.append(Ktot, K[counter])
        counter = counter + 1
    b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))
    return b


def _lmpc_loc_lin_reg(Q, b, state_features, input_features, qp):
    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    res_cons = qp(Q, b)  # This is ordered as [A B C]
    deltaTimer_tv = datetime.datetime.now() - startTimer
    result = np.squeeze(np.array(res_cons["x"]))
    A = result[0 : len(state_features)]
    B = result[len(state_features) : (len(state_features) + len(input_features))]
    C = result[-1]
    return A, B, C


class LMPCRacingParam:
    def __init__(
        self,
        matrix_Q=0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        matrix_R=1 * np.diag([1.0, 0.25]),
        matrix_Qslack=5 * np.diag([10, 0, 0, 1, 10, 0]),
        matrix_dR=5 * np.diag([0.8, 0.0]),
        num_ss_points=32 + 12,
        num_ss_iter=2,
        num_horizon=12,
        shift=0,
        timestep=None,
        lap_number=None,
        time_lmpc=None,
    ):
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.matrix_Qslack = matrix_Qslack
        self.matrix_dR = matrix_dR
        self.num_ss_points = num_ss_points
        self.num_ss_iter = num_ss_iter
        self.num_horizon = num_horizon
        self.shift = shift
        self.timestep = timestep
        self.lap_number = lap_number
        self.time_lmpc = time_lmpc

class RacingGameParam:
    def __init__(
        self,
        matrix_A=np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=","),
        matrix_B=np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=","),
        matrix_Q=np.diag([10.0, 0.0, 0.0, 5.0, 0.0, 50.0]),
        matrix_R=np.diag([0.1, 0.1]),
        matrix_R_planner=1 * np.diag([5, 0.10]),
        matrix_dR_planner=5 * np.diag([1.8, 0.0]),
        bezier_order=3,
        safety_factor=4.5,
        num_horizon_ctrl=10,
        num_horizon_planner=10,
        planning_prediction_factor=0.5,  # 2.0,
        alpha=0.98,
        timestep=None,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.matrix_R_planner = matrix_R_planner
        self.matrix_dR_planner = matrix_dR_planner
        self.num_horizon_ctrl = num_horizon_ctrl
        self.num_horizon_planner = num_horizon_planner
        self.planning_prediction_factor = planning_prediction_factor
        self.alpha = alpha
        self.timestep = timestep
        self.bezier_order = bezier_order
        self.safety_factor = safety_factor

class LMPCPrediction:
    """Object collecting the predictions and SS at eath time step"""

    def __init__(
        self,
        num_horizon=12,
        points_lmpc=5000,
        num_ss_points=32 + 12,
        lap_number=None,
    ):
        """
        Initialization:
            num_horizon: horizon length
            points_lmpc: maximum simulation timesteps
            num_ss_points: number used to build safe set at each time step
        """
        self.predicted_xcurv = np.zeros((num_horizon + 1, X_DIM, points_lmpc, lap_number))
        self.predicted_u = np.zeros((num_horizon, U_DIM, points_lmpc, lap_number))
        self.ss_used = np.zeros((X_DIM, num_ss_points, points_lmpc, lap_number))
        self.Qfun_used = np.zeros((num_ss_points, points_lmpc, lap_number))

class LMPCRacingGame(PlannerBase):
    def __init__(self, lmpc_param, racing_game_param=None, system_param=None):
        PlannerBase.__init__(self)
        self.path_planner = False
        self.lmpc_param = lmpc_param
        self.racing_game_param = racing_game_param
        self.system_param = system_param
        if self.path_planner:
            self.overtake_planner = OvertakePathPlanner(racing_game_param)
        else:
            self.overtake_planner = OvertakeTrajPlanner(racing_game_param)
        self.x_pred = None
        self.u_pred = None
        self.lin_points = None
        self.lin_input = None
        self.ss_point_selected_tot = None
        self.Qfun_selected_tot = None
        num_points = int(lmpc_param.time_lmpc / lmpc_param.timestep) + 1
        # Time at which each j-th iteration is completed
        self.time_ss = 10000 * np.ones(lmpc_param.lap_number).astype(int)
        self.ss_xcurv = 10000 * np.ones(
            (num_points, X_DIM, lmpc_param.lap_number)
        )  # Sampled Safe SS
        # Input associated with the points in SS
        self.u_ss = 10000 * np.ones((num_points, U_DIM, lmpc_param.lap_number))
        # Qfun: cost-to-go from each point in SS
        self.Qfun = 0 * np.ones((num_points, lmpc_param.lap_number))
        # SS in global (X-Y) used for plotting
        self.ss_glob = 10000 * np.ones((num_points, X_DIM, lmpc_param.lap_number))
        # Initialize the controller iteration
        self.iter = 0
        self.time_in_iter = 0
        self.p = Pool(4)
        self.openloop_prediction = None
        self.old_ey = None
        self.old_direction_flag = None

    def set_vehicles_track(self):
        if self.realtime_flag == False:
            vehicles = self.racing_sim.vehicles
            self.overtake_planner.track = self.track
        else:
            vehicles = self.vehicles
        self.overtake_planner.vehicles = vehicles

    def _lmpc(self, xcurv, matrix_Atv, matrix_Btv, matrix_Ctv, u_old):
        start_timer = datetime.datetime.now()
        ss_point_selected_tot = np.empty((X_DIM, 0))
        Qfun_selected_tot = np.empty((0))
        for jj in range(0, self.lmpc_param.num_ss_iter):
            ss_point_selected, Qfun_selected = _select_points(
                self.ss_curv,
                self.Qfun,
                iter - jj - 1,
                xcurv,
                self.lmpc_param.num_ss_points / self.lmpc_param.num_ss_iter,
                self.lmpc_param.shift,
            )
            ss_point_selected_tot = np.append(
                ss_point_selected_tot, ss_point_selected, axis=1)
            Qfun_selected_tot = np.append(Qfun_selected_tot, Qfun_selected, axis=0)
        # initialize the problem
        opti = ca.Opti()
        # define variables
        x = opti.variable(X_DIM, self.lmpc_param.num_horizon + 1)
        u = opti.variable(U_DIM, self.lmpc_param.num_horizon)
        lambd = opti.variable(Qfun_selected_tot.shape[0])
        slack = opti.variable(X_DIM)
        cost_mpc = 0
        cost_learning = 0
        # add constraints and cost function
        x_track = np.array([5.0, 0, 0, 0, 0, 0]).reshape(X_DIM, 1)
        opti.subject_to(x[:, 0] == xcurv)
        # state/input constraints
        for i in range(self.lmpc_param.num_horizon):
            opti.subject_to(
                x[:, i + 1]
                == ca.mtimes(matrix_Atv[i], x[:, i]) + ca.mtimes(matrix_Btv[i], u[:, i]) + matrix_Ctv[i]
            )
            # min and max of ey
            opti.subject_to(x[0, i] <= self.system_param.v_max)
            opti.subject_to(x[5, i] <= self.lap_width)
            opti.subject_to(-self.lap_width <= x[5, i])
            # min and max of delta
            opti.subject_to(-self.system_param.delta_max <= u[0, i])
            opti.subject_to(u[0, i] <= self.system_param.delta_max)
            # min and max of a
            opti.subject_to(-self.system_param.a_max <= u[1, i])
            opti.subject_to(u[1, i] <= self.system_param.a_max)
            # quadratic cost
            cost_mpc += ca.mtimes(
                (x[:, i] - x_track).T,
                ca.mtimes(self.lmpc_param.matrix_Q, x[:, i] - x_track),
            )
            cost_mpc += ca.mtimes(u[:, i].T, ca.mtimes(self.lmpc_param.matrix_R, u[:, i]))
            if i == 0:
                cost_mpc += ca.mtimes(
                    (u[:, i] - u_old.T).T,
                    ca.mtimes(self.lmpc_param.matrix_dR, u[:, i] - u_old.T),
                )
            else:
                cost_mpc += ca.mtimes(
                    (u[:, i] - u[:, i - 1]).T,
                    ca.mtimes(self.lmpc_param.matrix_dR, u[:, i] - u[:, i - 1]),
                )
        # convex hull for LMPC
        cost_mpc += ca.mtimes(
            (x[:, self.lmpc_param.num_horizon] - x_track).T,
            ca.mtimes(self.lmpc_param.matrix_Q, x[:, self.lmpc_param.num_horizon] - x_track),
        )
        cost_learning += ca.mtimes(slack.T, ca.mtimes(self.lmpc_param.matrix_Qslack, slack))
        opti.subject_to(lambd >= np.zeros(lambd.shape[0]))
        opti.subject_to(x[:, self.lmpc_param.num_horizon] ==
                        ca.mtimes(ss_point_selected_tot, lambd))
        opti.subject_to(ca.mtimes(np.ones((1, lambd.shape[0])), lambd) == 1)
        opti.subject_to(
            ca.mtimes(np.diag([1, 1, 1, 1, 1, 1]), slack) == np.zeros(X_DIM))
        cost_learning += ca.mtimes(np.array([Qfun_selected_tot]), lambd)
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


    def _mpc_multi_agents(
        self,
        xcurv,
        target_traj_xcurv=None,
        direction_flag=None,
        sorted_vehicles=None,
        time=None,
    ):
        print("overtaking")
        num_horizon = self.racing_game_param.num_horizon_ctrl
        start_timer = datetime.datetime.now()
        opti = ca.Opti()
        # define variables
        xvar = opti.variable(X_DIM, num_horizon + 1)
        uvar = opti.variable(U_DIM, num_horizon)
        cost = 0
        opti.subject_to(xvar[:, 0] == xcurv)
        vx = xcurv[0]
        f_traj = ca.interp1d(target_traj_xcurv[:, 4], target_traj_xcurv[:, 5])
        veh_len = self.overtake_planner.vehicles["ego"].param.length
        veh_width = self.overtake_planner.vehicles["ego"].param.width
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
            num_cycle_ego = int(xcurv[4] / self.track.lap_length)
            dist_ego = xcurv[4] - num_cycle_ego * self.track.lap_length
            for name in sorted_vehicles:
                if name != self.agent_name:
                    if realtime_flag == False:
                        obs_traj, _ = self.overtake_planner.vehicles[name].get_trajectory_nsteps(
                            time, timestep, num_horizon + 1
                        )
                    elif realtime_flag == True:
                        obs_traj, _ = self.overtake_planner.vehicles[name].get_trajectory_nsteps(
                            num_horizon + 1)
                    else:
                        pass
                    num_cycle_obs = int(obs_traj[4, 0] / self.track.lap_length)
                    dist_obs = obs_traj[4, 0] - num_cycle_obs * self.track.lap_length
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
                    num_cycle_obs = int(obs_traj[4, 0] / self.track.lap_length)
                    diffs = (
                        xvar[4, i] - obs_traj[4, i] -
                            (num_cycle_ego - num_cycle_obs) * self.track.lap_length
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
                == ca.mtimes(self.racing_game_param.matrix_A, xvar[:, i])
                + ca.mtimes(self.racing_game_param.matrix_B, uvar[:, i])
            )
            # min and max of delta
            opti.subject_to(-self.system_param.delta_max <= uvar[0, i])
            opti.subject_to(uvar[0, i] <= self.system_param.delta_max)
            # min and max of a
            opti.subject_to(-self.system_param.a_max <= uvar[1, i])
            opti.subject_to(uvar[1, i] <= self.system_param.a_max)
            # input cost
            cost += ca.mtimes(uvar[:, i].T,
                            ca.mtimes(self.racing_game_param.matrix_R, uvar[:, i]))
        for i in range(num_horizon + 1):
            # speed vx upper bound
            opti.subject_to(xvar[0, i] <= self.system_param.v_max)
            opti.subject_to(xvar[0, i] >= self.system_param.v_min)
            # min and max of ey
            opti.subject_to(xvar[5, i] <= self.track.width)
            opti.subject_to(-self.track.width <= xvar[5, i])
            s_tmp = vx * 0.1 * i + xcurv[4]
            if s_tmp < target_traj_xcurv[0, 4]:
                s_tmp = target_traj_xcurv[0, 4]
            if s_tmp >= target_traj_xcurv[-1, 4]:
                s_tmp = target_traj_xcurv[-1, 4]
            xtarget = np.array([vx, 0, 0, 0, 0, f_traj(s_tmp)])
            cost += ca.mtimes(
                (xvar[:, i] - xtarget).T,
                ca.mtimes(self.racing_game_param.matrix_Q, xvar[:, i] - xtarget),
            )
        for i in range(num_horizon + 1):
            # constraint on the left, first line is the track boundary
            s_tmp = vx * 0.1 * i + xcurv[4]
            if direction_flag == 0:
                pass
            else:
                name = sorted_vehicles[direction_flag - 1]
                epsi_agent = self.overtake_planner.vehicles[name].xcurv[3]
                s_agent = self.overtake_planner.vehicles[name].xcurv[4]
                while s_agent > self.track.lap_length:
                    s_agent = s_agent - self.track.lap_length
                s_veh = s_agent
                epsi_veh = epsi_agent
                ey_veh = self.overtake_planner.vehicles[name].xcurv[5]
                ey_veh_max, ey_veh_min, s_veh_max, s_veh_min = get_agent_range(
                    s_veh, ey_veh, epsi_veh, veh_len, veh_width
                )
                ey_ego_max, ey_ego_min, s_ego_max, s_ego_min = get_agent_range(
                    s_tmp, xcurv[5], xcurv[3], veh_len, veh_width
                )
                ego_agent_overlap_flag = ego_agent_overlap_checker(
                    s_ego_min, s_ego_max, s_veh_min, s_veh_max, self.track.lap_length
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
                epsi_agent = self.overtake_planner.vehicles[name].xcurv[3]
                s_agent = self.overtake_planner.vehicles[name].xcurv[4]
                while s_agent > self.track.lap_length:
                    s_agent = s_agent - self.track.lap_length
                s_veh = s_agent
                epsi_veh = epsi_agent
                ey_veh = self.overtake_planner.vehicles[name].xcurv[5]
                ey_veh_max, ey_veh_min, s_veh_max, s_veh_min = get_agent_range(
                    s_veh, ey_veh, epsi_veh, veh_len, veh_width
                )
                ey_ego_max, ey_ego_min, s_ego_max, s_ego_min = get_agent_range(
                    s_tmp, xcurv[5], xcurv[3], veh_len, veh_width
                )
                ego_agent_overlap_flag = ego_agent_overlap_checker(
                    s_ego_min, s_ego_max, s_veh_min, s_veh_max, self.track.lap_length
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

    def calc_input(self):
        self.overtake_planner.agent_name = self.agent_name
        self.overtake_planner.opti_traj_xcurv = self.opti_traj_xcurv
        matrix_Atv, matrix_Btv, matrix_Ctv, _ = self.estimate_ABC()
        x = copy.deepcopy(self.x)
        while x[4] > self.lap_length:
            x[4] = x[4] - self.lap_length
        if self.u_pred is None:
            u_old = np.zeros((1, 2))
        else:
            u_old = copy.deepcopy(self.u_pred[0, :])
        overtake_flag, vehicles_interest = self.overtake_planner.get_overtake_flag(x)
        if overtake_flag == False:
            (
                self.u_pred,
                self.x_pred,
                self.ss_point_selected_tot,
                self.Qfun_selected_tot,
                self.lin_points,
                self.lin_input,
            ) = self._lmpc(x, matrix_Atv, matrix_Btv, matrix_Ctv, u_old)
            self.u = self.u_pred[0, :]
            self.old_ey = None
            self.old_direction_flag = None
            iter = self.iter
            self.openloop_prediction.predicted_xcurv[:, :, self.time_in_iter, iter] = self.x_pred
            self.openloop_prediction.predicted_u[:, :, self.time_in_iter, iter] = self.u_pred
            self.openloop_prediction.ss_used[
                :, :, self.time_in_iter, iter
            ] = self.ss_point_selected_tot
            self.openloop_prediction.Qfun_used[:, self.time_in_iter, iter] = self.Qfun_selected_tot
            self.add_point(self.x, self.u, self.time_in_iter)
            self.time_in_iter = self.time_in_iter + 1
            x_pred_xglob = np.zeros((12 + 1, X_DIM))
            for jjj in range(0, 12 + 1):
                xxx, yyy = self.track.get_global_position(self.x_pred[jjj, 4], self.x_pred[jjj, 5])
                psipsi = self.track.get_orientation(self.x_pred[jjj, 4], self.x_pred[jjj, 5])
                x_pred_xglob[jjj, 0:3] = self.x_pred[jjj, 0:3]
                x_pred_xglob[jjj, 3] = psipsi
                x_pred_xglob[jjj, 4] = xxx
                x_pred_xglob[jjj, 5] = yyy
            self.overtake_planner.vehicles["ego"].local_trajs.append(None)
            self.overtake_planner.vehicles["ego"].vehicles_interest.append(None)
            self.overtake_planner.vehicles["ego"].splines.append(None)
            self.overtake_planner.vehicles["ego"].solver_time.append(None)
            self.overtake_planner.vehicles["ego"].all_splines.append(None)
            self.overtake_planner.vehicles["ego"].all_local_trajs.append(None)
            self.overtake_planner.vehicles["ego"].lmpc_prediction.append(x_pred_xglob)
            self.overtake_planner.vehicles["ego"].mpc_cbf_prediction.append(None)
        else:
            if self.path_planner:
                (
                    overtake_traj_xcurv,
                    overtake_traj_xglob,
                    direction_flag,
                    sorted_vehicles,
                    bezier_xglob,
                    solve_time,
                    all_bezier_xglob,
                    all_traj_xglob,
                ) = self.overtake_planner.get_local_path(x, self.time, vehicles_interest)
            else:
                (
                    overtake_traj_xcurv,
                    overtake_traj_xglob,
                    direction_flag,
                    sorted_vehicles,
                    bezier_xglob,
                    solve_time,
                    all_bezier_xglob,
                    all_traj_xglob,
                ) = self.overtake_planner.get_local_traj(
                    x,
                    self.time,
                    vehicles_interest,
                    matrix_Atv,
                    matrix_Btv,
                    matrix_Ctv,
                    self.old_ey,
                    self.old_direction_flag,
                )
            self.old_ey = overtake_traj_xcurv[-1, 5]
            self.old_direction_flag = direction_flag
            self.overtake_planner.vehicles["ego"].local_trajs.append(overtake_traj_xglob)
            self.overtake_planner.vehicles["ego"].vehicles_interest.append(vehicles_interest)
            self.overtake_planner.vehicles["ego"].splines.append(bezier_xglob)
            self.overtake_planner.vehicles["ego"].solver_time.append(solve_time)
            self.overtake_planner.vehicles["ego"].all_splines.append(all_bezier_xglob)
            self.overtake_planner.vehicles["ego"].all_local_trajs.append(all_traj_xglob)
            self.u, x_pred = self._mpc_multi_agents(
                x,
                target_traj_xcurv=overtake_traj_xcurv,
                direction_flag=direction_flag,
                target_traj_xglob=overtake_traj_xglob,
                sorted_vehicles=sorted_vehicles,
            )
            x_pred_xglob = np.zeros((10 + 1, X_DIM))
            for jjj in range(0, 10 + 1):
                xxx, yyy = self.track.get_global_position(x_pred[jjj, 4], x_pred[jjj, 5])
                psipsi = self.track.get_orientation(x_pred[jjj, 4], x_pred[jjj, 5])
                x_pred_xglob[jjj, 0:3] = x_pred[jjj, 0:3]
                x_pred_xglob[jjj, 3] = psipsi
                x_pred_xglob[jjj, 4] = xxx
                x_pred_xglob[jjj, 5] = yyy
            self.overtake_planner.vehicles["ego"].lmpc_prediction.append(None)
            self.overtake_planner.vehicles["ego"].mpc_cbf_prediction.append(x_pred_xglob)
        self.time += self.timestep

    def estimate_ABC(self):
        lin_points = self.lin_points
        lin_input = self.lin_input
        num_horizon = self.lmpc_param.num_horizon
        ss_xcurv = self.ss_xcurv
        u_ss = self.u_ss
        time_ss = self.time_ss
        point_and_tangent = self.point_and_tangent
        timestep = self.timestep
        iter = self.iter
        p = self.p
        Atv = []
        Btv = []
        Ctv = []
        index_used_list = []
        lap_used_for_linearization = 2
        used_iter = range(iter - lap_used_for_linearization, iter)
        max_num_point = 40
        for i in range(0, num_horizon):
            (Ai, Bi, Ci, index_selected,) = _regression_and_linearization(
                lin_points,
                lin_input,
                used_iter,
                ss_xcurv,
                u_ss,
                time_ss,
                max_num_point,
                qp,
                matrix,
                point_and_tangent,
                timestep,
                i,
            )
            Atv.append(Ai)
            Btv.append(Bi)
            Ctv.append(Ci)
            index_used_list.append(index_selected)
        return Atv, Btv, Ctv, index_used_list

    def add_point(self, x, u, i):
        counter = self.time_ss[self.iter - 1]
        self.ss_xcurv[counter + i + 1, :, self.iter - 1] = x + np.array(
            [0, 0, 0, 0, self.lap_length, 0]
        )
        self.u_ss[counter + i + 1, :, self.iter - 1] = u[:]

    def add_trajectory(self, ego, lap_number):

        iter = self.iter
        end_iter = int(round((ego.times[lap_number][-1] - ego.times[lap_number][0]) / ego.timestep))
        times = np.stack(ego.times[lap_number], axis=0)
        self.time_ss[iter] = end_iter
        xcurvs = np.stack(ego.xcurvs[lap_number], axis=0)
        self.ss_xcurv[0 : (end_iter + 1), :, iter] = xcurvs[0 : (end_iter + 1), :]
        xglobs = np.stack(ego.xglobs[lap_number], axis=0)
        self.ss_glob[0 : (end_iter + 1), :, iter] = xglobs[0 : (end_iter + 1), :]
        inputs = np.stack(ego.inputs[lap_number], axis=0)
        self.u_ss[0:end_iter, :, iter] = inputs[0:end_iter, :]
        self.Qfun[0 : (end_iter + 1), iter] = _compute_cost(
            xcurvs[0 : (end_iter + 1), :],
            inputs[0:(end_iter), :],
            self.lap_length,
        )
        for i in np.arange(0, self.Qfun.shape[0]):
            if self.Qfun[i, iter] == 0:
                self.Qfun[i, iter] = self.Qfun[i - 1, iter] - 1
        if self.iter == 0:
            self.lin_points = self.ss_xcurv[1 : self.lmpc_param.num_horizon + 2, :, iter]
            self.lin_input = self.u_ss[1 : self.lmpc_param.num_horizon + 1, :, iter]
        self.iter = self.iter + 1
        self.time_in_iter = 0

