import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
import datetime
from cvxopt.solvers import qp
from utils import racing_env


def compute_cost(xcurv, u, lap_length):
    # The cost has the same elements of the vector x --> time +1
    cost = 10000 * np.ones((xcurv.shape[0]))
    # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
    # We start from the last element of the vector x and we sum the running cost
    for i in range(0, xcurv.shape[0]):
        if (i == 0):  # Note that for i = 0 --> pick the latest element of the vector x
            cost[xcurv.shape[0] - 1 - i] = 0
        elif xcurv[xcurv.shape[0] - 1 - i, 4] < lap_length:
            cost[xcurv.shape[0] - 1 - i] = cost[xcurv.shape[0] - 1 - i + 1] + 1
        else:
            cost[xcurv.shape[0] - 1 - i] = 0
    return cost


def regression_and_linearization(lin_points, lin_input, used_iter, ss_xcurv, u_ss, time_ss, max_num_point, qp, n, d, matrix, point_and_tangent, dt, i):
    x0 = lin_points[i, :]
    Ai = np.zeros((n, n))
    Bi = np.zeros((n, d))
    Ci = np.zeros((n, 1))
    # Compute Index to use
    h = 5
    lamb = 0.0
    state_features = [0, 1, 2]
    consider_input = True
    if consider_input == True:
        scaling = np.array([[0.1, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]])
        x_lin = np.hstack((lin_points[i, state_features], lin_input[i, :]))
    else:
        scaling = np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        x_lin = lin_points[i, state_features]
    index_selected = []
    K = []
    for i in used_iter:
        index_selected_i, K_i = compute_index(h, ss_xcurv, u_ss, time_ss, i, x_lin, state_features, scaling, max_num_point,
                                              consider_input)
        index_selected.append(index_selected_i)
        K.append(K_i)
    # =========================
    # ====== Identify vx ======
    input_features = [1]
    Q_vx, M_vx = compute_Q_M(ss_xcurv, u_ss, index_selected,
                             state_features, input_features, used_iter, np, matrix, lamb, K)
    y_index = 0
    b = compute_b(ss_xcurv, y_index, used_iter,
                  matrix, M_vx, index_selected, K)
    Ai[y_index, state_features], Bi[y_index, input_features], Ci[y_index] = lmpc_loc_lin_reg(Q_vx, b, state_features,
                                                                                             input_features, qp)
    # =======================================
    # ====== Identify Lateral Dynamics ======
    input_features = [0]
    Q_lat, M_lat = compute_Q_M(ss_xcurv, u_ss, index_selected,
                               state_features, input_features, used_iter, np, matrix, lamb, K)
    y_index = 1  # vy
    b = compute_b(ss_xcurv, y_index, used_iter,
                  matrix, M_lat, index_selected, K)
    Ai[y_index, state_features], Bi[y_index, input_features], Ci[y_index] = lmpc_loc_lin_reg(Q_lat, b, state_features,
                                                                                             input_features, qp)
    y_index = 2  # wz
    b = compute_b(ss_xcurv, y_index, used_iter,
                  matrix, M_lat, index_selected, K)
    Ai[y_index, state_features], Bi[y_index, input_features], Ci[y_index] = lmpc_loc_lin_reg(Q_lat, b, state_features,
                                                                                             input_features, qp)
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
    cur = racing_env.get_curvature(
        point_and_tangent[-1, 3]+point_and_tangent[-1, 4], point_and_tangent, s)
    den = 1 - cur * ey
    # ===========================
    # ===== Linearize epsi ======
    # epsi_{k+1} = epsi + dt * ( wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur )
    depsi_vx = -dt * np.cos(epsi) / den * cur
    depsi_vy = dt * np.sin(epsi) / den * cur
    depsi_wz = dt
    depsi_epsi = 1 - dt * (-vx * np.sin(epsi) - vy * np.cos(epsi)) / den * cur
    depsi_s = 0  # Because cur = constant
    depsi_ey = dt * (vx * np.cos(epsi) - vy * np.sin(epsi)) / \
        (den ** 2) * cur * (-cur)
    Ai[3, :] = [depsi_vx, depsi_vy, depsi_wz, depsi_epsi, depsi_s, depsi_ey]
    Ci[3] = epsi + dt * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)
                               ) / (1 - cur * ey) * cur) - np.dot(Ai[3, :], x0)
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
    Ci[4] = s + dt * ((vx * np.cos(epsi) - vy * np.sin(epsi)) /
                      (1 - cur * ey)) - np.dot(Ai[4, :], x0)
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
    Ci[5] = ey + dt * (vx * np.sin(epsi) + vy *
                       np.cos(epsi)) - np.dot(Ai[5, :], x0)
    deltaTimer_tv = datetime.datetime.now() - startTimer
    return Ai, Bi, Ci, index_selected


def compute_index(h, ss_xcurv, u_ss, time_ss, iter, x0, state_features, scaling, max_num_point, consider_input):
    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    # What to learn a model such that: x_{k+1} = A x_k  + B u_k + C
    one_vec = np.ones((ss_xcurv[0:time_ss[iter], :, iter].shape[0]-1, 1))
    x0_vec = (np.dot(np.array([x0]).T, one_vec.T)).T
    if consider_input == True:
        DataMatrix = np.hstack(
            (ss_xcurv[0:time_ss[iter]-1, state_features, iter], u_ss[0:time_ss[iter]-1, :, iter]))
    else:
        DataMatrix = ss_xcurv[0:time_ss[iter]-1, state_features, iter]
    diff = np.dot((DataMatrix - x0_vec), scaling)
    norm = la.norm(diff, 1, axis=1)
    index_tot = np.squeeze(np.where(norm < h))
    if (index_tot.shape[0] >= max_num_point):
        index = np.argsort(norm)[0:max_num_point]
    else:
        index = index_tot

    K = (1 - (norm[index] / h)**2) * 3/4
    return index, K


def compute_Q_M(ss_xcurv, u_ss, index_selected, state_features, input_features, used_iter, np, matrix, lamb, K):
    counter = 0
    iter = 1
    X0 = np.empty((0, len(state_features)+len(input_features)))
    Ktot = np.empty((0))
    for iter in used_iter:
        X0 = np.append(X0, np.hstack((np.squeeze(ss_xcurv[np.ix_(index_selected[counter], state_features, [iter])]),
                                      np.squeeze(u_ss[np.ix_(index_selected[counter], input_features, [iter])], axis=2))), axis=0)
        Ktot = np.append(Ktot, K[counter])
        counter = counter + 1

    M = np.hstack((X0, np.ones((X0.shape[0], 1))))
    Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
    Q = matrix(Q0 + lamb * np.eye(Q0.shape[0]))
    return Q, M


def select_points(ss_xcurv, Qfun, iter, x0, num_ss_points, shift):
    # selects the closest point in the safe set to x0
    # returns a subset of the safe set which contains a range of points ahead of this point
    xcurv = ss_xcurv[:, :, iter]
    one_vec = np.ones((xcurv.shape[0], 1))
    x0_vec = (np.dot(np.array([x0]).T, one_vec.T)).T
    diff = xcurv - x0_vec
    norm = la.norm(diff, 1, axis=1)
    min_norm = np.argmin(norm)
    if (min_norm + shift >= 0):
        ss_points = xcurv[int(shift + min_norm):int(shift + min_norm + num_ss_points), :].T
        sel_Qfun = Qfun[int(shift + min_norm):int(shift +
                                                  min_norm + num_ss_points), iter]
    else:
        ss_points = xcurv[int(min_norm):int(min_norm + num_ss_Points), :].T
        sel_Qfun = Qfun[int(min_norm):int(min_norm + num_ss_Points), iter]
    return ss_points, sel_Qfun


class closedloop_data():
    """Object collecting closed loop data points
    Attributes:
        update_initial_conditions: function which updates initial conditions and clear the memory
    """

    def __init__(self, timestep, sim_time, v0):
        """Initialization
        Arguments:
            timestep: discretization time
            sim_time: maximum time [s] which can be recorded
            v0: velocity initial condition
        """
        self.timestep = timestep
        # Number of points in the simulation
        self.points = int(sim_time / timestep)
        self.u = np.zeros((self.points, 2))  # Initialize the input vector
        # Initialize state vector (In curvilinear abscissas)
        self.xcurv = np.zeros((self.points + 1, 6))
        # Initialize the state vector in absolute reference frame
        self.xglob = np.zeros((self.points + 1, 6))
        self.sim_points = 0.0
        self.xcurv[0, 0] = v0
        self.xglob[0, 0] = v0

    def update_initial_conditions(self, xcurv, xglob):
        """Clears memory and resets initial condition
        xcurv: initial condition is the curvilinear reference frame
        xglob: initial condition in the inertial reference frame
        """
        self.xcurv[0, :] = xcurv
        self.xglob[0, :] = xglob
        self.xcurv[1:, :] = 0*self.xcurv[1:, :]
        self.xglob[1:, :] = 0*self.xglob[1:, :]


class lmpc_prediction():
    """Object collecting the predictions and SS at eath time step
    """

    def __init__(self, num_horizon=12, xdim=6, udim=2, points_lmpc=5000, num_ss_points=32 + 12, lap_number = None):
        """
        Initialization:
            num_horizon: horizon length
            xdim, udim: state and input dimensions
            points_lmpc: maximum simulation timesteps
            num_ss_points: number used to build safe set at each time step
        """
        self.predicted_xcurv = np.zeros((num_horizon+1, xdim, points_lmpc, lap_number))
        self.predicted_u = np.zeros((num_horizon, udim, points_lmpc, lap_number))
        self.ss_used = np.zeros(
            (xdim, num_ss_points, points_lmpc, lap_number))
        self.Qfun_used = np.zeros((num_ss_points, points_lmpc, lap_number))


def compute_b(ss_xcurv, y_index, used_iter, matrix, M, index_selected, K):
    counter = 0
    y = np.empty((0))
    Ktot = np.empty((0))
    for iter in used_iter:
        y = np.append(y, np.squeeze(
            ss_xcurv[np.ix_(index_selected[counter] + 1, [y_index], [iter])]))
        Ktot = np.append(Ktot, K[counter])
        counter = counter + 1
    b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))
    return b


def lmpc_loc_lin_reg(Q, b, state_features, input_features, qp):
    startTimer = datetime.datetime.now()  # Start timer for LMPC iteration
    res_cons = qp(Q, b)  # This is ordered as [A B C]
    deltaTimer_tv = datetime.datetime.now() - startTimer
    result = np.squeeze(np.array(res_cons['x']))
    A = result[0:len(state_features)]
    B = result[len(state_features):(len(state_features)+len(input_features))]
    C = result[-1]
    return A, B, C
