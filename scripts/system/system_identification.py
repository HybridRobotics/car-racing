import numpy as np


def linear_regression(x, u, lamb):
    """Estimates linear system dynamics
    x, u: data used in the regression
    lamb: regularization coefficient
    """
    # Want to solve W^* = argmin sum_i ||W^T z_i - y_i ||_2^2 + lamb ||W||_F,
    # with z_i = [x_i u_i] and W \in R^{n + d} x n
    Y = x[2 : x.shape[0], :]
    X = np.hstack((x[1 : (x.shape[0] - 1), :], u[1 : (x.shape[0] - 1), :]))

    Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
    b = np.dot(X.T, Y)
    W = np.dot(Q, b)

    A = W.T[:, 0:6]
    B = W.T[:, 6:8]

    error_matrix = np.dot(X, W) - Y
    error_max = np.max(error_matrix, axis=0)
    error_min = np.min(error_matrix, axis=0)
    error = np.vstack((error_max, error_min))

    return A, B, error

def get_udata(ego):
    laps = ego.laps
    u = np.zeros((round(ego.time / ego.timestep), 2))
    counter = 0
    for i in range(0, laps):
        lap_time_steps = round(
            (ego.times[i][-1] - ego.times[i][0]) / ego.timestep)
        for j in range(0, lap_time_steps):
            u[counter, :] = ego.inputs[i][j][:]
            counter = counter + 1
    lap_time_steps = round(
        (ego.lap_times[-1] - ego.lap_times[0]) / ego.timestep)
    for i in range(0, lap_time_steps):
        u[counter, :] = ego.lap_inputs[i][:]
        counter = counter + 1
    return u
