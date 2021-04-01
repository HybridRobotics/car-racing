import numpy as np


def vehicle_dynamics(dynamics_param, curv, xglob, xcurv, delta_t, u):

    m, lf, lr, Iz, Df, Cf, Bf, Dr, Cr, Br = dynamics_param.get_params() 

    xglob_next = np.zeros(len(xglob))
    xcurv_next = np.zeros(len(xcurv))
    delta = u[0]
    a = u[1]

    psi = xglob[3]
    X = xglob[4]
    Y = xglob[5]

    vx = xcurv[0]
    vy = xcurv[1]
    wz = xcurv[2]
    epsi = xcurv[3]
    s = xcurv[4]
    ey = xcurv[5]

    alpha_f = delta - np.arctan2(vy + lf * wz, vx)
    alpha_r = -np.arctan2(vy - lf * wz, vx)

    # Compute lateral force at front and rear tire
    Fyf = 2 * Df * np.sin(Cf * np.arctan(Bf * alpha_f))
    Fyr = 2 * Dr * np.sin(Cr * np.arctan(Br * alpha_r))

    # Propagate the dynamics of delta_t
    xglob_next[0] = vx + delta_t * (a - 1 / m * Fyf * np.sin(delta) + wz * vy)
    xglob_next[1] = vy + delta_t * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
    xglob_next[2] = wz + delta_t * (1 / Iz * (lf * Fyf * np.cos(delta) - lr * Fyr))
    xglob_next[3] = psi + delta_t * (wz)
    xglob_next[4] = X + delta_t * ((vx * np.cos(psi) - vy * np.sin(psi)))
    xglob_next[5] = Y + delta_t * (vx * np.sin(psi) + vy * np.cos(psi))

    xcurv_next[0] = vx + delta_t * (a - 1 / m * Fyf * np.sin(delta) + wz * vy)
    xcurv_next[1] = vy + delta_t * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
    xcurv_next[2] = wz + delta_t * (1 / Iz * (lf * Fyf * np.cos(delta) - lr * Fyr))
    xcurv_next[3] = epsi + delta_t * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curv * ey) * curv)
    xcurv_next[4] = s + delta_t * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curv * ey))
    xcurv_next[5] = ey + delta_t * (vx * np.sin(epsi) + vy * np.cos(epsi))

    return xglob_next, xcurv_next 


