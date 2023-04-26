
import copy

import numpy as np
import sympy as sp

from racing_env.params import *

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

    # Compute tire slip angle
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
    xcurv_next[3] = epsi + delta_t * (
        wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curv * ey) * curv
    )
    xcurv_next[4] = s + delta_t * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - curv * ey))
    xcurv_next[5] = ey + delta_t * (vx * np.sin(epsi) + vy * np.cos(epsi))

    return xglob_next, xcurv_next

class ModelBase:
    def __init__(self, name=None, param=None, no_dynamics=False, system_param=None):
        self.name = name
        self.param = param
        self.system_param = system_param
        self.no_dynamics = False
        self.time = 0.0
        self.timestep = None
        self.xcurv = None
        self.xglob = None
        self.u = None
        self.zero_noise_flag = False
        self.lap_times = []
        if self.no_dynamics:
            pass
        else:
            self.lap_times.append(self.time)
        self.lap_xcurvs, self.lap_xglobs, self.lap_inputs = [], [], []
        self.times, self.xglobs, self.xcurvs, self.inputs = [], [], [], []
        self.laps = 0
        self.realtime_flag = False
        self.xglob_log = []
        self.xcurv_log = []
        self.local_trajs = []
        self.vehicles_interest = []
        self.splines = []
        self.solver_time = []
        self.all_splines = []
        self.all_local_trajs = []
        self.lmpc_prediction = []
        self.mpc_cbf_prediction = []

    def set_zero_noise(self):
        self.zero_noise_flag = True

    def set_timestep(self, dt):
        self.timestep = dt

    def set_state_curvilinear(self, xcurv):
        self.xcurv = xcurv

    def set_state_global(self, xglob):
        self.xglob = xglob

    def start_logging(self):
        self.lap_xcurvs, self.lap_xglobs, self.lap_inputs = [], [], []
        self.lap_xcurvs.append(self.xcurv)
        self.lap_xglobs.append(self.xglob)

    def set_track(self, track):
        self.track = track
        self.lap_length = track.lap_length
        self.point_and_tangent = track.point_and_tangent
        self.lap_width = track.width

    def set_ctrl_policy(self, ctrl_policy):
        self.ctrl_policy = ctrl_policy
        self.ctrl_policy.agent_name = self.name

    def calc_ctrl_input(self):
        self.ctrl_policy.set_state(self.xcurv, self.xglob)
        self.ctrl_policy.calc_input()
        self.u = self.ctrl_policy.get_input()

    def forward_dynamics(self):
        pass

    def forward_one_step(self, realtime_flag):
        if self.no_dynamics:
            self.forward_dynamics()
            self.update_memory()
        elif realtime_flag == False:
            self.calc_ctrl_input()
            self.forward_dynamics(realtime_flag)
            self.ctrl_policy.set_state(self.xcurv, self.xglob)
            self.update_memory()
        elif realtime_flag == True:
            self.forward_dynamics(realtime_flag)

    def update_memory(self):
        xcurv = copy.deepcopy(self.xcurv)
        xglob = copy.deepcopy(self.xglob)
        self.xglob_log.append(self.xglob)
        self.xcurv_log.append(self.xcurv)
        if xcurv[4] > self.lap_length:
            self.lap_xglobs.append(self.xglob)
            self.lap_times.append(self.time)
            self.lap_xcurvs.append(xcurv)
            self.lap_inputs.append(self.u)
            self.xglobs.append(self.lap_xglobs)
            self.times.append(self.lap_times)
            self.xcurvs.append(self.lap_xcurvs)
            self.inputs.append(self.lap_inputs)
            self.xcurv[4] = self.xcurv[4] - self.lap_length
            self.laps = self.laps + 1
            self.lap_xglobs, self.lap_xcurvs, self.lap_inputs, self.lap_times = [], [], [], []
            self.lap_xglobs.append(self.xglob)
            self.lap_times.append(self.time)
            self.lap_xcurvs.append(self.xcurv)
        else:
            self.lap_xglobs.append(self.xglob)
            self.lap_times.append(self.time)
            self.lap_xcurvs.append(self.xcurv)
            self.lap_inputs.append(self.u)

    def get_vehicle_in_rectangle(self, vehicle_xglob):
        car_length = self.param.length
        car_width = self.param.width
        car_dx = 0.5 * car_length
        car_dy = 0.5 * car_width
        car_xs_origin = [car_dx, car_dx, -car_dx, -car_dx, car_dx]
        car_ys_origin = [car_dy, -car_dy, -car_dy, car_dy, car_dy]
        car_frame = np.vstack((np.array(car_xs_origin), np.array(car_ys_origin)))
        x = vehicle_xglob[4]
        y = vehicle_xglob[5]
        R = np.matrix(
            [
                [np.cos(vehicle_xglob[3]), -np.sin(vehicle_xglob[3])],
                [np.sin(vehicle_xglob[3]), np.cos(vehicle_xglob[3])],
            ]
        )
        rotated_car_frame = R * car_frame
        return (
            x + rotated_car_frame[0, 2],
            y + rotated_car_frame[1, 2],
            2 * car_dx,
            2 * car_dy,
            vehicle_xglob[3] * 180 / 3.14,
        )


class NoDynamicsModel(ModelBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        ModelBase.__init__(self, name=name, param=param)
        self.no_dynamics = True

    def set_state_curvilinear_func(self, t_symbol, s_func, ey_func):
        self.t_symbol = t_symbol
        self.s_func = s_func
        self.ey_func = ey_func
        self.xcurv = np.zeros((X_DIM,))
        self.xglob = np.zeros((X_DIM,))
        self.xcurv, self.xglob = self.get_estimation(0)

    def get_estimation(self, t0):
        # position estimation in curvilinear coordinates
        xcurv_est = np.zeros((X_DIM,))
        xcurv_est[0] = sp.diff(self.s_func, self.t_symbol).subs(self.t_symbol, t0)
        xcurv_est[1] = sp.diff(self.ey_func, self.t_symbol).subs(self.t_symbol, t0)
        xcurv_est[2] = 0
        xcurv_est[3] = 0
        xcurv_est[4] = self.s_func.subs(self.t_symbol, t0)
        xcurv_est[5] = self.ey_func.subs(self.t_symbol, t0)
        # position estimation in global coordinates
        X, Y = self.track.get_global_position(xcurv_est[4], xcurv_est[5])
        psi = self.track.get_orientation(xcurv_est[4], xcurv_est[5])
        xglob_est = np.zeros((X_DIM,))
        xglob_est[0:3] = xcurv_est[0:3]
        xglob_est[3] = psi
        xglob_est[4] = X
        xglob_est[5] = Y
        return xcurv_est, xglob_est

    def get_trajectory_nsteps(self, t0, delta_t, n):
        xcurv_est_nsteps = np.zeros((X_DIM, n))
        xglob_est_nsteps = np.zeros((X_DIM, n))
        for index in range(n):
            xcurv_est, xglob_est = self.get_estimation(self.time + index * delta_t)
            xcurv_est_nsteps[:, index] = xcurv_est
            xglob_est_nsteps[:, index] = xglob_est
        return xcurv_est_nsteps, xglob_est_nsteps

    def forward_dynamics(self):
        self.time += self.timestep
        self.xcurv, self.xglob = self.get_estimation(self.time)


class DynamicBicycleModel(ModelBase):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None, system_param=None):
        ModelBase.__init__(self, name=name, param=param, system_param=system_param)

    def forward_dynamics(self, realtime_flag):
        # This function computes the system evolution. Note that the discretization is delta_t and therefore is needed that
        # dt <= delta_t and ( dt / delta_t) = integer value
        # Discretization Parameters
        delta_t = 0.001
        xglob_next = np.zeros((X_DIM,))
        xcurv_next = np.zeros((X_DIM,))
        xglob_next = self.xglob
        xcurv_next = self.xcurv
        vehicle_param = CarParam()
        # Initialize counter
        i = 0
        while (i + 1) * delta_t <= self.timestep:
            s = xcurv_next[4]
            curv = self.track.get_curvature(s)
            # for realtime simulation, if no controller is set up, the model won't update
            if self.u is None:
                pass
            else:
                xglob_next, xcurv_next = vehicle_dynamics(
                    vehicle_param.dynamics_param,
                    curv,
                    xglob_next,
                    xcurv_next,
                    delta_t,
                    self.u,
                )
            # Increment counter
            i = i + 1
        # Noises
        noise_vx = np.maximum(-0.05, np.minimum(np.random.randn() * 0.01, 0.05))
        noise_vy = np.maximum(-0.1, np.minimum(np.random.randn() * 0.01, 0.1))
        noise_wz = np.maximum(-0.05, np.minimum(np.random.randn() * 0.005, 0.05))
        if ((realtime_flag == True) and (self.u is None)) or self.zero_noise_flag:
            # for realtime simulation, if no controller is set up, no update for state
            pass
        else:
            xcurv_next[0] = xcurv_next[0] + 0.5 * noise_vx
            xcurv_next[1] = xcurv_next[1] + 0.5 * noise_vy
            xcurv_next[2] = xcurv_next[2] + 0.5 * noise_wz
        self.xcurv = xcurv_next
        self.xglob = xglob_next
        self.time += self.timestep


# Base Simulator
class CarRacingSim:
    def __init__(self):
        self.track = None
        self.vehicles = {}
        self.opti_traj_xglob = None

    def set_timestep(self, dt):
        self.timestep = dt

    def set_track(self, track):
        self.track = track

    def set_opti_traj(self, opti_traj_xglob):
        self.opti_traj_xglob = opti_traj_xglob