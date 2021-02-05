import numpy as np
import sympy as sp


class CarParam:
    def __init__(self, length=0.4, width=0.2, facecolor="None", edgecolor="black"):
        self.length = length
        self.width = width
        self.facecolor = facecolor
        self.edgecolor = edgecolor


class BaseModel:
    def __init__(self, name=None, param=None):
        self.name = name
        self.param = param
        self.no_dynamics = False
        self.xdim = 6
        self.udim = 2
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.xglob = None
        self.u = None
        self.closedloop_time = []
        self.closedloop_x = []
        self.closedloop_xglob = []
        self.closedloop_u = []

    def set_timestep(self, dt):
        self.timestep = dt

    def set_state_curvilinear(self, x):
        self.x = x

    def set_state_global(self, xglob):
        self.xglob = xglob

    def set_track(self, track):
        self.track = track

    def set_ctrl_policy(self, ctrl_policy):
        self.ctrl_policy = ctrl_policy
        self.ctrl_policy.agent_name = self.name

    def calc_ctrl_input(self):
        self.ctrl_policy.set_state(self.x)
        self.ctrl_policy.calc_input()
        self.u = self.ctrl_policy.get_input()

    def forward_dynamics(self):
        pass

    def forward_one_step(self):
        if self.no_dynamics:
            self.forward_dynamics()
            self.update_memory()
        else:
            self.calc_ctrl_input()
            self.forward_dynamics()
            self.update_memory()

    def update_memory(self):
        self.closedloop_time.append(self.time)
        self.closedloop_x.append(self.x)
        self.closedloop_xglob.append(self.xglob)
        self.closedloop_u.append(self.u)


class NoPolicyModel(BaseModel):
    def __init__(self, name=None, param=None, x=None, xglob=None):
        BaseModel.__init__(self, name=name, param=param)
        self.no_dynamics = True

    def set_state_curvilinear_func(self, t_symbol, s_func, ey_func):
        self.t_symbol = t_symbol
        self.s_func = s_func
        self.ey_func = ey_func

    def get_estimation(self, t0):
        # position estimation in curvilinear coordinates
        x_cur_est = np.zeros(self.xdim)
        x_cur_est[0] = sp.diff(self.s_func, self.t_symbol).subs(self.t_symbol, t0)
        x_cur_est[1] = sp.diff(self.ey_func, self.t_symbol).subs(self.t_symbol, t0)
        x_cur_est[2] = 0
        x_cur_est[3] = 0
        x_cur_est[4] = self.s_func.subs(self.t_symbol, t0)
        x_cur_est[5] = self.ey_func.subs(self.t_symbol, t0)
        # position estimation in global coordinates
        X, Y = self.track.get_global_position(x_cur_est[4], x_cur_est[5])
        psi = self.track.get_orientation(x_cur_est[4], x_cur_est[5])
        xglob_est = np.zeros(self.xdim)
        xglob_est[0:3] = x_cur_est[0:3]
        xglob_est[3] = psi
        xglob_est[4] = X
        xglob_est[5] = Y
        return x_cur_est, xglob_est

    def get_trajectory_nsteps(self, t0, delta_t, n):
        x_cur_est_nsteps = np.zeros((self.xdim, n))
        xglob_est_nsteps = np.zeros((self.xdim, n))
        for index in range(n):
            x_cur_est, xglob_est = self.get_estimation(self.time + index * delta_t)
            x_cur_est_nsteps[:, index] = x_cur_est
            xglob_est_nsteps[:, index] = xglob_est
        return x_cur_est_nsteps, xglob_est_nsteps

    def forward_dynamics(self):
        self.time += self.timestep
        self.x, self.xglob = self.get_estimation(self.time)


class DynamicBicycleModel(BaseModel):
    def __init__(self, name=None, param=None, x=None, xglob=None):
        BaseModel.__init__(self, name=name, param=param)

    def forward_dynamics(self):
        # This function computes the system evolution. Note that the discretization is delta_t and therefore is needed that
        # dt <= delta_t and ( dt / delta_t) = integer value

        # Vehicle Parameters
        m = 1.98
        lf = 0.125
        lr = 0.125
        Iz = 0.024
        Df = 0.8 * m * 9.81 / 2.0
        Cf = 1.25
        Bf = 1.0
        Dr = 0.8 * m * 9.81 / 2.0
        Cr = 1.25
        Br = 1.0

        # Discretization Parameters
        delta_t = 0.001
        x_next = np.zeros(self.xdim)
        cur_x_next = np.zeros(self.xdim)

        # Extract the value of the states
        delta = self.u[0]
        a = self.u[1]

        psi = self.xglob[3]
        X = self.xglob[4]
        Y = self.xglob[5]

        vx = self.x[0]
        vy = self.x[1]
        wz = self.x[2]
        epsi = self.x[3]
        s = self.x[4]
        ey = self.x[5]

        # Initialize counter
        i = 0
        while (i + 1) * delta_t <= self.timestep:
            # Compute tire split angle
            alpha_f = delta - np.arctan2(vy + lf * wz, vx)
            alpha_r = -np.arctan2(vy - lf * wz, vx)

            # Compute lateral force at front and rear tire
            Fyf = 2 * Df * np.sin(Cf * np.arctan(Bf * alpha_f))
            Fyr = 2 * Dr * np.sin(Cr * np.arctan(Br * alpha_r))

            # Propagate the dynamics of delta_t
            x_next[0] = vx + delta_t * (a - 1 / m * Fyf * np.sin(delta) + wz * vy)
            x_next[1] = vy + delta_t * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
            x_next[2] = wz + delta_t * (1 / Iz * (lf * Fyf * np.cos(delta) - lr * Fyr))
            x_next[3] = psi + delta_t * (wz)
            x_next[4] = X + delta_t * ((vx * np.cos(psi) - vy * np.sin(psi)))
            x_next[5] = Y + delta_t * (vx * np.sin(psi) + vy * np.cos(psi))

            cur = self.track.get_curvature(s)
            cur_x_next[0] = vx + delta_t * (a - 1 / m * Fyf * np.sin(delta) + wz * vy)
            cur_x_next[1] = vy + delta_t * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
            cur_x_next[2] = wz + delta_t * (1 / Iz * (lf * Fyf * np.cos(delta) - lr * Fyr))
            cur_x_next[3] = epsi + delta_t * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur)
            cur_x_next[4] = s + delta_t * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey))
            cur_x_next[5] = ey + delta_t * (vx * np.sin(epsi) + vy * np.cos(epsi))

            # Update the value of the states
            psi = x_next[3]
            X = x_next[4]
            Y = x_next[5]

            vx = cur_x_next[0]
            vy = cur_x_next[1]
            wz = cur_x_next[2]
            epsi = cur_x_next[3]
            s = cur_x_next[4]
            ey = cur_x_next[5]
            if s < 0:
                pass
                # Don't need this checker as curvature can be calculated even s < 0
                # print("Start Point: ", self.x, " Input: ", self.ctrl_policy.u)
                # print("x_next: ", x_next)
            # Increment counter
            i = i + 1
        # Noises
        noise_vx = np.maximum(-0.05, np.minimum(np.random.randn() * 0.01, 0.05))
        noise_vy = np.maximum(-0.1, np.minimum(np.random.randn() * 0.01, 0.1))
        noise_wz = np.maximum(-0.05, np.minimum(np.random.randn() * 0.005, 0.05))

        cur_x_next[0] = cur_x_next[0] + 0.1 * noise_vx
        cur_x_next[1] = cur_x_next[1] + 0.1 * noise_vy
        cur_x_next[2] = cur_x_next[2] + 0.1 * noise_wz

        self.x = cur_x_next
        self.xglob = x_next
        self.time += self.timestep
