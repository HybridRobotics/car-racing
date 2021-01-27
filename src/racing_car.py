import numpy as np


class CarParam:
    def __init__(self, length=0.4, width=0.2, color="b"):
        self.length = length
        self.width = width
        self.color = color


class BaseModel:
    def __init__(self):
        self.xdim = 6
        self.udim = 2
        self.x = None
        self.x_glob = None
        self.u = None

    def set_timestep(self, dt):
        self.dt = dt

    def set_track(self, track):
        self.track = track

    def set_ctrl_policy(self, ctrl_policy):
        self.ctrl_policy = ctrl_policy

    def calc_ctrl_input(self):
        self.u = self.ctrl_policy.calc_input(self.x)

    def forward_dynamics(self):
        pass


class DynamicBicycleModel(BaseModel):
    def __init__(self, name=None, car_param=None, x=None, x_glob=None):
        BaseModel.__init__(self)
        self.name = name
        self.car_param = car_param
        self.x = x
        self.x_glob = x_glob

    def forward_dynamics(self):
        # This function computes the system evolution. Note that the discretization is deltaT and therefore is needed that
        # dt <= deltaT and ( dt / deltaT) = integer value

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
        deltaT = 0.001
        x_next = np.zeros(self.xdim)
        cur_x_next = np.zeros(self.xdim)

        # Extract the value of the states
        delta = self.u[0]
        a = self.u[1]

        psi = self.x_glob[3]
        X = self.x_glob[4]
        Y = self.x_glob[5]

        vx = self.x[0]
        vy = self.x[1]
        wz = self.x[2]
        epsi = self.x[3]
        s = self.x[4]
        ey = self.x[5]

        # Initialize counter
        i = 0
        while (i + 1) * deltaT <= self.dt:
            # Compute tire split angle
            alpha_f = delta - np.arctan2(vy + lf * wz, vx)
            alpha_r = -np.arctan2(vy - lf * wz, vx)

            # Compute lateral force at front and rear tire
            Fyf = 2 * Df * np.sin(Cf * np.arctan(Bf * alpha_f))
            Fyr = 2 * Dr * np.sin(Cr * np.arctan(Br * alpha_r))

            # Propagate the dynamics of deltaT
            x_next[0] = vx + deltaT * (a - 1 / m * Fyf * np.sin(delta) + wz * vy)
            x_next[1] = vy + deltaT * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
            x_next[2] = wz + deltaT * (1 / Iz * (lf * Fyf * np.cos(delta) - lr * Fyr))
            x_next[3] = psi + deltaT * (wz)
            x_next[4] = X + deltaT * ((vx * np.cos(psi) - vy * np.sin(psi)))
            x_next[5] = Y + deltaT * (vx * np.sin(psi) + vy * np.cos(psi))

            cur = self.track.get_curvature(s)
            cur_x_next[0] = vx + deltaT * (a - 1 / m * Fyf * np.sin(delta) + wz * vy)
            cur_x_next[1] = vy + deltaT * (1 / m * (Fyf * np.cos(delta) + Fyr) - wz * vx)
            cur_x_next[2] = wz + deltaT * (1 / Iz * (lf * Fyf * np.cos(delta) - lr * Fyr))
            cur_x_next[3] = epsi + deltaT * (wz - (vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey) * cur)
            cur_x_next[4] = s + deltaT * ((vx * np.cos(epsi) - vy * np.sin(epsi)) / (1 - cur * ey))
            cur_x_next[5] = ey + deltaT * (vx * np.sin(epsi) + vy * np.cos(epsi))

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
        self.x_glob = x_next
