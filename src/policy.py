import numpy as np
import casadi as ca


class ControlPolicyBase:
    def __init__(self):
        self.xdim = 6
        self.udim = 2
        self.u = np.zeros([self.udim])

    def set_target_speed(self, vt):
        self.vt = vt

    def calc_input(self, x0):
        pass


class PIDSpeedTracking(ControlPolicyBase):
    def __init__(self, vt=0.6):
        ControlPolicyBase.__init__(self)
        self.set_target_speed(vt)

    def calc_input(self, x0):
        """Computes control action
        Arguments:
            x0: current state position
        """
        u_next = np.zeros(self.udim)
        vt = self.vt
        u_next[0] = -0.6 * x0[5] - 0.9 * x0[3] + np.maximum(-0.9, np.minimum(np.random.randn() * 0.25, 0.9))
        u_next[1] = 1.5 * (vt - x0[0]) + np.maximum(-0.8, np.minimum(np.random.randn() * 0.80, 0.8))
        self.u = u_next
        return self.u


class MPCSpeedTracking(ControlPolicyBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6):
        ControlPolicyBase.__init__(self)
        self.set_target_speed(vt)
        self.num_of_horizon = 10
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R

    def calc_input(self, x0):
        xt = np.array([self.vt, 0, 0, 0, 0, 0]).reshape(self.xdim, 1)
        opti = ca.Opti()
        # define variables
        xvar = opti.variable(self.xdim, self.num_of_horizon + 1)
        uvar = opti.variable(self.udim, self.num_of_horizon)
        cost = 0
        opti.subject_to(xvar[:, 0] == x0)
        for i in range(self.num_of_horizon):
            # system dynamics
            opti.subject_to(
                xvar[:, i + 1] == ca.mtimes(self.matrix_A, xvar[:, i]) + ca.mtimes(self.matrix_B, uvar[:, i])
            )
            # speed vx upper bound
            opti.subject_to(xvar[0, i] <= 10.0)
            opti.subject_to(xvar[0, i] >= 0.0)
            # min and max of ey
            opti.subject_to(xvar[5, i] <= 2.0)
            opti.subject_to(-2.0 <= xvar[5, i])
            # min and max of delta
            opti.subject_to(-0.5 <= uvar[0, i])
            opti.subject_to(uvar[0, i] <= 0.5)
            # min and max of a
            opti.subject_to(-1.0 <= uvar[1, i])
            opti.subject_to(uvar[1, i] <= 1.0)
            # quadratic cost
            cost += ca.mtimes((xvar[:, i] - xt).T, ca.mtimes(self.matrix_Q, xvar[:, i] - xt))
            cost += ca.mtimes(uvar[:, i].T, ca.mtimes(self.matrix_R, uvar[:, i]))
        # setup solver
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.minimize(cost)
        opti.solver("ipopt", option)
        sol = opti.solve()
        self.x_pred = sol.value(xvar).T
        self.u_pred = sol.value(uvar).T
        # print("u_pred", self.u_pred)
        self.u = self.u_pred[0, :]
        print(self.u)
        return self.u
