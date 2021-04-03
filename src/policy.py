import datetime
import numpy as np
import casadi as ca
import solver

class ControlPolicyBase:
    def __init__(self):
        self.agent_name = None
        self.xdim = 6
        self.udim = 2
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.u = None

    def set_racing_sim(self, racing_sim):
        self.racing_sim = racing_sim

    def set_timestep(self, timestep):
        self.timestep = timestep

    def set_target_speed(self, vt):
        self.vt = vt

    def set_target_deviation(self, eyt):
        self.eyt = eyt

    def set_state(self, x0):
        self.x = x0

    def calc_input(self):
        pass

    def get_input(self):
        return self.u


class PIDTracking(ControlPolicyBase):
    def __init__(self, vt=0.6, eyt=0.0):
        ControlPolicyBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(self.xdim, 1)
        self.u = solver.pid(self.x, xtarget, self.udim)
        self.time += self.timestep


class MPCTracking(ControlPolicyBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        ControlPolicyBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)
        self.num_of_horizon = 10
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(self.xdim, 1)
        self.u = solver.mpc(self.x, xtarget, self.udim, self.num_of_horizon, self.matrix_A, self.matrix_B, self.matrix_Q, self.matrix_R)
        self.time += self.timestep


class MPCCBFRacing(ControlPolicyBase):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        ControlPolicyBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)
        self.num_of_horizon = 10
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.alpha = 0.6

    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(self.xdim, 1)
        self.u = solver.mpccbf(self.x, xtarget, self.udim, self.num_of_horizon, self.matrix_A, self.matrix_B, self.matrix_Q, self.matrix_R, self.racing_sim.vehicles, self.agent_name, self.racing_sim.track.lap_length, self.time, self.timestep, self.alpha)
        self.time += self.timestep