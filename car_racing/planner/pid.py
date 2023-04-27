import datetime

import numpy as np

from planner.base import PlannerBase
from racing_env import U_DIM, X_DIM


class PIDTracking(PlannerBase):
    """ PID as the planner
    """
    def __init__(self, vt=0.6, eyt=0.0):
        PlannerBase.__init__(self)
        self.set_target_speed(vt)
        self.set_target_deviation(eyt)

    def _pid(self, xtarget):
        """The core to PID algorithm
        
        NOTE: the Kp, Ki, and Ki are hard coded
        """
        start_timer = datetime.datetime.now()
        u_next = np.zeros((U_DIM,))
        vt = xtarget[0]
        eyt = xtarget[5]
        u_next[0] = -0.6 * (self.x[5] - eyt) - 0.9 * self.x[3]
        u_next[1] = 1.5 * (vt - self.x[0])
        end_timer = datetime.datetime.now()
        solver_time = (end_timer - start_timer).total_seconds()
        print("solver time: {}".format(solver_time))
        return u_next
    
    def calc_input(self):
        xtarget = np.array([self.vt, 0, 0, 0, 0, self.eyt]).reshape(X_DIM, 1)
        self.u = self._pid(xtarget)
        if self.agent_name == "ego":
            if self.realtime_flag == False:
                vehicles = self.racing_env.vehicles
            else:
                vehicles = self.vehicles
            vehicles["ego"].local_trajs.append(None)
            vehicles["ego"].vehicles_interest.append(None)
            vehicles["ego"].splines.append(None)
            vehicles["ego"].all_splines.append(None)
            vehicles["ego"].all_local_trajs.append(None)
            vehicles["ego"].lmpc_prediction.append(None)
            vehicles["ego"].mpc_cbf_prediction.append(None)
        self.time += self.timestep