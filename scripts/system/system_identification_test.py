import matplotlib.pyplot as plt
import numpy as np
import sys
from scripts.racing import offboard
from scripts.utils import base, racing_env
from scripts.system import system_identification


def linear_time_invariant():
    # define closed_track
    track_spec = np.array(
        [
            [3, 0],
            [np.pi / 2 * 1.5, -1.5],
            [2, 0],
            [np.pi / 2 * 1.5, -1.5],
            [6, 0],
            [np.pi / 2 * 1.5, -1.5],
            [2.0, 0],
            [np.pi / 2 * 1.5, -1.5],
        ]
    )
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    # setup ego car
    ego = offboard.DynamicBicycleModelOffboard(name="ego", param=base.CarParam(edgecolor="black"))
    ego.set_state_curvilinear(np.array([0.3, 0, 0, 0, 0, 0]))
    ego.set_state_global(np.array([0.3, 0, 0, 0, 0, 0]))
    ego.set_ctrl_policy(offboard.PidTrackingOffboard(vt=0.5))
    ego.ctrl_policy.set_timestep(0.1)
    ego.set_track(track)
    # setup simulation
    simulator = offboard.CarRacingSimOffboard()
    simulator.set_timestep(0.1)
    simulator.set_track(track)
    simulator.add_vehicle(ego)
    ego.ctrl_policy.set_racing_sim(simulator)
    simulator.sim(sim_time=500.0)
    # calculate linearized dynamics
    xdata = np.stack(simulator.vehicles["ego"].xcurv_log, axis=0)
    udata = system_identification.get_udata(simulator.vehicles["ego"])
    lamb = 1e-9
    matrix_A, matrix_B, error = system_identification.linear_regression(
        xdata, udata, lamb)
    np.savetxt("data/track_layout/ellipse.csv", track_spec, delimiter=",")
    np.savetxt("data/sys/LTI/matrix_A.csv", matrix_A, delimiter=",")
    np.savetxt("data/sys/LTI/matrix_B.csv", matrix_B, delimiter=",")
    print(matrix_A)
    print(matrix_B)


if __name__ == "__main__":
    linear_time_invariant()
