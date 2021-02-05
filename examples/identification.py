import numpy as np
import matplotlib.pyplot as plt
from pycar_racing import racing_env, racing_car, racing_sim, policy, utils


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
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)
    # setup ego car
    ego = racing_car.DynamicBicycleModel(name="ego", param=racing_car.CarParam(edgecolor="black"))
    ego.set_state_curvilinear(np.array([0.3, 0, 0, 0, 0, 0]))
    ego.set_state_global(np.array([0.3, 0, 0, 0, 0, 0]))
    ego.set_ctrl_policy(policy.PIDTracking(vt=0.5))
    ego.ctrl_policy.set_timestep(0.1)
    # setup simulation
    simulator = racing_sim.CarRacingSim()
    simulator.set_timestep(0.1)
    simulator.set_track(track)
    simulator.add_vehicle(ego)
    simulator.sim(sim_time=500.0)
    # calculate linearized dynamics
    xdata = np.stack(simulator.closedloop_x["ego"], axis=0)
    udata = np.stack(simulator.closedloop_u["ego"], axis=0)
    lamb = 1e-9
    matrix_A, matrix_B, error = utils.linear_regression(xdata, udata, lamb)
    np.savetxt("data/track_spec/default.csv", track_spec, delimiter=",")
    np.savetxt("data/sys/LTI/matrix_A.csv", matrix_A, delimiter=",")
    np.savetxt("data/sys/LTI/matrix_B.csv", matrix_B, delimiter=",")
    print(matrix_A)
    print(matrix_B)


if __name__ == "__main__":
    linear_time_invariant()

