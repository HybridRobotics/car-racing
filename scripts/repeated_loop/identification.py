import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('src')
sys.path.append('src/utils')
import repeated_loop, base, racing_env, system_id



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
    ego = repeated_loop.DynamicBicycleModelRepeatedLoop(
        name="ego", param=base.CarParam(edgecolor="black"))
    ego.set_state_curvilinear(np.array([0.3, 0, 0, 0, 0, 0]))
    ego.set_state_global(np.array([0.3, 0, 0, 0, 0, 0]))
    ego.set_ctrl_policy(repeated_loop.PIDTrackingRepeatedLoop(vt=0.5))
    ego.ctrl_policy.set_timestep(0.1)
    # setup simulation
    simulator = repeated_loop.CarRacingSimRepeatedLoop()
    simulator.set_timestep(0.1)
    simulator.set_track(track)
    simulator.add_vehicle(ego)
    simulator.sim(sim_time=500.0)
    # calculate linearized dynamics
    xdata = np.stack(simulator.vehicles["ego"].closedloop_xcurv, axis=0)
    udata = np.stack(simulator.vehicles["ego"].closedloop_u, axis=0)
    lamb = 1e-9
    matrix_A, matrix_B, error = system_id.linear_regression(
        xdata, udata, lamb)
    np.savetxt("data/track_layout/ellipse.csv", track_spec, delimiter=",")
    np.savetxt("data/sys/LTI/matrix_A.csv", matrix_A, delimiter=",")
    np.savetxt("data/sys/LTI/matrix_B.csv", matrix_B, delimiter=",")
    print(matrix_A)
    print(matrix_B)


if __name__ == "__main__":
    linear_time_invariant()
