import numpy as np
import matplotlib.pyplot as plt
import racing_env, racing_car, racing_sim, policy, utils


def linear_time_invariant_identification():
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
    ego = racing_car.DynamicBicycleModel(
        name="ego",
        car_param=racing_car.CarParam(color="r"),
        x=np.array([0.3, 0, 0, 0, 0, 0]),
        x_glob=np.array([0.3, 0, 0, 0, 0, 0]),
    )
    ego.set_ctrl_policy(policy.PIDSpeedTracking(vt=0.5))
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


def mpc_tracking():
    track_spec = np.genfromtxt("data/track_spec/default.csv", delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)
    matrix_A = np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=",")
    matrix_B = np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=",")
    matrix_Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 100.0])
    matrix_R = np.diag([1.0, 10.0])
    # setup ego car
    ego = racing_car.DynamicBicycleModel(
        name="ego", car_param=racing_car.CarParam(color="r"), x=np.zeros((6,)), x_glob=np.zeros((6,)),
    )
    ego.set_ctrl_policy(policy.MPCSpeedTracking(matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6))
    # setup simulation
    simulator = racing_sim.CarRacingSim()
    simulator.set_timestep(0.1)
    simulator.set_track(track)
    simulator.add_vehicle(ego)
    simulator.sim(sim_time=100.0)


if __name__ == "__main__":
    linear_time_invariant_identification()
    mpc_tracking()
