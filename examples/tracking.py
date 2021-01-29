import numpy as np
import matplotlib.pyplot as plt
import racing_env, racing_car, racing_sim, policy, utils


def tracking(args):
    track_spec = np.genfromtxt("data/track_spec/default.csv", delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)
    matrix_A = np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=",")
    matrix_B = np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=",")
    matrix_Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.0, 100.0])
    matrix_R = np.diag([1.0, 10.0])
    # setup ego car
    ego = racing_car.DynamicBicycleModel(name="ego", param=racing_car.CarParam(color="r"))
    ego.set_curvilinear_state(np.zeros((6,)))
    ego.set_global_state(np.zeros((6,)))
    if args["ctrl_policy"] == "pid":
        ego.set_ctrl_policy(policy.PIDTracking(vt=0.6))
    elif args["ctrl_policy"] == "mpc-lti":
        ego.set_ctrl_policy(policy.MPCTracking(matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6))
    else:
        raise NotImplementedError
    # setup simulation
    simulator = racing_sim.CarRacingSim()
    simulator.set_timestep(0.1)
    simulator.set_track(track)
    simulator.add_vehicle(ego)
    simulator.setup()
    simulator.sim(sim_time=100.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl_policy", type=str, default="pid")
    args = vars(parser.parse_args())
    tracking(args)
