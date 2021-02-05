import numpy as np
import pickle
import matplotlib.pyplot as plt
import racing_env, racing_car, racing_sim, policy, utils


def tracking(args):
    if args["simulation"]:
        track_spec = np.genfromtxt("data/track_spec/default.csv", delimiter=",")
        track_width = 1.0
        track = racing_env.ClosedTrack(track_spec, track_width)
        # setup ego car
        ego = racing_car.DynamicBicycleModel(name="ego", param=racing_car.CarParam(edgecolor="black"))
        ego.set_state_curvilinear(np.zeros((6,)))
        ego.set_state_global(np.zeros((6,)))
        if args["ctrl_policy"] == "pid":
            ego.set_ctrl_policy(policy.PIDTracking(vt=0.8))
        elif args["ctrl_policy"] == "mpc-lti":
            matrix_A = np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=",")
            matrix_B = np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=",")
            matrix_Q = np.diag([10.0, 0.0, 0.0, 0.0, 0.0, 10.0])
            matrix_R = np.diag([0.1, 0.1])
            ego.set_ctrl_policy(policy.MPCTracking(matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.8))
        else:
            raise NotImplementedError
        ego.ctrl_policy.set_timestep(0.1)
        # setup simulation
        simulator = racing_sim.CarRacingSim()
        simulator.set_timestep(0.1)
        simulator.set_track(track)
        simulator.add_vehicle(ego)
        simulator.sim(sim_time=50.0)
        with open("./data/simulator/tracking.obj", "wb") as handle:
            pickle.dump(simulator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("./data/simulator/tracking.obj", "rb") as handle:
            simulator = pickle.load(handle)
    if args["plotting"]:
        simulator.plot_simulation()
        simulator.plot_state("ego")
    if args["animation"]:
        simulator.animate(filename="tracking")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl-policy", type=str)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--animation", action="store_true")
    args = vars(parser.parse_args())
    tracking(args)
