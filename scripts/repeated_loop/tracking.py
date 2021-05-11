import matplotlib.pyplot as plt
import pickle
import numpy as np
import repeated_loop, base, racing_env

def tracking(args):
    track_layout = args["track_layout"]
    track_spec = np.genfromtxt("data/track_layout/"+track_layout+".csv" ,delimiter=",")
    if args["simulation"]:
        track_width = 1.0
        track = racing_env.ClosedTrack(track_spec, track_width)
        # setup ego car
        ego = repeated_loop.DynamicBicycleModelRepeatedLoop(
            name="ego", param=base.CarParam(edgecolor="black"))
        ego.set_state_curvilinear(np.zeros((6,)))
        ego.set_state_global(np.zeros((6,)))
        if args["ctrl_policy"] == "pid":
            ego.set_ctrl_policy(repeated_loop.PIDTrackingRepeatedLoop(vt=0.8))
        elif args["ctrl_policy"] == "mpc-lti":
            matrix_A = np.genfromtxt(
                "data/sys/LTI/matrix_A.csv", delimiter=",")
            matrix_B = np.genfromtxt(
                "data/sys/LTI/matrix_B.csv", delimiter=",")
            matrix_Q = np.diag([10.0, 0.0, 0.0, 0.0, 0.0, 10.0])
            matrix_R = np.diag([0.1, 0.1])
            ego.set_ctrl_policy(repeated_loop.MPCTrackingRepeatedLoop(
                matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.8))
        else:
            raise NotImplementedError
        ego.ctrl_policy.set_timestep(0.1)
        ego.set_track(track)
        # setup simulation
        simulator = repeated_loop.CarRacingSimRepeatedLoop()
        simulator.set_timestep(0.1)
        simulator.set_track(track)
        simulator.add_vehicle(ego)
        simulator.sim(sim_time=50.0)
        with open("data/simulator/tracking.obj", "wb") as handle:
            pickle.dump(simulator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data/simulator/tracking.obj", "rb") as handle:
            simulator = pickle.load(handle)
    if args["plotting"]:
        simulator.plot_simulation()
        simulator.plot_state("ego")
        simulator.plot_input("ego")
    if args["animation"]:
        simulator.animate(filename="tracking")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ctrl-policy", type=str)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--track-layout", type=str)
    args = vars(parser.parse_args())
    tracking(args)
