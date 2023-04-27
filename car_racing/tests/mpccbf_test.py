import pickle

import sympy as sp
import numpy as np

from planner import *
from racing_env import *


def racing(args):
    track_layout = args["track_layout"]
    track_spec = np.genfromtxt("data/track_layout/" + track_layout + ".csv", delimiter=",")
    if args["simulation"]:
        track = ClosedTrack(track_spec, track_width=1.0)
        # setup ego car
        ego = OffboardDynamicBicycleModel(name="ego", param=CarParam(edgecolor="black"), system_param = SystemParam())
        mpc_cbf_param = MPCCBFRacingParam(vt=0.8)
        ego.set_state_curvilinear(np.zeros((X_DIM,)))
        ego.set_state_global(np.zeros((X_DIM,)))
        ego.start_logging()
        ego.set_ctrl_policy(MPCCBFRacing(mpc_cbf_param, ego.system_param))
        ego.ctrl_policy.set_timestep(0.1)
        ego.set_track(track)
        ego.ctrl_policy.set_track(track)
        # setup surrounding cars
        t_symbol = sp.symbols("t")
        car1 = OffboardNoDynamicsModel(name="car1", param=CarParam(edgecolor="orange"))
        car1.set_track(track)
        car1.set_state_curvilinear_func(t_symbol, 0.2 * t_symbol + 4.0, 0.1 + 0.0 * t_symbol)
        car1.start_logging()
        car2 = OffboardNoDynamicsModel(name="car2", param=CarParam(edgecolor="orange"))
        car2.set_track(track)
        car2.set_state_curvilinear_func(t_symbol, 0.2 * t_symbol + 10.0, -0.1 + 0.0 * t_symbol)
        car2.start_logging()
        # setup simulation
        simulator = RacingSim()
        simulator.set_timestep(0.1)
        simulator.set_track(track)
        simulator.add_vehicle(ego)
        ego.ctrl_policy.set_racing_sim(simulator)
        simulator.add_vehicle(car1)
        simulator.add_vehicle(car2)
        simulator.sim(sim_time=50.0)
        with open("data/simulator/racing.obj", "wb") as handle:
            pickle.dump(simulator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data/simulator/racing.obj", "rb") as handle:
            simulator = pickle.load(handle)
    if args["plotting"]:
        simulator.plot_simulation()
        simulator.plot_state("ego")
    if args["animation"]:
        simulator.animate(filename="racing", imagemagick=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--track-layout", type=str)
    args = vars(parser.parse_args())
    racing(args)
