import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import racing_env, racing_car, racing_sim, policy, utils


def racing(args):
    track_spec = np.genfromtxt("data/track_spec/default.csv", delimiter=",")
    track_width = 1.0
    track = racing_env.ClosedTrack(track_spec, track_width)
    matrix_A = np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=",")
    matrix_B = np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=",")
    matrix_Q = np.diag([100.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    matrix_R = np.diag([0.1, 0.1])
    # setup ego car
    ego = racing_car.DynamicBicycleModel(name="ego", param=racing_car.CarParam(color="r"))
    ego.set_state_curvilinear(np.zeros((6,)))
    ego.set_state_global(np.zeros((6,)))
    ego.set_ctrl_policy(policy.MPCCBFRacing(matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.8))
    ego.ctrl_policy.set_timestep(0.1)
    # setup surrounding cars
    t_symbol = sp.symbols("t")
    car1 = racing_car.NoPolicyModel(name="car1", param=racing_car.CarParam(color="b"))
    car1.set_state_curvilinear_func(t_symbol, 0.1 * t_symbol + 1.0, 0.1 + 0.0 * t_symbol)
    car2 = racing_car.NoPolicyModel(name="car2", param=racing_car.CarParam(color="b"))
    car2.set_state_curvilinear_func(t_symbol, 0.1 * t_symbol + 10.0, -0.1 + 0.0 * t_symbol)
    # setup simulation
    simulator = racing_sim.CarRacingSim()
    simulator.set_timestep(0.1)
    simulator.set_track(track)
    simulator.add_vehicle(ego)
    ego.ctrl_policy.set_racing_sim(simulator)
    simulator.add_vehicle(car1)
    simulator.add_vehicle(car2)
    simulator.sim(sim_time=50.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args())
    racing(args)
