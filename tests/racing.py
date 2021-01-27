import numpy as np
import matplotlib.pyplot as plt
import racing_env, racing_car, racing_sim, policy, utils


def racing(args):
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
    ego.set_ctrl_policy(policy.MPCTracking(matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6))
    # setup surrounding cars
    car1 = racing_car.DynamicBicycleModel(
        name="car1", car_param=racing_car.CarParam(color="b"), x=np.zeros((6,)), x_glob=np.zeros((6,)),
    )
    # car1.set_ctrl_policy(policy.MPCTracking(matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.2))
    car1.set_ctrl_policy(policy.PIDTracking(vt=0.6, eyt=0.2))
    car2 = racing_car.DynamicBicycleModel(
        name="car2", car_param=racing_car.CarParam(color="b"), x=np.zeros((6,)), x_glob=np.zeros((6,)),
    )
    # car2.set_ctrl_policy(policy.MPCTracking(matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=-0.2))
    car2.set_ctrl_policy(policy.PIDTracking(vt=0.6, eyt=-0.2))
    # setup simulation
    simulator = racing_sim.CarRacingSim()
    simulator.set_timestep(0.1)
    simulator.set_track(track)
    simulator.add_vehicle(ego)
    simulator.add_vehicle(car1)
    simulator.add_vehicle(car2)
    simulator.sim(sim_time=100.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args())
    racing(args)
