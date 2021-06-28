import matplotlib.pyplot as plt
import pickle
import sympy as sp
import numpy as np
import random
import math
from sim import offboard
from utils import racing_env, base, lmpc_helper


def lmpc_racing(args, index):
    track_layout = args["track_layout"]
    track_spec = np.genfromtxt(
        "data/track_layout/"+track_layout+".csv", delimiter=",")
    lap_number = args["lap_number"]
    opti_traj_xcurv = np.genfromtxt(
        "data/optimal_traj/xcurv_"+track_layout+".csv", delimiter=",")
    opti_traj_xglob = np.genfromtxt(
        "data/optimal_traj/xglob_"+track_layout+".csv", delimiter=",")
    if args["simulation"]:
        track_width = 0.8
        track = racing_env.ClosedTrack(track_spec, track_width)
        timestep = 1.0/10.0
        # load the trajectory generate from pid and mpc controller
        if args["direct_lmpc"]:
            with open("data/ego/ego.obj", "rb") as handle:
                ego = pickle.load(handle)
        else:
            ego = offboard.DynamicBicycleModel(
                name="ego", param=base.CarParam(edgecolor="black"))
            ego.set_timestep(timestep)
            # run the pid controller for the first lap to collect data
            time_pid = 50.0
            pid_controller = offboard.PIDTracking(vt=0.7, eyt=0.0)
            pid_controller.set_timestep(timestep)
            ego.set_ctrl_policy(pid_controller)
            pid_controller.set_track(track)
            ego.set_state_curvilinear(np.zeros((6,)))
            ego.set_state_global(np.zeros((6,)))
            ego.set_track(track)
            # run mpc-lti controller for the second lap to collect data
            time_mpc_lti = 50.0
            mpc_lti_param = base.MPCTrackingParam(vt=0.7, eyt=0.0)
            mpc_lti_controller = offboard.MPCTracking(mpc_lti_param)
            mpc_lti_controller.set_timestep(timestep)
            mpc_lti_controller.set_track(track)
        # lmpc controller
        time_lmpc = 5000*timestep
        if args["diff_alpha"]:
            alpha = 1.0 - index*0.05
        else:
            alpha = 0.6
        lmpc_param = base.LMPCRacingParam(
            timestep=timestep, lap_number=lap_number, time_lmpc=time_lmpc)
        racing_game_param = base.RacingGameParam(
            timestep=timestep, alpha=alpha)
        lmpc_controller = offboard.LMPCRacingGame(
            lmpc_param, racing_game_param)
        lmpc_controller.set_track(track)
        lmpc_controller.set_timestep(timestep)
        lmpc_controller.set_opti_traj(opti_traj_xcurv, opti_traj_xglob)
        lmpc_controller.openloop_prediction_lmpc = lmpc_helper.lmpc_prediction(
            lap_number=lap_number)
        # define a simulator
        simulator = offboard.CarRacingSim()
        simulator.set_timestep(timestep)
        simulator.set_track(track)
        simulator.add_vehicle(ego)
        simulator.set_opti_traj(opti_traj_xglob)
        # to do the simulation with a specific scenario, change the following file path and load the data
        if args["sim_back"]:
            with open("data/simulator/racing_game.obj", "rb") as handle:
                simulator = pickle.load(handle)
            car1_ini_xcurv = simulator.vehicles["car1"].xcurv_list[0][0][:]
            car2_ini_xcurv = simulator.vehicles["car2"].xcurv_list[0][0][:]
            car3_ini_xcurv = simulator.vehicles["car3"].xcurv_list[0][0][:]
        t_symbol = sp.symbols("t")
        if args["diy_sim"]:
            car1 = offboard.DynamicBicycleModel(
                name="car1", param=base.CarParam(edgecolor="orange"))
            x_car1_curv = np.zeros((6,))
            x_car1_curv[4] = 0
            x_car1_xglob = np.zeros((6,))
            x_car1_xglob[0], x_car1_xglob[1] = racing_env.get_global_position(
                track.lap_length, track.width, track.point_and_tangent, x_car1_curv[4], x_car1_curv[5])
            car1.set_state_curvilinear(x_car1_curv)
            car1.set_state_global(x_car1_xglob)
            car1.set_ctrl_policy(offboard.PIDTracking(vt=0.7, eyt=0.4))
            car1.ctrl_policy.set_timestep(0.1)
            car1.set_track(track)
            simulator.add_vehicle(car1)
        else:
            car1 = offboard.NoDynamicsModel(
                name="car1", param=base.CarParam(edgecolor="orange"))
            car1.set_track(track)
            car2 = offboard.NoDynamicsModel(
                name="car2", param=base.CarParam(edgecolor="orange"))
            car2.set_track(track)
            car3 = offboard.NoDynamicsModel(
                name="car3", param=base.CarParam(edgecolor="orange"))
            car3.set_track(track)
            if args["diff_alpha"]:
                car1.set_state_curvilinear_func(
                    t_symbol, 0.42 * t_symbol + 15.5, -0.31 + 0.0 * t_symbol)
                car2.set_state_curvilinear_func(
                    t_symbol, 0.75 * t_symbol + 6.0, 0.29 + 0.0 * t_symbol)
                car3.set_state_curvilinear_func(
                    t_symbol, 0.52 * t_symbol + 10.5, 0.45 + 0.0 * t_symbol)
            if args["random_sim"]:
                car1.set_state_curvilinear_func(t_symbol, 0.1*random.randint(
                    0, 10) * t_symbol + 3+random.randint(0, 14), 0.7-0.1*random.randint(0, 14) + 0.0 * t_symbol)
                car2.set_state_curvilinear_func(t_symbol, 0.1*random.randint(
                    0, 10) * t_symbol + 3+random.randint(0, 14), 0.7-0.1*random.randint(0, 14) + 0.0 * t_symbol)
                car3.set_state_curvilinear_func(t_symbol, 0.1*random.randint(
                    0, 10) * t_symbol + 3+random.randint(0, 14), 0.7-0.1*random.randint(0, 14) + 0.0 * t_symbol)
            if args["sim_back"]:
                car1.set_state_curvilinear_func(
                    t_symbol, car1_ini_xcurv[0] * t_symbol + car1_ini_xcurv[4], car1_ini_xcurv[5] + 0.0 * t_symbol)
                car2.set_state_curvilinear_func(
                    t_symbol, car2_ini_xcurv[0] * t_symbol + car2_ini_xcurv[4], car2_ini_xcurv[5] + 0.0 * t_symbol)
                car3.set_state_curvilinear_func(
                    t_symbol, car3_ini_xcurv[0] * t_symbol + car3_ini_xcurv[4], car3_ini_xcurv[5] + 0.0 * t_symbol)
            simulator.add_vehicle(car1)
            simulator.add_vehicle(car2)
            simulator.add_vehicle(car3)
        if args["direct_lmpc"]:
            pass
        else:
            pid_controller.set_racing_sim(simulator)
            mpc_lti_controller.set_racing_sim(simulator)
        lmpc_controller.set_racing_sim(simulator)
        lmpc_controller.set_vehicles_track()
        # start simulation
        for iter in range(lap_number):
            # for the first lap, run the pid controller to collect data
            if iter == 0:
                if args["direct_lmpc"]:
                    pass
                else:
                    simulator.sim(sim_time=time_pid,
                                  one_lap_flag=True, one_lap_name="ego")
            elif iter == 1:
                if args["direct_lmpc"]:
                    pass
                else:
                    # for the second lap, run the mpc-lti controller to collect data
                    ego.set_ctrl_policy(mpc_lti_controller)
                    simulator.sim(sim_time=time_mpc_lti,
                                  one_lap_flag=True, one_lap_name="ego")
                    with open("data/ego/ego.obj", "wb") as handle:
                        pickle.dump(
                            ego, handle, protocol=pickle.HIGHEST_PROTOCOL)
            elif iter == 2:
                lmpc_controller.add_trajectory(
                    ego.time_list, ego.timestep, ego.xcurv_list, ego.xglob_list, ego.u_list, 0)
                lmpc_controller.add_trajectory(
                    ego.time_list, ego.timestep, ego.xcurv_list, ego.xglob_list, ego.u_list, 1)
                # change the controller to lmpc controller
                ego.set_ctrl_policy(lmpc_controller)
                simulator.sim(sim_time=time_lmpc,
                              one_lap_flag=True, one_lap_name="ego")
                ego.ctrl_policy.add_trajectory(
                    ego.time_list, ego.timestep, ego.xcurv_list, ego.xglob_list, ego.u_list, 2)
            else:
                simulator.sim(sim_time=time_lmpc,
                              one_lap_flag=True, one_lap_name="ego")
                ego.ctrl_policy.add_trajectory(
                    ego.time_list, ego.timestep, ego.xcurv_list, ego.xglob_list, ego.u_list, iter)
        for i in range(0, lmpc_controller.iter):
            print("lap time at iteration", i, "is",
                  lmpc_controller.Qfun[0, i]*timestep, "s")
        if args["diff_alpha"]:
            file_name = "data/simulator/racing_game_alhpa_" + \
                str(alpha) + ".obj"
        if args["random_sim"]:
            file_name = "data/simulator/racing_game_random_" + \
                str(index) + ".obj"
        if args["diy_sim"]:
            file_name = "data/simulator/racing_game.obj"
        if args["sim_back"]:
            file_name = "data/simulator/racing_game.obj"
        with open(file_name, "wb") as handle:
            pickle.dump(simulator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data/simulator/racing_game.obj", "rb") as handle:
            simulator = pickle.load(handle)
    if args["plotting"]:
        simulator.plot_simulation()
        simulator.plot_state("ego")
        simulator.plot_input("ego")
    if args["animation"]:
        if args["diff_alpha"]:
            file_name = "racing_game_alpha_" + str(alpha)
        if args["random_sim"]:
            file_name = "racing_game_random_" + str(index)
        if args["diy_sim"]:
            file_name = "racing_game"
        if args["sim_back"]:
            file_name = "racing_game"
        simulator.animate(filename=file_name, only_last_lap=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-layout", type=str)
    parser.add_argument("--lap-number", type=int)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--direct-lmpc", action="store_true")
    parser.add_argument("--diff-alpha", action="store_true")
    parser.add_argument("--random-sim", action="store_true")
    parser.add_argument("--diy-sim", action="store_true")
    parser.add_argument("--sim-back", action="store_true")
    args = vars(parser.parse_args())
    if args["diy_sim"] or args["sim_back"]:
        lmpc_racing(args, 0)
    else:
        for index in range(10):
            lmpc_racing(args, index)
