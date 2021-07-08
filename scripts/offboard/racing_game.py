import matplotlib.pyplot as plt
import pickle
import sympy as sp
import numpy as np
import random
import math
from sim import offboard
from utils import racing_env, base, lmpc_helper


def lmpc_racing(args):
    track_layout = args["track_layout"]
    track_spec = np.genfromtxt(
        "data/track_layout/" + track_layout + ".csv", delimiter=","
    )
    lap_number = args["lap_number"]
    opti_traj_xcurv = np.genfromtxt(
        "data/optimal_traj/xcurv_" + track_layout + ".csv", delimiter=","
    )
    opti_traj_xglob = np.genfromtxt(
        "data/optimal_traj/xglob_" + track_layout + ".csv", delimiter=","
    )
    track_width = 0.8
    track = racing_env.ClosedTrack(track_spec, track_width)
    num_veh = args["number_other_agents"]
    if args["diff_alpha"]:
        alpha_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    else:
        alpha_list = [0.6]
    for alpha in alpha_list:
        if args["simulation"]:
            timestep = 1.0 / 10.0
            # load the trajectory generate from pid and mpc controller or build new ego and controller
            if args["direct_lmpc"]:
                with open("data/ego/ego_" + track_layout + ".obj", "rb") as handle:
                    ego = pickle.load(handle)
            else:
                ego, pid_controller, mpc_lti_controller = set_up_ego(timestep, track)
            # this will remove the noise in dynamics update
            if args["zero_noise"]:
                ego.set_zero_noise()
            # lmpc controller
            lmpc_controller, time_lmpc = set_up_lmpc(
                timestep,
                track,
                lap_number,
                alpha,
                opti_traj_xcurv,
                opti_traj_xglob,
            )
            # define a simulator
            simulator = offboard.CarRacingSim()
            simulator.set_timestep(timestep)
            simulator.set_track(track)
            simulator.add_vehicle(ego)
            simulator.set_opti_traj(opti_traj_xglob)
            # according to different settings, get different initial states for other vehicles
            t_symbol = sp.symbols("t")
            if args["sim_replay"]:
                with open("data/simulator/racing_game.obj", "rb") as handle:
                    simulator = pickle.load(handle)
                    num = len(simulator.vehicles) - 1
                    veh_list = set_up_other_vehicles(track, num)
                    for index in range(0, num):
                        veh_name = "car" + str(index + 1)
                        car_ini_xcurv = simulator.vehicles[veh_name].xcurv_list[0][0][:]
                        veh_list[index].set_state_curvilinear_func(
                            t_symbol,
                            car_ini_xcurv[0] * t_symbol + car_ini_xcurv[4],
                            car_ini_xcurv[5] + 0.0 * t_symbol,
                        )
                        simulator.add_vehicle(veh_list[index])
            elif args["random_other_agents"]:
                veh_list = set_up_other_vehicles(track, num_veh)
                for index in range(0, num_veh):
                    veh_name = "car" + str(index + 1)
                    veh_list[index].set_state_curvilinear_func(
                        t_symbol,
                        0.1 * random.randint(0, 10) * t_symbol
                        + 3
                        + random.randint(0, 14),
                        0.7 - 0.1 * random.randint(0, 14) + 0.0 * t_symbol,
                    )
                    simulator.add_vehicle(veh_list[index])
            else:
                veh_list = set_up_other_vehicles(track, num_veh)
                for index in range(0, num_veh):
                    veh_name = "car" + str(index + 1)
                    veh_list[index].set_state_curvilinear_func(
                        t_symbol,
                        (0.3 + index * 0.3) * t_symbol + 3 + index * 2,
                        -0.3 + index * 0.2 + 0.0 * t_symbol,
                    )
                    simulator.add_vehicle(veh_list[index])
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
                        simulator.sim(
                            sim_time=time_pid, one_lap=True, one_lap_name="ego"
                        )
                elif iter == 1:
                    if args["direct_lmpc"]:
                        pass
                    else:
                        # for the second lap, run the mpc-lti controller to collect data
                        ego.set_ctrl_policy(mpc_lti_controller)
                        simulator.sim(
                            sim_time=time_mpc_lti,
                            one_lap=True,
                            one_lap_name="ego",
                        )
                        with open(
                            "data/ego/ego_" + track_layout + ".obj", "wb"
                        ) as handle:
                            pickle.dump(ego, handle, protocol=pickle.HIGHEST_PROTOCOL)
                elif iter == 2:
                    lmpc_controller.add_trajectory(
                        ego.time_list,
                        ego.timestep,
                        ego.xcurv_list,
                        ego.xglob_list,
                        ego.u_list,
                        0,
                    )
                    lmpc_controller.add_trajectory(
                        ego.time_list,
                        ego.timestep,
                        ego.xcurv_list,
                        ego.xglob_list,
                        ego.u_list,
                        1,
                    )
                    # change the controller to lmpc controller
                    ego.set_ctrl_policy(lmpc_controller)
                    simulator.sim(sim_time=time_lmpc, one_lap=True, one_lap_name="ego")
                    ego.ctrl_policy.add_trajectory(
                        ego.time_list,
                        ego.timestep,
                        ego.xcurv_list,
                        ego.xglob_list,
                        ego.u_list,
                        2,
                    )
                else:
                    simulator.sim(sim_time=time_lmpc, one_lap=True, one_lap_name="ego")
                    ego.ctrl_policy.add_trajectory(
                        ego.time_list,
                        ego.timestep,
                        ego.xcurv_list,
                        ego.xglob_list,
                        ego.u_list,
                        iter,
                    )
            for i in range(0, lmpc_controller.iter):
                print(
                    "lap time at iteration",
                    i,
                    "is",
                    lmpc_controller.Qfun[0, i] * timestep,
                    "s",
                )
            if args["diff_alpha"]:
                file_name = "data/simulator/racing_game_alhpa_" + str(alpha) + ".obj"
            elif args["random_other_agents"]:
                file_name = "data/simulator/racing_game_random.obj"
            else:
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
            elif args["random_other_agents"]:
                file_name = "racing_game_random"
            else:
                file_name = "racing_game"
            simulator.animate(filename=file_name, only_last_lap=True)


def set_up_ego(timestep, track):
    ego = offboard.DynamicBicycleModel(
        name="ego", param=base.CarParam(edgecolor="black")
    )
    ego.set_timestep(timestep)
    # run the pid controller for the first lap to collect data
    time_pid = 90.0
    pid_controller = offboard.PIDTracking(vt=0.7, eyt=0.0)
    pid_controller.set_timestep(timestep)
    ego.set_ctrl_policy(pid_controller)
    pid_controller.set_track(track)
    ego.set_state_curvilinear(np.zeros((6,)))
    ego.set_state_global(np.zeros((6,)))
    ego.set_track(track)
    # run mpc-lti controller for the second lap to collect data
    time_mpc_lti = 90.0
    mpc_lti_param = base.MPCTrackingParam(vt=0.7, eyt=0.0)
    mpc_lti_controller = offboard.MPCTracking(mpc_lti_param)
    mpc_lti_controller.set_timestep(timestep)
    mpc_lti_controller.set_track(track)
    return ego, pid_controller, mpc_lti_controller


def set_up_lmpc(timestep, track, lap_number, alpha, opti_traj_xcurv, opti_traj_xglob):
    time_lmpc = 10000 * timestep
    lmpc_param = base.LMPCRacingParam(
        timestep=timestep, lap_number=lap_number, time_lmpc=time_lmpc
    )
    racing_game_param = base.RacingGameParam(timestep=timestep, alpha=alpha)
    lmpc_controller = offboard.LMPCRacingGame(
        lmpc_param, racing_game_param=racing_game_param
    )
    lmpc_controller.set_track(track)
    lmpc_controller.set_timestep(timestep)
    lmpc_controller.set_opti_traj(opti_traj_xcurv, opti_traj_xglob)
    lmpc_controller.openloop_prediction_lmpc = lmpc_helper.lmpc_prediction(
        lap_number=lap_number
    )
    return lmpc_controller, time_lmpc


def set_up_other_vehicles(track, num_veh):
    veh_list = []
    for index in range(0, num_veh):
        veh_name = "car" + str(index + 1)
        veh_list.append(
            offboard.NoDynamicsModel(
                name=veh_name, param=base.CarParam(edgecolor="orange")
            )
        )
        veh_list[index].set_track(track)
    return veh_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--track-layout", type=str)
    parser.add_argument("--lap-number", type=int)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--direct-lmpc", action="store_true")
    parser.add_argument("--sim-replay", action="store_true")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--diff-alpha", action="store_true")
    parser.add_argument("--random-other-agents", action="store_true")
    parser.add_argument("--number-other-agents", type=int)
    args = vars(parser.parse_args())
    lmpc_racing(args)
