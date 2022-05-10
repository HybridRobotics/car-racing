import pickle
import sympy as sp
import numpy as np
import random
from scripts.control.lmpc_helper import LMPCPrediction
from scripts.racing import offboard
from scripts.utils import base, racing_env
from scripts.utils.constants import *


def racing_overtake(args, file_number):
    if args["save_trajectory"]:
        save_lmpc_traj = True
    else:
        save_lmpc_traj = False
    track_layout = args["track_layout"]
    track_spec = np.genfromtxt("data/track_layout/" + track_layout + ".csv", delimiter=",")
    lap_number = args["lap_number"]
    opti_traj_xcurv = np.genfromtxt(
        "data/optimal_traj/xcurv_" + track_layout + ".csv", delimiter=","
    )
    opti_traj_xglob = np.genfromtxt(
        "data/optimal_traj/xglob_" + track_layout + ".csv", delimiter=","
    )
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    num_veh = args["number_other_agents"]
    if args["diff_alpha"]:
        alphas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    else:
        alphas = [0.8]
    for alpha in alphas:
        if args["simulation"]:
            timestep = 1.0 / 10.0
            # load the trajectory generate from pid and mpc controller or build new ego and controller
            if args["direct_lmpc"]:
                with open("data/ego/ego_" + track_layout + "_multi_laps.obj", "rb") as handle:
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
                ego.system_param
            )
            # define a simulator
            simulator = offboard.CarRacingSimOffboard()
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
                    vehicles = set_up_other_vehicles(track, num)
                    for index in range(0, num):
                        veh_name = "car" + str(index + 1)
                        car_ini_xcurv = simulator.vehicles[veh_name].xcurvs[0][0][:]
                        vehicles[index].set_state_curvilinear_func(
                            t_symbol,
                            car_ini_xcurv[0] * t_symbol + car_ini_xcurv[4],
                            car_ini_xcurv[5] + 0.0 * t_symbol,
                        )
                        simulator.add_vehicle(vehicles[index])
            elif args["random_other_agents"] or args["multi_tests"]:
                vehicles = set_up_other_vehicles(track, num_veh)
                for index in range(0, num_veh):
                    veh_name = "car" + str(index + 1)
                    vehicles[index].set_state_curvilinear_func(
                        t_symbol,
                        0.1 * random.randint(0, 10) * t_symbol + 3 + random.randint(0, 14),
                        0.7 - 0.1 * random.randint(0, 14) + 0.0 * t_symbol,
                    )
                    vehicles[index].start_logging()
                    simulator.add_vehicle(vehicles[index])
            else:
                vehicles = set_up_other_vehicles(track, num_veh)
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
                        simulator.sim(sim_time=90, one_lap=True, one_lap_name="ego")
                elif iter == 1:
                    if args["direct_lmpc"]:
                        pass
                    else:
                        # for the second lap, run the mpc-lti controller to collect data
                        ego.set_ctrl_policy(mpc_lti_controller)
                        simulator.sim(
                            sim_time=90,
                            one_lap=True,
                            one_lap_name="ego",
                        )
                        with open("data/ego/ego_" + track_layout + ".obj", "wb") as handle:
                            pickle.dump(ego, handle, protocol=pickle.HIGHEST_PROTOCOL)
                elif iter == 2:
                    if args["direct_lmpc"]:
                        pass
                    else:
                        lmpc_controller.add_trajectory(
                            ego,
                            0,
                        )
                        lmpc_controller.add_trajectory(
                            ego,
                            1,
                        )
                        # change the controller to lmpc controller
                        ego.set_ctrl_policy(lmpc_controller)
                        simulator.sim(sim_time=time_lmpc, one_lap=True, one_lap_name="ego")
                        ego.ctrl_policy.add_trajectory(
                            ego,
                            2,
                        )
                else:
                    if args["direct_lmpc"]:
                        # to speed up the process, the proposed racing strategy will used the 5th and 6th iteraion as first groups of input states
                        if iter < 5:
                            pass
                        elif iter == 5:
                            if save_lmpc_traj:
                                with open(
                                    "data/ego/ego_" + track_layout + "_multi_laps.obj",
                                    "wb",
                                ) as handle:
                                    pickle.dump(ego, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        elif iter == 6:
                            # change the paramter in the following part will result in different initial setup for other vehicles
                            for index in range(0, num_veh):
                                veh_name = "car" + str(index + 1)
                                vehicles[index].set_state_curvilinear_func(
                                    t_symbol,
                                    (1.2 + index * 0.02) * t_symbol + 10.5 + index * 1.5,
                                    -0.5 + index * 0.3 + 0.0 * t_symbol,
                                )
                                vehicles[index].start_logging()
                                simulator.add_vehicle(vehicles[index])
                            lmpc_controller.add_trajectory(
                                ego,
                                4,
                            )
                            lmpc_controller.add_trajectory(
                                ego,
                                5,
                            )
                            ego.set_ctrl_policy(lmpc_controller)
                            ego.solver_time = []
                            ego.all_local_trajs = []
                            ego.all_splines = []
                            ego.xcurv_log = []
                            ego.lmpc_prediction = []
                            ego.mpc_cbf_prediction = []
                            simulator.sim(sim_time=time_lmpc, one_lap=True, one_lap_name="ego")
                            ego.ctrl_policy.add_trajectory(
                                ego,
                                iter,
                            )
                        else:
                            simulator.sim(sim_time=time_lmpc, one_lap=True, one_lap_name="ego")
                            ego.ctrl_policy.add_trajectory(
                                ego,
                                iter,
                            )
                    else:
                        simulator.sim(sim_time=time_lmpc, one_lap=True, one_lap_name="ego")
                        ego.ctrl_policy.add_trajectory(
                            ego,
                            iter,
                        )
                        if iter == 5:
                            if save_lmpc_traj:
                                with open(
                                    "data/ego/ego_" + track_layout + "_multi_laps.obj",
                                    "wb",
                                ) as handle:
                                    pickle.dump(ego, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print the lap timing infomation
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
            elif args["multi_tests"]:
                file_name = "data/simulator/test_results/racing_game" + str(file_number) + ".obj"
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
        if args["save_trajectory"]:
            ego_xcurv = np.stack(simulator.vehicles["ego"].xcurvs[lap_number - 1], axis=0)
            ego_xglob = np.stack(simulator.vehicles["ego"].xglobs[lap_number - 1], axis=0)
            np.savetxt(
                "data/optimal_traj/xcurv_" + track_layout + ".csv",
                ego_xcurv,
                delimiter=",",
            )
            np.savetxt(
                "data/optimal_traj/xglob_" + track_layout + ".csv",
                ego_xglob,
                delimiter=",",
            )
        if args["animation"]:
            if args["diff_alpha"]:
                file_name = "racing_game_alpha_" + str(alpha)
            elif args["random_other_agents"]:
                file_name = "racing_game_random"
            else:
                file_name = "racing_game_" + track_layout
            simulator.animate(filename=file_name, ani_time=250, racing_game=True)


def set_up_ego(timestep, track):
    ego = offboard.DynamicBicycleModelOffboard(name="ego", param=base.CarParam(edgecolor="black"), system_param = base.SystemParam())
    ego.set_timestep(timestep)
    # run the pid controller for the first lap to collect data
    pid_controller = offboard.PidTrackingOffboard(vt=0.7, eyt=0.0)
    pid_controller.set_timestep(timestep)
    ego.set_ctrl_policy(pid_controller)
    pid_controller.set_track(track)
    ego.set_state_curvilinear(np.zeros((X_DIM,)))
    ego.set_state_global(np.zeros((X_DIM,)))
    ego.start_logging()
    ego.set_track(track)
    # run mpc-lti controller for the second lap to collect data
    mpc_lti_param = base.MpcTrackingParam(vt=0.7, eyt=0.0)
    mpc_lti_controller = offboard.MpcTrackingOffboard(mpc_lti_param, ego.system_param)
    mpc_lti_controller.set_timestep(timestep)
    mpc_lti_controller.set_track(track)
    return ego, pid_controller, mpc_lti_controller


def set_up_lmpc(timestep, track, lap_number, alpha, opti_traj_xcurv, opti_traj_xglob, system_param):
    time_lmpc = 10000 * timestep
    lmpc_param = base.LmpcRacingParam(timestep=timestep, lap_number=lap_number, time_lmpc=time_lmpc)
    racing_game_param = base.RacingGameParam(timestep=timestep, alpha=alpha, num_horizon_planner=10)
    lmpc_controller = offboard.LmpcRacingGameOffboard(lmpc_param, racing_game_param=racing_game_param, system_param = system_param)
    lmpc_controller.set_track(track)
    lmpc_controller.set_timestep(timestep)
    lmpc_controller.set_opti_traj(opti_traj_xcurv, opti_traj_xglob)
    lmpc_controller.openloop_prediction = LMPCPrediction(lap_number=lap_number)
    return lmpc_controller, time_lmpc


def set_up_other_vehicles(track, num_veh):
    vehicles = []
    for index in range(0, num_veh):
        veh_name = "car" + str(index + 1)
        vehicles.append(
            offboard.NoDynamicsModelOffboard(name=veh_name, param=base.CarParam(edgecolor="orange"))
        )
        vehicles[index].set_track(track)
    return vehicles


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
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--multi-tests", action="store_true")
    args = vars(parser.parse_args())
    if args["multi_tests"]:
        for index in range(1, 101):
            racing_overtake(args, index)
    else:
        racing_overtake(args, 0)
