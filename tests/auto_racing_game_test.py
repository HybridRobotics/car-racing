import pickle
import sympy as sp
import numpy as np
import random
from control.lmpc_helper import LMPCPrediction
from racing import offboard
from utils import base, racing_env
from utils.constants import *


def test_racing_overtake():
    track_spec = np.genfromtxt("data/track_layout/l_shape.csv", delimiter=",")
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    lap_number = 4
    opti_traj_xcurv = np.genfromtxt(
        "data/optimal_traj/xcurv_l_shape.csv", delimiter=","
    )
    opti_traj_xglob = np.genfromtxt(
        "data/optimal_traj/xglob_l_shape.csv", delimiter=","
    )
    track = racing_env.ClosedTrack(track_spec, track_width=1.0)
    num_veh = 2
    alpha = 0.8
    timestep = 1.0 / 10.0
    ego, pid_controller, mpc_lti_controller = set_up_ego(timestep, track)
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
    simulator = offboard.CarRacingSim()
    simulator.set_timestep(timestep)
    simulator.set_track(track)
    simulator.add_vehicle(ego)
    simulator.set_opti_traj(opti_traj_xglob)
    t_symbol = sp.symbols("t")
    vehicles = set_up_other_vehicles(track, num_veh)
    pid_controller.set_racing_sim(simulator)
    mpc_lti_controller.set_racing_sim(simulator)
    lmpc_controller.set_racing_sim(simulator)
    lmpc_controller.set_vehicles_track()
    for iter in range(lap_number):
        if iter == 0:
            simulator.sim(sim_time=90, one_lap=True, one_lap_name="ego")
        elif iter == 1:
            ego.set_ctrl_policy(mpc_lti_controller)
            simulator.sim(
                sim_time=90,
                one_lap=True,
                one_lap_name="ego",
            )
        elif iter == 2:
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
            if iter == 3:
                for index in range(0, num_veh):
                    veh_name = "car" + str(index + 1)
                    vehicles[index].set_state_curvilinear_func(
                        t_symbol,
                        (0.7 + index * 0.02) * t_symbol + 5.5 + index * 2,
                        -0.5 + index * 0.3 + 0.0 * t_symbol,
                    )
                    vehicles[index].start_logging()
                    simulator.add_vehicle(vehicles[index])
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
    # print the lap timing infomation
    for i in range(0, lmpc_controller.iter):
        print(
            "lap time at iteration",
            i,
            "is",
            lmpc_controller.Qfun[0, i] * timestep,
            "s",
        )  
    file_name = "racing_game_m_shape"
    simulator.plot_simulation()
    simulator.plot_state("ego")
    simulator.plot_input("ego")
    simulator.animate(filename=file_name, ani_time=50, racing_game=True, imagemagick=True)


def set_up_ego(timestep, track):
    ego = offboard.DynamicBicycleModel(name="ego", param=base.CarParam(edgecolor="black"), system_param = base.SystemParam())
    ego.set_timestep(timestep)
    # run the pid controller for the first lap to collect data
    pid_controller = offboard.PIDTracking(vt=0.7, eyt=0.0)
    pid_controller.set_timestep(timestep)
    ego.set_ctrl_policy(pid_controller)
    pid_controller.set_track(track)
    ego.set_state_curvilinear(np.zeros((X_DIM,)))
    ego.set_state_global(np.zeros((X_DIM,)))
    ego.start_logging()
    ego.set_track(track)
    # run mpc-lti controller for the second lap to collect data
    mpc_lti_param = base.MPCTrackingParam(vt=0.7, eyt=0.0)
    mpc_lti_controller = offboard.MPCTracking(mpc_lti_param, ego.system_param)
    mpc_lti_controller.set_timestep(timestep)
    mpc_lti_controller.set_track(track)
    return ego, pid_controller, mpc_lti_controller


def set_up_lmpc(timestep, track, lap_number, alpha, opti_traj_xcurv, opti_traj_xglob, system_param):
    time_lmpc = 10000 * timestep
    lmpc_param = base.LMPCRacingParam(timestep=timestep, lap_number=lap_number, time_lmpc=time_lmpc)
    racing_game_param = base.RacingGameParam(timestep=timestep, alpha=alpha, num_horizon_planner=10)
    lmpc_controller = offboard.LMPCRacingGame(lmpc_param, racing_game_param=racing_game_param, system_param = system_param)
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
            offboard.NoDynamicsModel(name=veh_name, param=base.CarParam(edgecolor="orange"))
        )
        vehicles[index].set_track(track)
    return vehicles