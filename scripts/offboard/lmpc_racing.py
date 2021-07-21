import matplotlib.pyplot as plt
import pickle
import sympy as sp
import numpy as np
from sim import offboard
from utils import racing_env, base, lmpc_helper
from utils.constants import *
from utils import racing_env, base
from utils.lmpc_helper import LMPCPrediction


def lmpc_racing(args):
    track_layout = args["track_layout"]
    track_spec = np.genfromtxt(
        "data/track_layout/" + track_layout + ".csv", delimiter=","
    )
    lap_number = args["lap_number"]
    if args["simulation"]:
        track = racing_env.ClosedTrack(track_spec, track_width = 0.8)
        ego = offboard.DynamicBicycleModel(
            name="ego", param=base.CarParam(edgecolor="black")
        )
        timestep = 1.0 / 10.0
        ego.set_timestep(timestep)
        # run the pid controller for the first lap to collect data
        time_pid = 90.0
        pid_controller = offboard.PIDTracking(vt=1.2, eyt=0.1)
        pid_controller.set_timestep(timestep)
        ego.set_ctrl_policy(pid_controller)
        pid_controller.set_track(track)
        ego.set_state_curvilinear(np.zeros((X_DIM, 1)))
        ego.set_state_global(np.zeros((X_DIM, 1)))
        ego.set_track(track)
        # run mpc-lti controller for the second lap to collect data
        time_mpc_lti = 90.0
        mpc_lti_param = base.MPCTrackingParam(vt=1.2, eyt=0.1)
        mpc_lti_controller = offboard.MPCTracking(mpc_lti_param)
        mpc_lti_controller.set_timestep(timestep)
        mpc_lti_controller.set_track(track)
        # lmpc controller
        time_lmpc = 10000 * timestep
        lmpc_param = base.LMPCRacingParam(
            timestep=timestep,
            lap_number=lap_number,
            time_lmpc=time_lmpc,
        )
        lmpc_controller = offboard.LMPCRacingGame(lmpc_param)
        lmpc_controller.set_track(track)
        lmpc_controller.set_timestep(timestep)
        lmpc_controller.openloop_prediction = LMPCPrediction(
            N,
            int(round(time_lmpc / timestep)),
            num_ss_points,
            lap_number,
        )
        # define a simulator
        simulator = offboard.CarRacingSim()
        simulator.set_timestep(timestep)
        simulator.set_track(track)
        simulator.add_vehicle(ego)
        pid_controller.set_racing_sim(simulator)
        mpc_lti_controller.set_racing_sim(simulator)
        lmpc_controller.set_racing_sim(simulator)
        lmpc_controller.set_vehicles_track()
        # start simulation
        for iter in range(lap_number):
            # for the first lap, run the pid controller to collect data
            if iter == 0:
                simulator.sim(sim_time=time_pid, one_lap=True, one_lap_name="ego")
            elif iter == 1:
                # for the second lap, run the mpc-lti controller to collect data
                ego.set_ctrl_policy(mpc_lti_controller)
                simulator.sim(
                    sim_time=time_mpc_lti,
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
                simulator.sim(sim_time=time_lmpc, one_lap=True, one_lap_name="ego")
                ego.ctrl_policy.add_trajectory(
                    ego,
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
        with open("data/simulator/lmpc_racing.obj", "wb") as handle:
            pickle.dump(simulator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data/simulator/lmpc_racing.obj", "rb") as handle:
            simulator = pickle.load(handle)
    if args["plotting"]:
        simulator.plot_simulation()
        simulator.plot_state("ego")
        simulator.plot_input("ego")
    if args["animation"]:
        # currently, only animate the last lap of simulation
        simulator.animate(filename="lmpc_racing", only_last_lap=True)
    if args["save_trajectory"]:
        ego_xcurv = np.stack(
            simulator.vehicles["ego"].xcurv_list[lap_number - 1], axis=0
        )
        ego_xglob = np.stack(
            simulator.vehicles["ego"].xglob_list[lap_number - 1], axis=0
        )
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--track-layout", type=str)
    parser.add_argument("--lap-number", type=int)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--save-trajectory", action="store_true")
    args = vars(parser.parse_args())
    lmpc_racing(args)
