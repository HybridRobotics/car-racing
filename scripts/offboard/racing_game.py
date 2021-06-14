import matplotlib.pyplot as plt
import pickle
import sympy as sp
import numpy as np
from sim import offboard
from utils import racing_env, base, lmpc_helper


def lmpc_racing(args):
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
        ego = offboard.DynamicBicycleModel(
            name="ego", param=base.CarParam(edgecolor="black"))
        timestep = 1.0/10.0
        ego.set_timestep(timestep)
        vt = 0.7
        N = 12
        xdim = 6
        udim = 2
        # run the pid controller for the first lap to collect data
        time_pid = 50.0
        pid_controller = offboard.PIDTracking(vt=vt, eyt=0.0)
        pid_controller.set_timestep(timestep)
        ego.set_ctrl_policy(pid_controller)
        pid_controller.set_track(track)
        ego.set_state_curvilinear(np.zeros((6,)))
        ego.set_state_global(np.zeros((6,)))
        ego.set_track(track)
        # run mpc-lti controller for the second lap to collect data
        time_mpc_lti = 50.0
        matrix_A = np.genfromtxt(
            "data/sys/LTI/matrix_A.csv", delimiter=",")
        matrix_B = np.genfromtxt(
            "data/sys/LTI/matrix_B.csv", delimiter=",")
        matrix_Q = np.diag([10.0, 0.0, 0.0, 4.0, 0.0, 40.0])
        matrix_R = np.diag([0.1, 0.1])
        mpc_lti_param = base.MPCTrackingParam(
            matrix_A, matrix_B, matrix_Q, matrix_R, vt=vt, eyt=0.0)
        mpc_lti_controller = offboard.MPCTracking(mpc_lti_param)
        mpc_lti_controller.set_timestep(timestep)
        mpc_lti_controller.set_track(track)
        t_symbol = sp.symbols("t")
        car1 = offboard.NoDynamicsModel(
            name="car1", param=base.CarParam(edgecolor="orange"))
        car1.set_track(track)
        car1.set_state_curvilinear_func(
            t_symbol, 0.6 * t_symbol + 3.0, -0.25 + 0.0 * t_symbol)

        car2 = offboard.NoDynamicsModel(
            name="car2", param=base.CarParam(edgecolor="orange"))
        car2.set_track(track)
        car2.set_state_curvilinear_func(
            t_symbol, 0.6 * t_symbol + 13.0, 0.35 + 0.0 * t_symbol)

        # lmpc controller
        num_ss_it = 2
        num_ss_points = 32 + N
        shift = 0
        points_lmpc = 5000
        time_lmpc = points_lmpc * timestep
        # Cost on the slack variable for the terminal constraint
        matrix_Qslack = 5*np.diag([10, 1, 1, 1, 10, 1])
        # State cost x = [vx, vy, wz, epsi, s, ey]
        matrix_Q_LMPC = 0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Input cost u = [delta, a]
        matrix_R_LMPC = 1 * np.diag([1.0, 1.0])
        # Input rate cost u
        matrix_dR_LMPC = 5 * np.diag([1.0, 1.0])
        lmpc_param = base.LMPCRacingParam(num_ss_points, num_ss_it, N, matrix_Qslack, matrix_Q_LMPC,
                                          matrix_R_LMPC, matrix_dR_LMPC, shift, timestep, lap_number, time_lmpc)
        racing_game_param = base.RacingGameParam(
            matrix_A, matrix_B, matrix_Q, matrix_R, timestep)
        lmpc_controller = offboard.LMPCRacingGame(
            lmpc_param, racing_game_param)
        lmpc_controller.set_track(track)
        lmpc_controller.set_timestep(timestep)
        lmpc_controller.set_opti_traj(opti_traj_xcurv, opti_traj_xglob)
        lmpc_controller.openloop_prediction_lmpc = lmpc_helper.lmpc_prediction(
            N, xdim, udim, points_lmpc, num_ss_points, lap_number)
        # define a simulator
        simulator = offboard.CarRacingSim()
        simulator.set_timestep(timestep)
        simulator.set_track(track)
        simulator.add_vehicle(ego)
        simulator.add_vehicle(car1)
        simulator.add_vehicle(car2)
        pid_controller.set_racing_sim(simulator)
        mpc_lti_controller.set_racing_sim(simulator)
        lmpc_controller.set_racing_sim(simulator)
        lmpc_controller.set_vehicles_track()

        # start simulation
        for iter in range(lap_number):
            # for the first lap, run the pid controller to collect data
            if iter == 0:
                simulator.sim(sim_time=time_pid,
                              one_lap_flag=True, one_lap_name="ego")
            elif iter == 1:
                # for the second lap, run the mpc-lti controller to collect data
                ego.set_ctrl_policy(mpc_lti_controller)
                simulator.sim(sim_time=time_mpc_lti,
                              one_lap_flag=True, one_lap_name="ego")
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
        with open("data/simulator/racing_game.obj", "wb") as handle:
            pickle.dump(simulator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data/simulator/racing_game.obj", "rb") as handle:
            simulator = pickle.load(handle)
    if args["plotting"]:
        simulator.plot_simulation()
        simulator.plot_state("ego")
        simulator.plot_input("ego")
    if args["animation"]:
        # currently, only animate the last lap of simulation
        simulator.animate(filename="racing_game", only_last_lap=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-layout", type=str)
    parser.add_argument("--lap-number", type=int)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--animation", action="store_true")
    args = vars(parser.parse_args())
    lmpc_racing(args)
