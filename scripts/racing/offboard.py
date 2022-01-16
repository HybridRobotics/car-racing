import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from scripts.utils import base, racing_env
from scripts.system import vehicle_dynamics
from matplotlib import animation
from scripts.utils.constants import *
import pickle

# off-board controller
class PIDTracking(base.PIDTracking):
    def __init__(self, vt=0.6, eyt=0.0):
        base.PIDTracking.__init__(self, vt, eyt)


class MPCTracking(base.MPCTracking):
    def __init__(self, mpc_lti_param):
        base.MPCTracking.__init__(self, mpc_lti_param)


class MPCCBFRacing(base.MPCCBFRacing):
    def __init__(self, mpc_cbf_param):
        base.MPCCBFRacing.__init__(self, mpc_cbf_param)
        self.realtime_flag = False


class LMPCRacingGame(base.LMPCRacingGame):
    def __init__(self, lmpc_param, racing_game_param=None):
        base.LMPCRacingGame.__init__(self, lmpc_param, racing_game_param=racing_game_param)
        self.realt = False


# off-board dynamic model
class DynamicBicycleModel(base.DynamicBicycleModel):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        base.DynamicBicycleModel.__init__(self, name=name, param=param)

    # in this estimation, the vehicles is assumed to move with input is equal to zero
    def get_estimation(self, xglob, xcurv):
        curv = racing_env.get_curvature(self.lap_length, self.point_and_tangent, xcurv[4])
        xcurv_est = np.zeros((X_DIM,))
        xglob_est = np.zeros((X_DIM,))
        xcurv_est[0:3] = xcurv[0:3]
        xcurv_est[3] = xcurv[3] + self.timestep * (
            xcurv[2]
            - (xcurv[0] * np.cos(xcurv[3]) - xcurv[1] * np.sin(xcurv[3]))
            / (1 - curv * xcurv[5])
            * curv
        )
        xcurv_est[4] = xcurv[4] + self.timestep * (
            (xcurv[0] * np.cos(xcurv[3]) - xcurv[1] * np.sin(xcurv[3])) / (1 - curv * xcurv[5])
        )
        xcurv_est[5] = xcurv[5] + self.timestep * (
            xcurv[0] * np.sin(xcurv[3]) + xcurv[1] * np.cos(xcurv[3])
        )
        xglob_est[0:3] = xglob[0:3]
        xglob_est[3] = xglob[3] + self.timestep * (xglob[2])
        xglob_est[4] = xglob[4] + self.timestep * (
            xglob[0] * np.cos(xglob[3]) - xglob[1] * np.sin(xglob[3])
        )
        xglob_est[4] = xglob[4] + self.timestep * (
            xglob[0] * np.sin(xglob[3]) + xglob[1] * np.cos(xglob[3])
        )

        return xcurv_est, xglob_est

    # get prediction for mpc-cbf controller
    def get_trajectory_nsteps(self, n):
        xcurv_nsteps = np.zeros((X_DIM, n))
        xglob_nsteps = np.zeros((X_DIM, n))
        for index in range(n):
            if index == 0:
                xcurv_est, xglob_est = self.get_estimation(self.xglob, self.xcurv)
            else:
                xcurv_est, xglob_est = self.get_estimation(
                    xglob_nsteps[:, index - 1], xcurv_nsteps[:, index - 1]
                )
            while xcurv_est[4] > self.lap_length:
                xcurv_est[4] = xcurv_est[4] - self.lap_length
            xcurv_nsteps[:, index] = xcurv_est
            xglob_nsteps[:, index] = xglob_est
        return xcurv_nsteps, xglob_nsteps


class NoDynamicsModel(base.NoDynamicsModel):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        base.NoDynamicsModel.__init__(self, name=name, param=param)


# off-board simulator
class CarRacingSim(base.CarRacingSim):
    def __init__(self):
        base.CarRacingSim.__init__(self)
        self.ax = None
        self.fig = None

    def add_vehicle(self, vehicle):
        self.vehicles[vehicle.name] = vehicle
        self.vehicles[vehicle.name].set_track(self.track)
        self.vehicles[vehicle.name].set_timestep(self.timestep)

    def sim(
        self,
        sim_time=50.0,
        one_lap=False,
        one_lap_name=None,
        animating_flag=False,
    ):
        if one_lap == True:
            current_lap = self.vehicles[one_lap_name].laps

        for i in range(0, int(sim_time / self.timestep)):
            for name in self.vehicles:
                # update system state
                self.vehicles[name].forward_one_step(self.vehicles[name].realtime_flag)

            if (one_lap == True) and (self.vehicles[one_lap_name].laps > current_lap):
                print("lap completed")
                break

    def plot_state(self, name):
        laps = self.vehicles[name].laps
        time = np.zeros(int(round(self.vehicles[name].time / self.timestep)) + 1)
        traj = np.zeros((int(round(self.vehicles[name].time / self.timestep)) + 1, X_DIM))
        counter = 0
        for i in range(0, laps):
            for j in range(
                0,
                int(
                    round(
                        (self.vehicles[name].times[i][-1] - self.vehicles[name].times[i][0])
                        / self.timestep
                    )
                ),
            ):
                time[counter] = self.vehicles[name].times[i][j]

                traj[counter, :] = self.vehicles[name].xcurvs[i][j][:]
                counter = counter + 1
        for i in range(
            0,
            int(
                round(
                    (self.vehicles[name].lap_times[-1] - self.vehicles[name].lap_times[0])
                    / self.timestep
                )
            )
            + 1,
        ):
            time[counter] = self.vehicles[name].lap_times[i]
            traj[counter, :] = self.vehicles[name].lap_xcurvs[i][:]
            counter = counter + 1
        fig, axs = plt.subplots(4)
        axs[0].plot(time, traj[:, 0], "-o", linewidth=1, markersize=1)
        axs[0].set_xlabel("time [s]", fontsize=14)
        axs[0].set_ylabel("$v_x$ [m/s]", fontsize=14)

        axs[1].plot(time, traj[:, 1], "-o", linewidth=1, markersize=1)
        axs[1].set_xlabel("time [s]", fontsize=14)
        axs[1].set_ylabel("$v_y$ [m/s]", fontsize=14)

        axs[2].plot(time, traj[:, 3], "-o", linewidth=1, markersize=1)
        axs[2].set_xlabel("time [s]", fontsize=14)
        axs[2].set_ylabel("$e_{\psi}$ [rad]", fontsize=14)

        axs[3].plot(time, traj[:, 5], "-o", linewidth=1, markersize=1)
        axs[3].set_xlabel("time [s]", fontsize=14)
        axs[3].set_ylabel("$e_y$ [m]", fontsize=14)
        plt.show()

    def plot_states(self):
        for name in self.vehicles:
            self.plot_state(name)
        plt.show()

    def plot_input(self, name):
        laps = self.vehicles[name].laps
        time = np.zeros(int(round(self.vehicles[name].time / self.timestep)))
        u = np.zeros((int(round(self.vehicles[name].time / self.timestep)), 2))
        counter = 0
        for i in range(0, laps):
            for j in range(
                0,
                int(
                    round(
                        (self.vehicles[name].times[i][-1] - self.vehicles[name].times[i][0])
                        / self.timestep
                    )
                ),
            ):
                time[counter] = self.vehicles[name].times[i][j]
                u[counter, :] = self.vehicles[name].inputs[i][j][:]
                counter = counter + 1
        for i in range(
            0,
            int(
                round(
                    (self.vehicles[name].lap_times[-1] - self.vehicles[name].lap_times[0])
                    / self.timestep
                )
            ),
        ):
            time[counter] = self.vehicles[name].lap_times[i]
            u[counter, :] = self.vehicles[name].lap_inputs[i][:]
            counter = counter + 1
        fig, axs = plt.subplots(2)
        axs[0].plot(time, u[:, 0], "-o", linewidth=1, markersize=1)
        axs[0].set_xlabel("time [s]", fontsize=14)
        axs[0].set_ylabel("$/delta$ [rad]", fontsize=14)
        axs[1].plot(time, u[:, 1], "-o", linewidth=1, markersize=1)
        axs[1].set_xlabel("time [s]", fontsize=14)
        axs[1].set_ylabel("$a$ [m/s^2]", fontsize=14)
        plt.show()

    def plot_inputs(self):
        for name in self.vehicles:
            self.plot_input(name)
        plt.show()

    def plot_simulation(self):
        fig, ax = plt.subplots()
        # plotting racing track
        self.track.plot_track(ax)
        # plot trajectories
        for name in self.vehicles:
            laps = self.vehicles[name].laps
            trajglob = np.zeros((int(round(self.vehicles[name].time / self.timestep)) + 1, X_DIM))
            counter = 0
            for i in range(0, laps):
                for j in range(
                    0,
                    int(
                        round(
                            (self.vehicles[name].times[i][-1] - self.vehicles[name].times[i][0])
                            / self.timestep
                        )
                    ),
                ):
                    trajglob[counter, :] = self.vehicles[name].xglobs[i][j][:]
                    counter = counter + 1
            for i in range(
                0,
                int(
                    round(
                        (self.vehicles[name].lap_times[-1] - self.vehicles[name].lap_times[0])
                        / self.timestep
                    )
                )
                + 1,
            ):
                trajglob[counter, :] = self.vehicles[name].lap_xglobs[i][:]
                counter = counter + 1
            ax.plot(trajglob[:, 4], trajglob[:, 5])
        plt.show()

    def animate(
        self, filename="untitled", ani_time=400, lap_number=None, racing_game=False, mpc_cbf=False
    ):
        num_veh = len(self.vehicles) - 1
        if racing_game:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_axes([0.05, 0.07, 0.56, 0.9])
            ax_1 = fig.add_axes([0.63, 0.07, 0.36, 0.9])
            ax_1.set_xticks([])
            ax_1.set_yticks([])
            self.track.plot_track(ax_1, center_line=False)
            patches_vehicles_1 = {}
            patches_vehicles_lmpc_prediction = []
            patches_vehicles_mpc_cbf_prediction = []
            (lmpc_prediciton_line,) = ax.plot([], [])
            (mpc_cbf_prediction_line,) = ax.plot([], [])
            vehicles_interest = []
            all_local_spline = []
            all_local_traj = []
            (local_line,) = ax_1.plot([], [])
            (local_spline,) = ax_1.plot([], [])
            for jj in range(num_veh + 1):
                (local_spline_1,) = ax_1.plot([], [])
                (local_traj_1,) = ax_1.plot([], [])
                all_local_spline.append(local_spline_1)
                all_local_traj.append(local_traj_1)
            horizon_planner = self.vehicles["ego"].ctrl_policy.racing_game_param.num_horizon_planner
            local_traj_xglob = np.zeros((ani_time, horizon_planner + 1, X_DIM))
            local_spline_xglob = np.zeros((ani_time, horizon_planner + 1, X_DIM))
            mpc_cbf_prediction = np.zeros((ani_time, 10 + 1, X_DIM))
            lmpc_prediction = np.zeros((ani_time, 12 + 1, X_DIM))
            all_local_traj_xglob = []
            all_local_spline_xglob = []
        else:
            fig, ax = plt.subplots()
        # plotting racing track
        self.track.plot_track(ax, center_line=False)
        # plot vehicles
        vertex_directions = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
        patches_vehicles = {}
        trajglobs = {}
        lap_number = self.vehicles["ego"].laps
        sim_time = (
            int(
                round(
                    (
                        self.vehicles["ego"].times[lap_number - 1][-1]
                        - self.vehicles["ego"].times[lap_number - 1][0]
                    )
                    / self.vehicles["ego"].timestep
                )
            )
            + 1
        )
        if ani_time > sim_time:
            ani_time = sim_time
        for name in self.vehicles:
            if name == "ego":
                face_color = "red"
            else:
                face_color = "blue"
            edge_color = "None"
            patches_vehicle = patches.Polygon(
                vertex_directions,
                alpha=1.0,
                closed=True,
                fc=face_color,
                ec="None",
                zorder=10,
                linewidth=2,
            )
            if racing_game:
                patches_vehicle_1 = patches.Polygon(
                    vertex_directions,
                    alpha=1.0,
                    closed=True,
                    fc=face_color,
                    ec="None",
                    zorder=10,
                    linewidth=2,
                )
                if name == "ego":
                    for jjjj in range(0, 6 + 1):
                        patch_lmpc = patches.Polygon(
                            vertex_directions,
                            alpha=1.0 - jjjj * 0.15,
                            closed=True,
                            fc="None",
                            zorder=10,
                            linewidth=2,
                        )
                        patches_vehicles_lmpc_prediction.append(patch_lmpc)
                        ax.add_patch(patches_vehicles_lmpc_prediction[jjjj])
                    for iiii in range(0, 5 + 1):
                        patch_mpc_cbf = patches.Polygon(
                            vertex_directions,
                            alpha=1.0 - iiii * 0.15,
                            closed=True,
                            fc="None",
                            zorder=10,
                            linewidth=2,
                        )
                        patches_vehicles_mpc_cbf_prediction.append(patch_mpc_cbf)
                        ax.add_patch(patches_vehicles_mpc_cbf_prediction[iiii])
            if name == "ego":
                if racing_game:
                    pass
                else:
                    ax.add_patch(patches_vehicle)
            else:
                ax.add_patch(patches_vehicle)
            if racing_game:
                ax_1.add_patch(patches_vehicle_1)
                ax_1.add_line(local_line)
                ax_1.add_line(local_spline)
                ax.add_line(lmpc_prediciton_line)
                ax.add_line(mpc_cbf_prediction_line)
                for jj in range(num_veh + 1):
                    ax_1.add_line(all_local_spline[jj])
                    ax_1.add_line(all_local_traj[jj])
                ax_1.axis("equal")
                patches_vehicles_1[name] = patches_vehicle_1
            ax.axis("equal")
            patches_vehicles[name] = patches_vehicle
            counter = 0
            trajglob = np.zeros((ani_time, X_DIM))
            for j in range(ani_time):
                trajglob[ani_time - 1 - counter, :] = self.vehicles[name].xglob_log[-1 - j][:]
                if racing_game:
                    if name == "ego":
                        if self.vehicles[name].local_trajs[-1 - j] is None:
                            local_traj_xglob[ani_time - 1 - counter, :, :] = np.zeros(
                                (horizon_planner + 1, X_DIM)
                            )
                            mpc_cbf_prediction[ani_time - 1 - counter, :, :] = np.zeros(
                                (10 + 1, X_DIM)
                            )
                            lmpc_prediction[ani_time - 1 - counter, :, :] = self.vehicles[
                                name
                            ].lmpc_prediction[-1 - j][:, :]
                        else:
                            local_traj_xglob[ani_time - 1 - counter, :, :] = self.vehicles[
                                name
                            ].local_trajs[-1 - j][:, :]
                            mpc_cbf_prediction[ani_time - 1 - counter, :, :] = self.vehicles[
                                name
                            ].mpc_cbf_prediction[-1 - j][:, :]
                            lmpc_prediction[ani_time - 1 - counter, :, :] = np.zeros(
                                (12 + 1, X_DIM)
                            )
                        if self.vehicles[name].vehicles_interest[-1 - j] is None:
                            vehicles_interest.insert(0, None)
                            all_local_traj_xglob.insert(0, None)
                            all_local_spline_xglob.insert(0, None)
                        else:
                            vehicles_interest.insert(
                                0,
                                self.vehicles[name].vehicles_interest[-1 - j],
                            )
                            all_local_traj_xglob.insert(
                                0, self.vehicles[name].all_local_trajs[-1 - j][:, :, :]
                            )
                            all_local_spline_xglob.insert(
                                0, self.vehicles[name].all_splines[-1 - j][:, :, :]
                            )
                        if self.vehicles[name].splines[-1 - j] is None:
                            local_spline_xglob[ani_time - 1 - counter, :, :] = np.zeros(
                                (horizon_planner + 1, X_DIM)
                            )
                        else:
                            local_spline_xglob[ani_time - 1 - counter, :, :] = self.vehicles[
                                name
                            ].splines[-1 - j][:, :]
                counter = counter + 1
            trajglobs[name] = trajglob

        def update(i):
            if racing_game:
                ax_1.set_xlim([trajglobs["ego"][i - 1, 4] - 2, trajglobs["ego"][i - 1, 4] + 2])
                ax_1.set_ylim([trajglobs["ego"][i - 1, 5] - 2, trajglobs["ego"][i - 1, 5] + 2])
            for name in patches_vehicles:
                x, y = trajglobs[name][i - 1, 4], trajglobs[name][i - 1, 5]
                psi = trajglobs[name][i - 1, 3]
                l = self.vehicles[name].param.length / 2
                w = self.vehicles[name].param.width / 2
                vertex_x = [
                    x + l * np.cos(psi) - w * np.sin(psi),
                    x + l * np.cos(psi) + w * np.sin(psi),
                    x - l * np.cos(psi) + w * np.sin(psi),
                    x - l * np.cos(psi) - w * np.sin(psi),
                ]
                vertex_y = [
                    y + l * np.sin(psi) + w * np.cos(psi),
                    y + l * np.sin(psi) - w * np.cos(psi),
                    y - l * np.sin(psi) - w * np.cos(psi),
                    y - l * np.sin(psi) + w * np.cos(psi),
                ]
                patches_vehicles[name].set_xy(np.array([vertex_x, vertex_y]).T)
                if racing_game:
                    patches_vehicles_1[name].set_xy(np.array([vertex_x, vertex_y]).T)
                    if name == "ego":
                        patches_vehicles[name].set_facecolor("None")
                        if mpc_cbf_prediction[i, :, :].all == 0:
                            for jjj in range(0, 5 + 1):
                                patches_vehicles_mpc_cbf_prediction[jjj].set_facecolor("None")
                        else:
                            for iii in range(0, 5 + 1):
                                x, y = (
                                    mpc_cbf_prediction[i - 1, iii * 2, 4],
                                    mpc_cbf_prediction[i - 1, iii * 2, 5],
                                )
                                if x == 0.0 and y == 0.0:
                                    patches_vehicles_mpc_cbf_prediction[iii].set_facecolor("None")
                                else:
                                    patches_vehicles_mpc_cbf_prediction[iii].set_facecolor("red")
                                psi = mpc_cbf_prediction[i - 1, iii, 3]
                                vertex_x = [
                                    x + l * np.cos(psi) - w * np.sin(psi),
                                    x + l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) - w * np.sin(psi),
                                ]
                                vertex_y = [
                                    y + l * np.sin(psi) + w * np.cos(psi),
                                    y + l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) + w * np.cos(psi),
                                ]
                                patches_vehicles_mpc_cbf_prediction[iii].set_xy(
                                    np.array([vertex_x, vertex_y]).T
                                )
                        if lmpc_prediction[i, :, :].all == 0:
                            for iii in range(0, 6 + 1):
                                patches_vehicles_lmpc_prediction[iii].set_facecolor("None")
                        else:
                            for jjj in range(0, 6 + 1):
                                x, y = (
                                    lmpc_prediction[i - 1, jjj * 2, 4],
                                    lmpc_prediction[i - 1, jjj * 2, 5],
                                )
                                if x == 0 and y == 0:
                                    patches_vehicles_lmpc_prediction[jjj].set_facecolor("None")
                                else:
                                    patches_vehicles_lmpc_prediction[jjj].set_facecolor("red")
                                psi = lmpc_prediction[i - 1, jjj, 3]
                                vertex_x = [
                                    x + l * np.cos(psi) - w * np.sin(psi),
                                    x + l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) + w * np.sin(psi),
                                    x - l * np.cos(psi) - w * np.sin(psi),
                                ]
                                vertex_y = [
                                    y + l * np.sin(psi) + w * np.cos(psi),
                                    y + l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) - w * np.cos(psi),
                                    y - l * np.sin(psi) + w * np.cos(psi),
                                ]
                                patches_vehicles_lmpc_prediction[jjj].set_xy(
                                    np.array([vertex_x, vertex_y]).T
                                )
                    # plot the local planned trajectory for ego vehicle if exists
                    if local_traj_xglob[i, :, :].all == 0:
                        local_line.set_data([], [])
                    else:

                        local_line.set_data(local_traj_xglob[i, :, 4], local_traj_xglob[i, :, 5])
                        local_line.set_color("orange")
                        local_line.set_linewidth(6)
                    if mpc_cbf_prediction[i, :, :].all == 0:
                        mpc_cbf_prediction_line.set_data([], [])
                    else:
                        mpc_cbf_prediction_line.set_data(
                            mpc_cbf_prediction[i - 1, :, 4], mpc_cbf_prediction[i - 1, :, 5]
                        )
                        mpc_cbf_prediction_line.set_color("slategray")
                        mpc_cbf_prediction_line.set_linewidth(2)
                    if lmpc_prediction[i, :, :].all == 0:
                        lmpc_prediciton_line.set_data([], [])
                    else:
                        lmpc_prediciton_line.set_data(
                            lmpc_prediction[i - 1, :, 4], lmpc_prediction[i - 1, :, 5]
                        )
                        lmpc_prediciton_line.set_color("purple")
                        lmpc_prediciton_line.set_linewidth(2)
                    if local_spline_xglob[i, :, :].all == 0:
                        local_spline.set_data([], [])
                    if vehicles_interest == []:
                        pass
                    else:
                        if vehicles_interest[i] is None:
                            if name == "ego":
                                patches_vehicles[name].set_facecolor("None")
                                patches_vehicles_1[name].set_facecolor("red")
                            else:
                                patches_vehicles[name].set_facecolor("blue")
                                patches_vehicles_1[name].set_facecolor("blue")
                            for jjj in range(num_veh + 1):
                                all_local_spline[jjj].set_data([], [])
                                all_local_traj[jjj].set_data([], [])
                            local_spline.set_data([], [])
                            local_line.set_data([], [])
                        else:
                            veh_of_interest = False
                            num_interest = len(vehicles_interest[i])
                            for ii in range(num_interest + 1):
                                if all_local_spline_xglob[i] is None:
                                    all_local_spline[ii].set_data([], [])
                                    all_local_traj[ii].set_data([], [])
                                else:
                                    all_local_spline[ii].set_data(
                                        all_local_spline_xglob[i][ii, :, 4],
                                        all_local_spline_xglob[i][ii, :, 5],
                                    )
                                    all_local_spline[ii].set_color("black")
                                    all_local_spline[ii].set_linestyle("-.")
                                    all_local_spline[ii].set_linewidth(1.5)
                                    all_local_traj[ii].set_data(
                                        all_local_traj_xglob[i][ii, :, 4],
                                        all_local_traj_xglob[i][ii, :, 5],
                                    )
                                    all_local_traj[ii].set_color("brown")
                                    all_local_traj[ii].set_linewidth(1.5)
                            if num_interest < num_veh:
                                delta_num = num_veh - num_interest
                                for iiii in range(0, delta_num):
                                    all_local_spline[num_veh - iiii].set_data([], [])
                                    all_local_traj[num_veh - iiii].set_data([], [])
                            for name_1 in list(vehicles_interest[i]):
                                if name == name_1:
                                    veh_of_interest = True
                            if veh_of_interest:
                                patches_vehicles[name].set_facecolor("green")
                                patches_vehicles_1[name].set_facecolor("green")
                            else:
                                if name == "ego":
                                    patches_vehicles[name].set_facecolor("None")
                                    patches_vehicles_1[name].set_facecolor("red")
                                else:
                                    patches_vehicles[name].set_facecolor("blue")
                                    patches_vehicles_1[name].set_facecolor("blue")

        media = anim.FuncAnimation(
            fig, update, frames=np.arange(0, trajglob.shape[0]), interval=100
        )
        if mpc_cbf:
            media.save(
                "media/animation/" + filename + ".gif",
                dpi=80,
                writer="imagemagick",
            )
        else:
            media.save(
                "media/animation/" + filename + ".gif",
                dpi=80,
                writer=animation.writers["ffmpeg"](fps=10),
            )
