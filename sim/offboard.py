import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from utils import vehicle_dynamics, base, racing_env
from matplotlib import animation
from utils.constants import *

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
        base.LMPCRacingGame.__init__(
            self, lmpc_param, racing_game_param=racing_game_param
        )
        self.realt = False


# off-board dynamic model
class DynamicBicycleModel(base.DynamicBicycleModel):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        base.DynamicBicycleModel.__init__(self, name=name, param=param)

    # in this estimation, the vehicles is assumed to move with input is equal to zero
    def get_estimation(self, xglob, xcurv):
        curv = racing_env.get_curvature(
            self.lap_length, self.point_and_tangent, xcurv[4]
        )
        xcurv_est = np.zeros((X_DIM, ))
        xglob_est = np.zeros((X_DIM, ))
        xcurv_est[0:3] = xcurv[0:3]
        xcurv_est[3] = xcurv[3] + self.timestep * (
            xcurv[2]
            - (xcurv[0] * np.cos(xcurv[3]) - xcurv[1] * np.sin(xcurv[3]))
            / (1 - curv * xcurv[5])
            * curv
        )
        xcurv_est[4] = xcurv[4] + self.timestep * (
            (xcurv[0] * np.cos(xcurv[3]) - xcurv[1] * np.sin(xcurv[3]))
            / (1 - curv * xcurv[5])
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
                        (
                            self.vehicles[name].times[i][-1]
                            - self.vehicles[name].times[i][0]
                        )
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
                    (
                        self.vehicles[name].lap_times[-1]
                        - self.vehicles[name].lap_times[0]
                    )
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
                        (
                            self.vehicles[name].times[i][-1]
                            - self.vehicles[name].times[i][0]
                        )
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
                    (
                        self.vehicles[name].lap_times[-1]
                        - self.vehicles[name].lap_times[0]
                    )
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
            trajglob = np.zeros(
                (int(round(self.vehicles[name].time / self.timestep)) + 1, X_DIM)
            )
            counter = 0
            for i in range(0, laps):
                for j in range(
                    0,
                    int(
                        round(
                            (
                                self.vehicles[name].times[i][-1]
                                - self.vehicles[name].times[i][0]
                            )
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
                        (
                            self.vehicles[name].lap_times[-1]
                            - self.vehicles[name].lap_times[0]
                        )
                        / self.timestep
                    )
                )
                + 1,
            ):
                trajglob[counter, :] = self.vehicles[name].lap_xglobs[i][:]
                counter = counter + 1
            ax.plot(trajglob[:, 4], trajglob[:, 5])
        plt.show()

    def animate(self, filename="untitled", only_last_lap=False, lap_number=None):
        fig, ax = plt.subplots()
        # plotting racing track
        self.track.plot_track(ax)
        if self.opti_traj_xglob is None:
            pass
        else:
            ax.plot(self.opti_traj_xglob[:, 4], self.opti_traj_xglob[:, 5], "--g")
        # plot vehicles
        vertex_directions = np.array(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]]
        )
        patches_vehicles = {}
        trajglobs = {}
        (local_line,) = ax.plot([], [])
        (local_spline,) = ax.plot([], [])
        vehicles_interest = []
        if only_last_lap:
            ani_time = 600
        else:
            lap_number = self.vehicles["ego"].laps
            ani_time = int(round((self.vehicles["ego"].times[lap_number-1][-1]-self.vehicles["ego"].times[lap_number-1][0])/self.vehicles["ego"].timestep))+1
        horizon_planner = self.vehicles["ego"].ctrl_policy.racing_game_param.num_horizon_planner
        local_traj_xglob = np.zeros((ani_time, horizon_planner+1, X_DIM))
        local_spline_xglob = np.zeros((ani_time, horizon_planner+1, X_DIM))
        for name in self.vehicles:
            patches_vehicle = patches.Polygon(
                vertex_directions,
                alpha=1.0,
                closed=True,
                fc=self.vehicles[name].param.facecolor,
                ec=self.vehicles[name].param.edgecolor,
                zorder=10,
                linewidth=2,
            )
            ax.add_patch(patches_vehicle)
            ax.add_line(local_line)
            ax.add_line(local_spline)
            ax.axis("equal")
            patches_vehicles[name] = patches_vehicle
            if only_last_lap:
                counter = 0
                trajglob = np.zeros((ani_time, X_DIM))
                for j in range(ani_time):
                    trajglob[ani_time - 1 - counter, :] = self.vehicles[name].xglob_log[
                        -1 - j
                    ][:]
                    if name == "ego":
                        if self.vehicles[name].local_trajs[-1 - j] is None:
                            local_traj_xglob[ani_time - 1 - counter, :, :] = np.zeros(
                                (horizon_planner+1, X_DIM)
                            )
                        else:
                            local_traj_xglob[
                                ani_time - 1 - counter, :, :
                            ] = self.vehicles[name].local_trajs[-1 - j][:, :]
                        if self.vehicles[name].vehicles_interest[-1 - j] is None:
                            vehicles_interest.insert(0, None)
                        else:
                            vehicles_interest.insert(
                                0,
                                self.vehicles[name].vehicles_interest[-1 - j],
                            )
                        if self.vehicles[name].splines[-1 - j] is None:
                            local_spline_xglob[ani_time - 1 - counter, :, :] = np.zeros(
                                (horizon_planner+1, X_DIM)
                            )
                        else:
                            local_spline_xglob[
                                ani_time - 1 - counter, :, :
                            ] = self.vehicles[name].splines[-1 - j][:, :]
                    counter = counter + 1
            else:
                local_traj_xglob = np.zeros((int(round(self.vehicles[name].time / self.timestep)) + 1, horizon_planner+1, X_DIM))
                local_spline_xglob = np.zeros((int(round(self.vehicles[name].time / self.timestep)) + 1, horizon_planner+1, X_DIM))
                laps = self.vehicles[name].laps
                trajglob = np.zeros(
                    (
                        int(round(self.vehicles[name].time / self.timestep)) + 1,
                        X_DIM,
                    )
                )
                counter = 0
                for i in range(0, laps):
                    for j in range(
                        0,
                        int(
                            round(
                                (
                                    self.vehicles[name].times[i][-1]
                                    - self.vehicles[name].times[i][0]
                                )
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
                            (
                                self.vehicles[name].lap_times[-1]
                                - self.vehicles[name].lap_times[0]
                            )
                            / self.timestep
                        )
                    )
                    + 1,
                ):
                    trajglob[counter, :] = self.vehicles[name].lap_xglobs[i][:]
                    counter = counter + 1
            trajglobs[name] = trajglob
        # update vehicles for animation

        def update(i):
            for name in patches_vehicles:
                x, y = trajglobs[name][i, 4], trajglobs[name][i, 5]
                psi = trajglobs[name][i, 3]
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
                # plot the local planned trajectory for ego vehicle if exists
                if local_traj_xglob[i, :, :].all == 0:
                    pass
                else:
                    local_line.set_data(
                        local_traj_xglob[i, :, 4], local_traj_xglob[i, :, 5]
                    )
                if local_spline_xglob[i, :, :].all == 0:
                    pass
                else:
                    local_spline.set_data(
                        local_spline_xglob[i, :, 4],
                        local_spline_xglob[i, :, 5],
                    )
                    local_spline.set_color("black")
                if vehicles_interest == []:
                    pass
                else:
                    if vehicles_interest[i] is None:
                        patches_vehicles[name].set_edgecolor(
                            self.vehicles[name].param.edgecolor
                        )
                    else:
                        veh_of_interest = False
                        for name_1 in list(vehicles_interest[i]):
                            if name == name_1:
                                veh_of_interest = True
                            else:
                                pass
                        if veh_of_interest:
                            patches_vehicles[name].set_edgecolor("red")
                        else:
                            patches_vehicles[name].set_edgecolor(
                                self.vehicles[name].param.edgecolor
                            )

        media = anim.FuncAnimation(
            fig, update, frames=np.arange(0, trajglob.shape[0]), interval=100
        )
        media.save(
            "media/animation/" + filename + ".gif",
            dpi=80,
            writer="imagemagick",
        )
