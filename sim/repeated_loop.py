import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as anim
from utils import vehicle_dynamics, base


# repeated loop controller
class PIDTrackingRepeatedLoop(base.PIDTracking):
    def __init__(self, vt=0.6, eyt=0.0):
        base.PIDTracking.__init__(self, vt, eyt)


class MPCTrackingRepeatedLoop(base.MPCTracking):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        base.MPCTracking.__init__(
            self, matrix_A, matrix_B, matrix_Q, matrix_R, vt, eyt)


class MPCCBFRacingRepeatedLoop(base.MPCCBFRacing):
    def __init__(self, matrix_A, matrix_B, matrix_Q, matrix_R, vt=0.6, eyt=0.0):
        base.MPCCBFRacing.__init__(
            self, matrix_A, matrix_B, matrix_Q, matrix_R, vt, eyt)
        self.realtime_flag = False

class LMPCRacingRepeatedLoop(base.LMPCRacing):
    def __init__(self, num_ss_points, num_ss_it, N, matrix_Qslack, matrix_Q_LMPC, matrix_R_LMPC, matrix_dR_LMPC, xdim, udim, shift, timestep, laps, time_lmpc):
        base.LMPCRacing.__init__(self, num_ss_points, num_ss_it, N, matrix_Qslack, matrix_Q_LMPC, matrix_R_LMPC, matrix_dR_LMPC, xdim, udim, shift, timestep, laps, time_lmpc)


# repeated loop dynamic model
class DynamicBicycleModelRepeatedLoop(base.DynamicBicycleModel):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        base.DynamicBicycleModel.__init__(
            self, name=name, param=param)


class NoDynamicsModelRepeatedLoop(base.NoDynamicsModel):
    def __init__(self, name=None, param=None, xcurv=None, xglob=None):
        base.NoDynamicsModel.__init__(self, name=name, param=param)


# repeated loop simulator
class CarRacingSimRepeatedLoop(base.CarRacingSim):
    def __init__(self):
        base.CarRacingSim.__init__(self)

    def add_vehicle(self, vehicle):
        self.vehicles[vehicle.name] = vehicle
        self.vehicles[vehicle.name].set_track(self.track)
        self.vehicles[vehicle.name].set_timestep(self.timestep)

    def sim(self, sim_time=50.0, one_lap_flag=False, one_lap_name=None):
        if one_lap_flag ==True:
            current_lap = self.vehicles[one_lap_name].laps
        for i in range(0, int(sim_time / self.timestep)):
            for name in self.vehicles:
                # update system state
                self.vehicles[name].forward_one_step(self.vehicles[name].realtime_flag)
            if (one_lap_flag == True) and (self.vehicles[one_lap_name].laps > current_lap):
                print("lap completed")
                break

    def plot_state(self, name):
        laps = self.vehicles[name].laps
        time = np.zeros(int(round(self.vehicles[name].time/self.timestep))+1)
        traj = np.zeros((int(round(self.vehicles[name].time/self.timestep))+1,6))
        counter = 0
        for i in range(0,laps):
            for j in range(0, int(round((self.vehicles[name].time_list[i][-1]-self.vehicles[name].time_list[i][0])/self.timestep))):
                time[counter] = self.vehicles[name].time_list[i][j]
                
                traj[counter,:] = self.vehicles[name].xcurv_list[i][j][:]
                counter =  counter + 1
        for i in range(0,int(round((self.vehicles[name].traj_time[-1]-self.vehicles[name].traj_time[0])/self.timestep))+1):
            time[counter] = self.vehicles[name].traj_time[i]
            traj[counter,:] = self.vehicles[name].traj_xcurv[i][:]
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
    
    def plot_input(self,name):
        laps = self.vehicles[name].laps
        time = np.zeros(int(round(self.vehicles[name].time/self.timestep)))
        u = np.zeros((int(round(self.vehicles[name].time/self.timestep)),2))
        counter = 0
        for i in range(0,laps):
            for j in range(0, int(round((self.vehicles[name].time_list[i][-1]-self.vehicles[name].time_list[i][0])/self.timestep))):
                time[counter] = self.vehicles[name].time_list[i][j]
                u[counter,:] = self.vehicles[name].u_list[i][j][:]
                counter =  counter + 1
        for i in range(0,int(round((self.vehicles[name].traj_time[-1]-self.vehicles[name].traj_time[0])/self.timestep))):
            time[counter] = self.vehicles[name].traj_time[i]
            u[counter,:] = self.vehicles[name].traj_u[i][:]
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
        num_sampling_per_meter = 100
        num_track_points = int(
            np.floor(num_sampling_per_meter * self.track.lap_length))
        points_out = np.zeros((num_track_points, 2))
        points_center = np.zeros((num_track_points, 2))
        points_in = np.zeros((num_track_points, 2))
        for i in range(0, num_track_points):
            points_out[i, :] = self.track.get_global_position(
                i / float(num_sampling_per_meter), self.track.width)
            points_center[i, :] = self.track.get_global_position(
                i / float(num_sampling_per_meter), 0.0)
            points_in[i, :] = self.track.get_global_position(
                i / float(num_sampling_per_meter), -self.track.width)
        # ax.plot(self.track.point_and_tangent[:, 0], self.track.point_and_tangent[:, 1], "o") # plot joint point between segments
        ax.plot(points_center[:, 0], points_center[:, 1], "--r")
        ax.plot(points_in[:, 0], points_in[:, 1], "-b")
        ax.plot(points_out[:, 0], points_out[:, 1], "-b")
        # plot trajectories
        for name in self.vehicles:
            laps = self.vehicles[name].laps
            trajglob = np.zeros((int(round(self.vehicles[name].time/self.timestep))+1,6))
            counter = 0
            for i in range(0,laps):
                for j in range(0, int(round((self.vehicles[name].time_list[i][-1]-self.vehicles[name].time_list[i][0])/self.timestep))):
                    trajglob[counter,:] = self.vehicles[name].xglob_list[i][j][:]
                    counter =  counter + 1
            for i in range(0,int(round((self.vehicles[name].traj_time[-1]-self.vehicles[name].traj_time[0])/self.timestep))+1):
                trajglob[counter,:] = self.vehicles[name].traj_xglob[i][:]
                counter = counter + 1
            ax.plot(trajglob[:, 4], trajglob[:, 5])
        plt.show()

    def animate(self, filename="untitled", only_last_lap = False):
        fig, ax = plt.subplots()
        # plotting racing track
        num_sampling_per_meter = 100
        num_track_points = int(
            np.floor(num_sampling_per_meter * self.track.lap_length))
        points_out = np.zeros((num_track_points, 2))
        points_center = np.zeros((num_track_points, 2))
        points_in = np.zeros((num_track_points, 2))
        for i in range(0, num_track_points):
            points_out[i, :] = self.track.get_global_position(
                i / float(num_sampling_per_meter), self.track.width)
            points_center[i, :] = self.track.get_global_position(
                i / float(num_sampling_per_meter), 0.0)
            points_in[i, :] = self.track.get_global_position(
                i / float(num_sampling_per_meter), -self.track.width)
        # ax.plot(self.track.point_and_tangent[:, 0], self.track.point_and_tangent[:, 1], "o") # plot joint point between segments
        ax.plot(points_center[:, 0], points_center[:, 1], "--r")
        ax.plot(points_in[:, 0], points_in[:, 1], "-b")
        ax.plot(points_out[:, 0], points_out[:, 1], "-b")
        # plot vehicles
        vertex_directions = np.array(
            [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
        patches_vehicles = {}
        trajglobs = {}
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
            patches_vehicles[name] = patches_vehicle
            if only_last_lap:
                lap_number = self.vehicles[name].laps
                trajglob = np.zeros(( int(round((self.vehicles[name].time_list[lap_number-1][-1]-self.vehicles[name].time_list[lap_number-1][0])/self.vehicles[name].timestep))+1 ,6))
                counter = 0
                for j in range(int(round((self.vehicles[name].time_list[lap_number-1][-1]-self.vehicles[name].time_list[lap_number-1][0])/self.vehicles[name].timestep))+1):
                    trajglob[counter,:] = self.vehicles[name].xglob_list[lap_number-1][j][:]
                    counter = counter + 1
            else:
                laps = self.vehicles[name].laps
                trajglob = np.zeros((int(round(self.vehicles[name].time/self.timestep))+1,6))
                counter = 0
                for i in range(0,laps):
                    for j in range(0, int(round((self.vehicles[name].time_list[i][-1]-self.vehicles[name].time_list[i][0])/self.timestep))):
                        trajglob[counter,:] = self.vehicles[name].xglob_list[i][j][:]
                        counter =  counter + 1
                for i in range(0,int(round((self.vehicles[name].traj_time[-1]-self.vehicles[name].traj_time[0])/self.timestep))+1):
                    trajglob[counter,:] = self.vehicles[name].traj_xglob[i][:]
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

        media = anim.FuncAnimation(fig, update, frames=np.arange(
            0, trajglob.shape[0]), interval=100)
        media.save("media/animation/" + filename +".gif", dpi=80, writer="imagemagick")
