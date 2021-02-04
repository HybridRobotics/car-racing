import numpy as np
import matplotlib.pyplot as plt
import racing_env
import racing_car


class CarRacingSim:
    def __init__(self):
        self.track = None
        self.vehicles = {}

    def set_timestep(self, dt):
        self.timestep = dt

    def set_track(self, track):
        self.track = track

    def add_vehicle(self, vehicle):
        self.vehicles[vehicle.name] = vehicle
        self.vehicles[vehicle.name].set_track(self.track)
        self.vehicles[vehicle.name].set_timestep(self.timestep)

    def sim(self, sim_time=50.0):
        for i in range(0, int(sim_time / self.timestep)):
            for name in self.vehicles:
                # update system state
                self.vehicles[name].forward_one_step()
        self.plot_simulation()
        self.plot_states()

    def plot_state(self, name):
        time = np.stack(self.vehicles[name].closedloop_time, axis=0)
        traj = np.stack(self.vehicles[name].closedloop_x, axis=0)
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

    def plot_states(self):
        for name in self.vehicles:
            self.plot_state(name)
        plt.show()

    def plot_simulation(self):
        fig, ax = plt.subplots()
        # plotting racing track
        num_sampling_per_meter = 100
        num_track_points = int(np.floor(num_sampling_per_meter * self.track.lap_length))
        points_out = np.zeros((num_track_points, 2))
        points_center = np.zeros((num_track_points, 2))
        points_in = np.zeros((num_track_points, 2))
        for i in range(0, num_track_points):
            points_out[i, :] = self.track.get_global_position(i / float(num_sampling_per_meter), self.track.width)
            points_center[i, :] = self.track.get_global_position(i / float(num_sampling_per_meter), 0.0)
            points_in[i, :] = self.track.get_global_position(i / float(num_sampling_per_meter), -self.track.width)
        # ax.plot(self.track.point_and_tangent[:, 0], self.track.point_and_tangent[:, 1], "o") # plot joint point between segments
        ax.plot(points_center[:, 0], points_center[:, 1], "--r")
        ax.plot(points_in[:, 0], points_in[:, 1], "-b")
        ax.plot(points_out[:, 0], points_out[:, 1], "-b")
        # plot trajectories
        for name in self.vehicles:
            traj = np.stack(self.vehicles[name].closedloop_xglob, axis=0)
            ax.plot(traj[:, 4], traj[:, 5])
        plt.show()
