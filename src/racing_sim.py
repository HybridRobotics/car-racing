import numpy as np
import matplotlib.pyplot as plt
import racing_env
import racing_car


class CarRacingSim:
    def __init__(self):
        self.racing_track = None
        self.vehicles = {}
        self.closedloop_x = {}
        self.closedloop_x_glob = {}
        self.closedloop_u = {}

    def set_timestep(self, dt):
        self.dt = dt

    def set_track(self, track):
        self.racing_track = track

    def add_vehicle(self, vehicle):
        self.vehicles[vehicle.name] = vehicle
        self.vehicles[vehicle.name].set_track(self.racing_track)
        self.vehicles[vehicle.name].set_timestep(self.dt)

    def setup(self):
        # setup simulation
        for name in self.vehicles:
            self.closedloop_x[name] = []
            self.closedloop_x_glob[name] = []
            self.closedloop_u[name] = []
            self.closedloop_x[name].append(self.vehicles[name].x)
            self.closedloop_x_glob[name].append(self.vehicles[name].x_glob)

    def sim(self, sim_time=10.0):
        for i in range(0, int(sim_time / self.dt)):
            for name in self.vehicles:
                # update system state
                if self.vehicles[name].model_type == "no-policy":
                    self.vehicles[name].forward_dynamics()
                else:
                    self.vehicles[name].calc_ctrl_input()
                    self.vehicles[name].forward_dynamics()
                # collect trajectory
                self.closedloop_x[name].append(self.vehicles[name].x)
                self.closedloop_x_glob[name].append(self.vehicles[name].x_glob)
                self.closedloop_u[name].append(self.vehicles[name].u)
        self.plot_simulation()

    def plot_trajectories(self, ax=None):
        for name in self.vehicles:
            traj = np.stack(self.closedloop_x_glob[name], axis=0)
            ax.plot(traj[:, 4], traj[:, 5])

    def plot_track(self, ax):
        num_sampling_per_meter = 100
        num_track_points = int(np.floor(num_sampling_per_meter * self.racing_track.track_length))
        points_out = np.zeros((num_track_points, 2))
        points_center = np.zeros((num_track_points, 2))
        points_in = np.zeros((num_track_points, 2))
        for i in range(0, num_track_points):
            points_out[i, :] = self.racing_track.get_global_position(
                i / float(num_sampling_per_meter), self.racing_track.width
            )
            points_center[i, :] = self.racing_track.get_global_position(i / float(num_sampling_per_meter), 0.0)
            points_in[i, :] = self.racing_track.get_global_position(
                i / float(num_sampling_per_meter), -self.racing_track.width
            )
        # ax.plot(self.racing_track.point_and_tangent[:, 0], self.racing_track.point_and_tangent[:, 1], "o") # plot joint point between segments
        ax.plot(points_center[:, 0], points_center[:, 1], "--r")
        ax.plot(points_in[:, 0], points_in[:, 1], "-b")
        ax.plot(points_out[:, 0], points_out[:, 1], "-b")

    def plot_simulation(self):
        fig, ax = plt.subplots()
        self.plot_track(ax=ax)
        self.plot_trajectories(ax=ax)
        plt.show()
