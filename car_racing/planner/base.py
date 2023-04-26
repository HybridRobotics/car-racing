import copy

import numpy as np

from racing_env import X_DIM

class AgentInfo:
    def __init__(self):
        self.max_delta_v = None
        self.min_delta_v = None
        self.max_s = None
        self.min_s = None
        self.max_vx = None
        self.min_vx = None

class PlannerBase:
    def __init__(self):
        self.agent_name = None
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.xglob = None
        self.u = None
        self.realtime_flag = False
        # store the information (e.g. states, inputs) of current lap
        self.lap_times, self.lap_xcurvs, self.lap_xglobs, self.lap_inputs = [], [], [], []
        self.lap_times.append(self.time)
        # store the information (e.g. state, inputs) of the whole simulation
        self.times, self.xglobs, self.xcurvs, self.inputs = [], [], [], []
        self.laps = 0
        self.track = None
        self.opti_traj_xcurv = None
        self.opti_traj_xglob = None

    def set_track(self, track):
        self.track = track
        self.lap_length = track.lap_length
        self.point_and_tangent = track.point_and_tangent
        self.lap_width = track.width

    def set_opti_traj(self, opti_traj_xcurv, opti_traj_xglob):
        self.opti_traj_xcurv = opti_traj_xcurv
        self.opti_traj_xglob = opti_traj_xglob

    def set_racing_sim(self, racing_sim):
        self.racing_sim = racing_sim

    def set_timestep(self, timestep):
        self.timestep = timestep

    def set_target_speed(self, vt):
        self.vt = vt

    def set_target_deviation(self, eyt):
        self.eyt = eyt

    def set_state(self, xcurv, xglob):
        self.x = xcurv
        self.xglob = xglob

    def calc_input(self):
        pass

    def get_input(self):
        return self.u

    def update_memory(self, current_lap):
        xcurv = copy.deepcopy(self.x)
        xglob = copy.deepcopy(self.xglob)
        time = copy.deepcopy(self.time)
        if xcurv[4] > self.lap_length * (current_lap + 1):
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            xcurv[4] = xcurv[4] - current_lap * self.lap_length
            self.lap_xcurvs.append(xcurv)
            self.lap_inputs.append(self.u)
            self.xglobs.append(self.lap_xglobs)
            self.times.append(self.lap_times)
            self.xcurvs.append(self.lap_xcurvs)
            self.inputs.append(self.lap_inputs)
            x = copy.deepcopy(self.x)
            x[4] = x[4] - self.lap_length * (current_lap + 1)
            self.laps = self.laps + 1
            self.lap_xglobs, self.lap_xcurvs, self.lap_inputs, self.lap_times = [], [], [], []
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            self.lap_xcurvs.append(x)
        else:
            xcurv[4] = xcurv[4] - current_lap * self.lap_length
            self.lap_xglobs.append(xglob)
            self.lap_times.append(time)
            self.lap_xcurvs.append(xcurv)
            self.lap_inputs.append(self.u)

def get_agent_range(s_agent, ey_agent, epsi_agent, length, width):
    ey_agent_max = ey_agent + 0.5 * length * np.sin(epsi_agent) + 0.5 * width * np.cos(epsi_agent)
    ey_agent_min = ey_agent - 0.5 * length * np.sin(epsi_agent) - 0.5 * width * np.cos(epsi_agent)
    s_agent_max = s_agent + 0.5 * length * np.cos(epsi_agent) + 0.5 * width * np.sin(epsi_agent)
    s_agent_min = s_agent - 0.5 * length * np.cos(epsi_agent) - 0.5 * width * np.sin(epsi_agent)
    return ey_agent_max, ey_agent_min, s_agent_max, s_agent_min


def ego_agent_overlap_checker(s_ego_min, s_ego_max, s_veh_min, s_veh_max, lap_length):
    overlap_flag = True
    if (
        (s_ego_max <= s_veh_min or s_ego_min >= s_veh_max)
        or (s_ego_max <= s_veh_min + lap_length or s_ego_min >= s_veh_max + lap_length)
        or (s_ego_max + lap_length <= s_veh_min or s_ego_min + lap_length >= s_veh_max)
    ):
        overlap_flag = False
    return overlap_flag

def check_ego_agent_distance(ego, agent, racing_game_param, lap_length):
    vehicle_interest = False
    delta_v = abs(ego.xcurv[0] - agent.xcurv[0])
    s_agent = copy.deepcopy(agent.xcurv[4])
    s_ego = copy.deepcopy(ego.xcurv[4])
    while s_agent > lap_length:
        s_agent = s_agent - lap_length
    while s_ego > lap_length:
        s_ego = s_ego - lap_length
    if (
        # agent and ego in same lap, agent is in front of the ego
        (
            (
                s_agent - s_ego
                <= racing_game_param.safety_factor * ego.param.length
                + racing_game_param.planning_prediction_factor * delta_v
            )
            and (s_agent >= s_ego)
        )
        or (
            # agent is in next lap, agent is in front of the ego
            (
                s_agent + lap_length - s_ego
                <= racing_game_param.safety_factor * ego.param.length
                + racing_game_param.planning_prediction_factor * delta_v
            )
            and (s_agent + lap_length >= s_ego)
        )
        or (
            # agent and ego in same lap, ego is in front of the agent
            (
                -s_agent + s_ego
                <= 1.0 * ego.param.length
                + 0 * racing_game_param.planning_prediction_factor * delta_v
            )
            and (s_agent <= s_ego)
        )
        or (
            # ego is in next lap, ego is in front of the agent
            (
                -s_agent + s_ego + lap_length
                <= 1.0 * ego.param.length
                + 0 * racing_game_param.planning_prediction_factor * delta_v
            )
            and (s_agent <= s_ego + lap_length)
        )
    ):
        vehicle_interest = True
    return vehicle_interest

def get_agent_info(vehicles, sorted_vehicles, track):
    next_lap_range = 20
    agent_info = AgentInfo()
    delta_vs = np.zeros(len(sorted_vehicles))
    agent_vxs = np.zeros(len(sorted_vehicles))
    curv_distances = np.zeros(len(sorted_vehicles))
    for index in range(len(sorted_vehicles)):
        name = sorted_vehicles[index]
        delta_vs[index] = abs(vehicles["ego"].xcurv[0] - vehicles[name].xcurv[0])
        agent_vxs[index] = vehicles[name].xcurv[0]
        if vehicles[name].xcurv[4] <= next_lap_range:
            curv_distances[index] = vehicles[name].xcurv[4] + track.lap_length
        else:
            curv_distances[index] = vehicles[name].xcurv[4]
    agent_info.min_vx = min(agent_vxs)
    agent_info.max_vx = max(agent_vxs)
    agent_info.min_delta_v = min(delta_vs)
    agent_info.max_delta_v = max(delta_vs)
    agent_info.min_s = min(curv_distances)
    agent_info.max_s = max(curv_distances)
    while agent_info.min_s > track.lap_length:
        agent_info.min_s = agent_info.min_s - track.lap_length
    while agent_info.max_s > track.lap_length:
        agent_info.max_s = agent_info.max_s - track.lap_length
    return agent_info


def get_traj_xglob(traj_xcurv, track):
    traj_len = np.size(traj_xcurv, 0)
    target_traj_xglob = np.zeros((traj_len, X_DIM))
    for index in range(traj_len):
        s_i = copy.deepcopy(traj_xcurv[index, 4])
        while s_i > track.lap_length:
            s_i = s_i - track.lap_length
        (target_traj_xglob[index, 4], target_traj_xglob[index, 5],) = track.get_global_position(
            s_i,
            traj_xcurv[index, 5],
        )
    return target_traj_xglob

