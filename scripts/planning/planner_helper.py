import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from scripts.utils.constants import *


def get_agent_range(s_agent, ey_agent, epsi_agent, length, width):
    ey_agent_max = (
        ey_agent
        + 0.5 * length * np.sin(epsi_agent)
        + 0.5 * width * np.cos(epsi_agent)
    )
    ey_agent_min = (
        ey_agent
        - 0.5 * length * np.sin(epsi_agent)
        - 0.5 * width * np.cos(epsi_agent)
    )
    s_agent_max = (
        s_agent
        + 0.5 * length * np.cos(epsi_agent)
        + 0.5 * width * np.sin(epsi_agent)
    )
    s_agent_min = (
        s_agent
        - 0.5 * length * np.cos(epsi_agent)
        - 0.5 * width * np.sin(epsi_agent)
    )
    return ey_agent_max, ey_agent_min, s_agent_max, s_agent_min


def ego_agent_overlap_checker(s_ego_min, s_ego_max, s_veh_min, s_veh_max, lap_length):
    overlap_flag = True
    if (
        (
            s_ego_max
            <= s_veh_min
            or s_ego_min
            >= s_veh_max
        )
        or (
            s_ego_max
            <= s_veh_min + lap_length
            or s_ego_min
            >= s_veh_max + lap_length
        )
        or (
            s_ego_max
            + lap_length
            <= s_veh_min
            or s_ego_min
            + lap_length
            >= s_veh_max
        )
    ):
        overlap_flag = False
    return overlap_flag


def get_bezier_control_points(vehicles_interest, veh_info_list, agent_info, racing_game_param, track, optimal_traj_xcurv, sorted_vehicles, xcurv_ego):
    num_veh = len(vehicles_interest)
    veh_length = vehicles_interest[(list(vehicles_interest)[0])].param.length
    veh_width = vehicles_interest[(list(vehicles_interest)[0])].param.width
    prediction_factor = racing_game_param.planning_prediction_factor
    safety_factor = racing_game_param.safety_factor
    bezier_control_point = np.zeros((num_veh + 1, racing_game_param.bezier_order + 1, 2)) # for each point, coordinate in (s, ey)
    func_optimal_ey = interp1d(optimal_traj_xcurv[:, 4], optimal_traj_xcurv[:, 5])
    for index in range(num_veh + 1):
        # s0
        bezier_control_point[index, 0, 0] = xcurv_ego[4] 
        # s3
        bezier_control_point[index, 3, 0] = xcurv_ego[4] +prediction_factor * agent_info.max_delta_v + 4
        # when the s3 is ahead start line, s0 is behind start line
        if bezier_control_point[index, 0, 0] > bezier_control_point[index, 3, 0]:
            # s1
            bezier_control_point[index, 1, 0] = (
                bezier_control_point[index, 3, 0]
                + track.lap_length
                - bezier_control_point[index, 0, 0]
            ) / 3.0 + bezier_control_point[index, 0, 0]
            # s2
            bezier_control_point[index, 2, 0] = (
                2.0
                * (
                    bezier_control_point[index, 3, 0]
                    + track.lap_length
                    - bezier_control_point[index, 0, 0]
                )
                / 3.0
                + bezier_control_point[index, 0, 0]
            )
            # new s3 value, add lap length
            bezier_control_point[index, 3, 0] = (
                bezier_control_point[index, 3, 0] + track.lap_length
            )
        # when s0 and s3 is in the same side of start line
        else:
            # s1
            bezier_control_point[index, 1, 0] = (
                bezier_control_point[index, 3, 0]
                - bezier_control_point[index, 0, 0]
            ) / 3.0 + bezier_control_point[index, 0, 0]
            # s2
            bezier_control_point[index, 2, 0] = (
                2.0
                * (
                    bezier_control_point[index, 3, 0]
                    - bezier_control_point[index, 0, 0]
                )
                / 3.0
                + bezier_control_point[index, 0, 0]
            )
        # ey0
        if bezier_control_point[index, 0, 0] < 0:
            bezier_control_point[index, 0, 1] = func_optimal_ey(
                bezier_control_point[index, 0, 0] + track.lap_length
            )
        elif bezier_control_point[index, 0, 0] < optimal_traj_xcurv[0, 4]:
            bezier_control_point[index, 0, 1] = optimal_traj_xcurv[0, 5]
        else:
            bezier_control_point[index, 0, 1] = func_optimal_ey(
                bezier_control_point[index, 0, 0]
            )
        bezier_control_point[index, 0, 1] = xcurv_ego[5]
        # ey1 and ey2
        # the first curve
        if index == 0:
            bezier_control_point[index, 1, 1] = 0.8*track.width-( - veh_info_list[index, 1] - 0.5 * veh_width)*0.2
            bezier_control_point[index, 2, 1] = 0.8*track.width-( - veh_info_list[index, 1] - 0.5 * veh_width)*0.2
        # the last curve
        elif index == num_veh:
            bezier_control_point[index, 1, 1] = -0.8*track.width+((veh_info_list[index-1, 1] - 0.5 * veh_width))*0.2
            bezier_control_point[index, 2, 1] = -0.8*track.width+((veh_info_list[index-1, 1] - 0.5 * veh_width))*0.2
        else:
            bezier_control_point[index, 1, 1] = 0.7*(veh_info_list[index, 1] + 0.5 * veh_width)+0.3*(veh_info_list[index-1, 1] - 0.5 * veh_width)
            bezier_control_point[index, 2, 1] = 0.7*(veh_info_list[index, 1] + 0.5 * veh_width)+0.3*(veh_info_list[index-1, 1] - 0.5 * veh_width)
        # ey3
        if bezier_control_point[index, 3, 0] >= track.lap_length:
            if (
                bezier_control_point[index, 3, 0] - track.lap_length
                <= optimal_traj_xcurv[0, 4]
            ):
                bezier_control_point[index, 3, 1] = optimal_traj_xcurv[0, 5]
            else:
                bezier_control_point[index, 3, 1] = func_optimal_ey(
                    bezier_control_point[index, 3, 0] - track.lap_length
                )
        else:
            if bezier_control_point[index, 3, 0] <= optimal_traj_xcurv[0, 4]:
                bezier_control_point[index, 3, 1] = optimal_traj_xcurv[0, 5]
            else:
                bezier_control_point[index, 3, 1] = func_optimal_ey(
                    bezier_control_point[index, 3, 0]
                )
    return bezier_control_point


def get_bezier_curve(bezier_control_point, t):
    s0, s1, s2, s3 = bezier_control_point[:, 0]
    ey0, ey1, ey2, ey3 = bezier_control_point[:, 1]
    bezier_curve_s = (
        s0 * ((1 - t) ** 3)
        + 3 * s1 * t * ((1 - t) ** 2)
        + 3 * s2 * (t ** 2) * (1 - t)
        + s3 * (t ** 3)
    )
    bezier_curve_ey = (
        ey0 * ((1 - t) ** 3)
        + 3 * ey1 * t * ((1 - t) ** 2)
        + 3 * ey2 * (t ** 2) * (1 - t)
        + ey3 * (t ** 3)
    )
    return [bezier_curve_s, bezier_curve_ey]


# used to plot the states at each time step during debugging
def debug_plot(track, vehicles, target_traj_xglob):
    fig, ax = plt.subplots()
    track.plot_track(ax)
    x_ego, y_ego, dx_ego, dy_ego, alpha_ego = vehicles["ego"].get_vehicle_in_rectangle(vehicles["ego"].xglob)
    ax.add_patch(
        patches.Rectangle((x_ego, y_ego), dx_ego, dy_ego, alpha_ego, color="red")
    )
    for name in list(vehicles):
        if name == "ego":
            pass
        else:
            x_car, y_car, dx_car, dy_car, alpha_car = vehicles[name].get_vehicle_in_rectangle(
                vehicles[name].xglob
            )
            ax.add_patch(
                patches.Rectangle(
                    (x_car, y_car), dx_car, dy_car, alpha_car, color="blue"
                )
            )
    ax.plot(target_traj_xglob[:, 4], target_traj_xglob[:, 5])
    ax.axis("equal")
    plt.show()


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
        if vehicles[name].xcurv[4]<=next_lap_range:
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
            (
                target_traj_xglob[index, 4],
                target_traj_xglob[index, 5],
            ) = track.get_global_position(
                s_i,
                traj_xcurv[index, 5],
            )
    return target_traj_xglob

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
                <= racing_game_param.safety_factor
                * ego.param.length
                + racing_game_param.planning_prediction_factor
                * delta_v
            )
            and (s_agent >= s_ego)
        )
        or (
            # agent is in next lap, agent is in front of the ego
            (
                s_agent + lap_length - s_ego
                <= racing_game_param.safety_factor
                * ego.param.length
                + racing_game_param.planning_prediction_factor
                * delta_v
            )
            and (s_agent + lap_length >= s_ego)
        )
        or (
            # agent and ego in same lap, ego is in front of the agent
            (
                -s_agent + s_ego
                <= 1.0
                * ego.param.length
                + 0*racing_game_param.planning_prediction_factor
                * delta_v
            )
            and (s_agent <= s_ego)
        )
        or (
            # ego is in next lap, ego is in front of the agent
            (
                -s_agent + s_ego + lap_length
                <= 1.0
                * ego.param.length
                + 0*racing_game_param.planning_prediction_factor
                * delta_v
            )
            and (s_agent <= s_ego + lap_length)
        )
    ): 
        vehicle_interest = True
    return vehicle_interest
    

class AgentInfo:
    def __init__(self):
        self.max_delta_v = None
        self.min_delta_v = None
        self.max_s = None
        self.min_s = None
        self.max_vx = None
        self.min_vx = None
        