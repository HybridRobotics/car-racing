import numpy as np


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

