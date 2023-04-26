import copy
from typing import Dict, Tuple, Union, List

import numpy as np

from racing_env import X_DIM, ClosedTrack, RacingEnv, ModelBase

class AgentInfo:
    """ Bounding of an agent, i.e., vehicle's information,
    which will be used to construct "safe" overtaking paths. 
    """
    def __init__(self):
        self.max_delta_v: Union[float, None] = None
        self.min_delta_v: Union[float, None] = None
        self.max_s: Union[float, None]       = None
        self.min_s: Union[float, None]       = None
        self.max_vx: Union[float, None]      = None
        self.min_vx: Union[float, None]      = None

    @classmethod
    def get_agent_info(cls, vehicles: Dict[str, ModelBase], sorted_vehicles: List[ModelBase], track: ClosedTrack) -> "AgentInfo":
        """ Construct the AgentInfo
        """
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

class RacingGameParam:
    """Collection of racing-behaviour-relevant parameters, tuning which will impact the
    planner's behaviours, like planning horizon"""
    def __init__(
        self,
        matrix_A=np.genfromtxt("data/sys/LTI/matrix_A.csv", delimiter=","),
        matrix_B=np.genfromtxt("data/sys/LTI/matrix_B.csv", delimiter=","),
        matrix_Q=np.diag([10.0, 0.0, 0.0, 5.0, 0.0, 50.0]),
        matrix_R=np.diag([0.1, 0.1]),
        matrix_R_planner=1 * np.diag([5, 0.10]),
        matrix_dR_planner=5 * np.diag([1.8, 0.0]),
        bezier_order=3,
        safety_factor=4.5,
        num_horizon_ctrl=10,
        num_horizon_planner=10,
        planning_prediction_factor=0.5,  # 2.0,
        alpha=0.98,
        timestep=None,
    ):
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.matrix_R_planner = matrix_R_planner
        self.matrix_dR_planner = matrix_dR_planner
        self.num_horizon_ctrl = num_horizon_ctrl
        self.num_horizon_planner = num_horizon_planner
        self.planning_prediction_factor = planning_prediction_factor
        self.alpha = alpha
        self.timestep = timestep
        self.bezier_order = bezier_order
        self.safety_factor = safety_factor

def get_agent_range(s_agent, ey_agent, epsi_agent, length, width) -> Tuple[float, float, float, float]:
    """Compute the area an vehicle may occupy (the other vehicle shall not enter!)

    Params
    ------
    s_agent, ey_agent: the local position of the vehicle
    epsi_agent: the direction of the vehicle
    length, width: the shape parameter of the vehicle

    Returns
    -------
    The area occupied by the vehicle, as a rectangle. 
    """
    ey_agent_max = ey_agent + 0.5 * length * np.sin(epsi_agent) + 0.5 * width * np.cos(epsi_agent)
    ey_agent_min = ey_agent - 0.5 * length * np.sin(epsi_agent) - 0.5 * width * np.cos(epsi_agent)
    s_agent_max = s_agent + 0.5 * length * np.cos(epsi_agent) + 0.5 * width * np.sin(epsi_agent)
    s_agent_min = s_agent - 0.5 * length * np.cos(epsi_agent) - 0.5 * width * np.sin(epsi_agent)
    return ey_agent_max, ey_agent_min, s_agent_max, s_agent_min

def ego_agent_overlap_checker(s_ego_min, s_ego_max, s_veh_min, s_veh_max, lap_length) -> bool:
    """Helper function to determine whether two vehicles overlap each other. 

    Here, overlapping means the two vehicles have the same projection onto the track
    centre line, not necessarily crashing. 

    Params
    ------
    s_ego_min, s_ego_max, s_veh_min, s_veh_max: specify the rectangular areas occupied
        by the ego and another vehicle, respecitively, from `get_agent_range`. 
    lap_length: the lap length

    Returns
    -------
    Boolean, true if overlapping
    """
    overlap_flag = True
    if (
        (s_ego_max <= s_veh_min or s_ego_min >= s_veh_max)
        or (s_ego_max <= s_veh_min + lap_length or s_ego_min >= s_veh_max + lap_length)
        or (s_ego_max + lap_length <= s_veh_min or s_ego_min + lap_length >= s_veh_max)
    ):
        overlap_flag = False
    return overlap_flag

def check_ego_agent_distance(ego: ModelBase, agent: ModelBase, racing_game_param: RacingGameParam, lap_length: float) -> bool:
    """ Whether overtaking or not, determining by distance
    """
    delta_v = abs(ego.xcurv[0] - agent.xcurv[0])
    s_agent = copy.deepcopy(agent.xcurv[4])
    s_ego = copy.deepcopy(ego.xcurv[4])
    while s_agent > lap_length:
        s_agent = s_agent - lap_length
    while s_ego > lap_length:
        s_ego = s_ego - lap_length
    return (
            # agent and ego in sam lap, agent is in front of the ego
            s_agent - s_ego <= racing_game_param.safety_factor * ego.param.length
                + racing_game_param.planning_prediction_factor * delta_v
            and (s_agent >= s_ego)
        ) or (
            # agent is in next lap, agent is in front of the ego
            s_agent + lap_length - s_ego <= racing_game_param.safety_factor * ego.param.length
                + racing_game_param.planning_prediction_factor * delta_v
            and s_agent + lap_length >= s_ego
        ) or (
            # agent and ego in same lap, ego is in front of the agent
            -s_agent + s_ego <= 1.0 * ego.param.length and s_agent <= s_ego
        ) or (
            # ego is in next lap, ego is in front of the agent
            -s_agent + s_ego + lap_length <= 1.0 * ego.param.length and s_agent <= s_ego + lap_length
        )


def get_traj_xglob(traj_xcurv: np.ndarray, track: ClosedTrack):
    """From local representation to global representation
    
    Params
    ------
    traj_xcurv: the trajectory in local state repsentation
    track: the ClosedTrack object

    Returns
    -------
    global trajectory
    """
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


class PlannerBase:
    """ Base class of all [pid, lqr, ilqr, mpc, lmpc] planner
    """
    def __init__(self):
        self.agent_name: Union[str, None] = None
        """Name of the agent (vehicle) being planned, i.e, the
        ego vehicle to this planner"""
        self.time: float = 0.0
        self.timestep: float = None
        """Delta T, None if unset"""
        self.x: np.ndarray = None
        """Current x (local)"""
        self.xglob: np.ndarray = None
        """Current global x"""
        self.u: np.ndarray = None
        """The planned control inputs"""
        self.realtime_flag = False
        """False if run in simulator"""
        # store the information (e.g. states, inputs) of current lap
        self.lap_times: List[float] = []
        self.lap_xcurvs: List[np.ndarray] = []
        self.lap_xglobs: List[np.ndarray] = []
        self.lap_inputs: List[np.ndarray] = []
        self.lap_times.append(self.time)
        # store the information (e.g. state, inputs) of the whole simulation
        self.times: List[float] = []
        self.xglobs: List[float] = []
        self.xcurvs: List[float] = []
        self.inputs: List[float] = []

        self.laps = 0
        """Number of laps"""
        self.track: ClosedTrack = None
        self.opti_traj_xcurv: np.ndarray = None
        """The optimal trajectory as baseline for overtaking, represented as local states"""
        self.opti_traj_xglob: np.ndarray = None
        """The optimal trajectory as baseline for overtaking, represented as global states"""

    def set_track(self, track: ClosedTrack):
        """ Set the track parameters for the planner
        """
        self.track = track
        self.lap_length = track.lap_length
        self.point_and_tangent = track.point_and_tangent
        self.lap_width = track.width

    def set_opti_traj(self, opti_traj_xcurv: np.ndarray, opti_traj_xglob: np.ndarray):
        """ Set the optimal trajectory as reference
        """
        self.opti_traj_xcurv = opti_traj_xcurv
        self.opti_traj_xglob = opti_traj_xglob

    def set_racing_env(self, racing_env: RacingEnv):
        """Specify the RacingEnv object for simulation / hardware communication"""
        self.racing_env = racing_env

    def set_timestep(self, timestep: float):
        """ Speficy the detal T"""
        self.timestep = timestep

    def set_target_speed(self, vt: float):
        self.vt = vt

    def set_target_deviation(self, eyt: float):
        self.eyt = eyt

    def set_state(self, xcurv: np.ndarray, xglob: np.ndarray):
        """Set the current state (both in local and global representation)

        Params
        ------
        xcurv: local state representation
        xglob: global state representation
        """
        self.x = xcurv
        self.xglob = xglob

    def calc_input(self):
        """ Run the planner and generate the next control inputs

        The function has no inputs nor outputs. The current states
        are expected to be stored in `self.x` and `self.xglob`, set
        by `self.set_state(...)` and the calculation results can be
        found in `self.u`, accessible via `self.get_input()`
        """
        pass

    def get_input(self):
        """ After calling the calc_input, use this method to retrieve
        the computed planner results, i.e., inputs to vehicle controllers. 
        """
        return self.u

    def update_memory(self, current_lap: int):
        """ Update the internal state variable after one planning step. 

        Params
        ------
        current_lap: the current lap index
        """
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


