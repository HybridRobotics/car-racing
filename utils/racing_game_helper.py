import numpy as np
from scipy.interpolate import interp1d


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

def get_bezier_control_points(num_veh, bezier_order, prediction_factor, safety_factor, track, veh_length, veh_width, optimal_traj_xcurv, max_s_obs, min_s_obs,max_delta_v):
    bezier_control_point = np.zeros((num_veh + 1, bezier_order + 1, 2)) # for each point, coordinate in (s, ey)
    func_optimal_ey = interp1d(optimal_traj_xcurv[:, 4], optimal_traj_xcurv[:, 5])
    for index in range(num_veh + 1):
        # s0
        bezier_control_point[index, 0, 0] = (
            min_s_obs
            - prediction_factor * max_delta_v
            - safety_factor * veh_length
        )
        # s3
        bezier_control_point[index, 3, 0] = (
            max_s_obs
            + prediction_factor * max_delta_v
            + safety_factor * veh_length
        )
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
        # ey1 and ey2
        # the first curve
        if index == 0:
            bezier_control_point[index, 1, 1] = track.width
            bezier_control_point[index, 2, 1] = track.width
        # the last curve
        elif index == num_veh:
            bezier_control_point[index, 1, 1] = -track.width
            bezier_control_point[index, 2, 1] = -track.width
        else:
            bezier_control_point[index, 1, 1] = (
                veh_info_list[index, 1] + 0.5 * veh_width
            )
            bezier_control_point[index, 2, 1] = (
                veh_info_list[index, 1] + 0.5 * veh_width
            )
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



def func_bezier_s(t, s0, s1, s2, s3):
    func_bezier_s = (
        s0 * ((1 - t) ** 3)
        + 3 * s1 * t * ((1 - t) ** 2)
        + 3 * s2 * (t ** 2) * (1 - t)
        + s3 * (t ** 3)
    )
    return func_bezier_s


def func_bezier_ey(t, ey0, ey1, ey2, ey3):
    func_bezier_ey = (
        ey0 * ((1 - t) ** 3)
        + 3 * ey1 * t * ((1 - t) ** 2)
        + 3 * ey2 * (t ** 2) * (1 - t)
        + ey3 * (t ** 3)
    )
    return func_bezier_ey


def debug_plot(track, vehicles, target_traj_xglob):
    fig, ax = plt.subplots()
    num_sampling_per_meter = 100
    num_track_points = int(np.floor(num_sampling_per_meter * track.lap_length))
    points_out = np.zeros((num_track_points, 2))
    points_center = np.zeros((num_track_points, 2))
    points_in = np.zeros((num_track_points, 2))
    for i in range(0, num_track_points):
        points_out[i, :] = track.get_global_position(
            i / float(num_sampling_per_meter), track.width
        )
        points_center[i, :] = track.get_global_position(
            i / float(num_sampling_per_meter), 0.0
        )
        points_in[i, :] = track.get_global_position(
            i / float(num_sampling_per_meter), -track.width
        )
    ax.plot(points_center[:, 0], points_center[:, 1], "--r")
    ax.plot(points_in[:, 0], points_in[:, 1], "-b")
    ax.plot(points_out[:, 0], points_out[:, 1], "-b")
    x_ego, y_ego, dx_ego, dy_ego, alpha_ego = get_vehicle_in_rectangle(
        vehicles["ego"].xglob, vehicles["ego"].param
    )
    ax.add_patch(
        patches.Rectangle((x_ego, y_ego), dx_ego, dy_ego, alpha_ego, color="red")
    )
    for name in list(vehicles):
        if name == "ego":
            pass
        else:
            x_car, y_car, dx_car, dy_car, alpha_car = get_vehicle_in_rectangle(
                vehicles[name].xglob, vehicles[name].param
            )
            ax.add_patch(
                patches.Rectangle(
                    (x_car, y_car), dx_car, dy_car, alpha_car, color="blue"
                )
            )
    ax.plot(target_traj_xglob[:, 4], target_traj_xglob[:, 5])
    ax.axis("equal")
    plt.show()


def get_vehicle_in_rectangle(vehicle_state_glob, veh_param):
    car_length = veh_param.length
    car_width = veh_param.width
    car_dx = 0.5 * car_length
    car_dy = 0.5 * car_width
    car_xs_origin = [car_dx, car_dx, -car_dx, -car_dx, car_dx]
    car_ys_origin = [car_dy, -car_dy, -car_dy, car_dy, car_dy]
    car_frame = np.vstack((np.array(car_xs_origin), np.array(car_ys_origin)))
    x = vehicle_state_glob[4]
    y = vehicle_state_glob[5]
    R = np.matrix(
        [
            [np.cos(vehicle_state_glob[3]), -np.sin(vehicle_state_glob[3])],
            [np.sin(vehicle_state_glob[3]), np.cos(vehicle_state_glob[3])],
        ]
    )
    rotated_car_frame = R * car_frame
    return (
        x + rotated_car_frame[0, 2],
        y + rotated_car_frame[1, 2],
        2 * car_dx,
        2 * car_dy,
        vehicle_state_glob[3] * 180 / 3.14,
    )
