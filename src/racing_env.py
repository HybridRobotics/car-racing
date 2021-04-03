import numpy as np
import numpy.linalg as la
import pdb


class ClosedTrack:
    """map object
    Methods:
        get_global_position: convert position from (s, ey) to (X,Y)
        get_orientation: get (psi) from (s, ey)
        get_local_position: get (s, ey, epsi, CompletedFlag) from (X, Y, psi)
    """

    def __init__(self, spec, width):
        """Initialization
        spec: geometry of the track
        width: track width
        point_and_tangent: specs of each segments
        lap_length: length of the closed track
        """
        self.width = width
        self.spec = spec
        # Now given the above segments we compute the (x, y) points of the track and the angle of the tangent vector (psi) at
        # these points. For each segment we compute the (x, y, psi) coordinate at the last point of the segment. Furthermore,
        # we compute also the cumulative s at the starting point of the segment at signed curvature
        # point_and_tangent = [x, y, psi, cumulative s, segment length, signed curvature]
        point_and_tangent = np.zeros((spec.shape[0] + 1, 6))
        for i in range(0, spec.shape[0]):
            if spec[i, 1] == 0.0:  # If the current segment is a straight line
                l = spec[i, 0]  # Length of the segments
                if i == 0:
                    ang = 0  # Angle of the tangent vector at the starting point of the segment
                    x = 0 + l * np.cos(ang)  # x coordinate of the last point of the segment
                    y = 0 + l * np.sin(ang)  # y coordinate of the last point of the segment
                else:
                    ang = point_and_tangent[
                        i - 1, 2
                    ]  # Angle of the tangent vector at the starting point of the segment
                    x = point_and_tangent[i - 1, 0] + l * np.cos(ang)  # x coordinate of the last point of the segment
                    y = point_and_tangent[i - 1, 1] + l * np.sin(ang)  # y coordinate of the last point of the segment
                psi = ang  # Angle of the tangent vector at the last point of the segment

                if i == 0:
                    newline = np.array([x, y, psi, point_and_tangent[i, 3], l, 0])
                else:
                    newline = np.array([x, y, psi, point_and_tangent[i - 1, 3] + point_and_tangent[i - 1, 4], l, 0])

                point_and_tangent[i, :] = newline  # Write the new info
            else:
                l = spec[i, 0]  # Length of the segment
                r = spec[i, 1]  # Radius of curvature

                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                if i == 0:
                    ang = 0  # Angle of the tangent vector at the
                    # starting point of the segment
                    CenterX = 0 + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                    CenterY = 0 + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle
                else:
                    ang = point_and_tangent[i - 1, 2]  # Angle of the tangent vector at the
                    # starting point of the segment
                    CenterX = point_and_tangent[i - 1, 0] + np.abs(r) * np.cos(
                        ang + direction * np.pi / 2
                    )  # x coordinate center of circle
                    CenterY = point_and_tangent[i - 1, 1] + np.abs(r) * np.sin(
                        ang + direction * np.pi / 2
                    )  # y coordinate center of circle

                spanAng = l / np.abs(r)  # Angle spanned by the circle
                psi = wrap(ang + spanAng * np.sign(r))  # Angle of the tangent vector at the last point of the segment

                angleNormal = wrap((direction * np.pi / 2 + ang))
                angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))
                x = CenterX + np.abs(r) * np.cos(
                    angle + direction * spanAng
                )  # x coordinate of the last point of the segment
                y = CenterY + np.abs(r) * np.sin(
                    angle + direction * spanAng
                )  # y coordinate of the last point of the segment

                if i == 0:
                    newline = np.array([x, y, psi, point_and_tangent[i, 3], l, 1 / r])
                else:
                    newline = np.array([x, y, psi, point_and_tangent[i - 1, 3] + point_and_tangent[i - 1, 4], l, 1 / r])

                point_and_tangent[i, :] = newline  # Write the new info
            # plt.plot(x, y, 'or')

        xs = point_and_tangent[-2, 0]
        ys = point_and_tangent[-2, 1]
        xf = 0
        yf = 0
        psif = 0

        # plt.plot(xf, yf, 'or')
        # plt.show()
        l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)

        newline = np.array([xf, yf, psif, point_and_tangent[-2, 3] + point_and_tangent[-2, 4], l, 0])
        point_and_tangent[-1, :] = newline

        self.point_and_tangent = point_and_tangent
        self.lap_length = point_and_tangent[-1, 3] + point_and_tangent[-1, 4]

    def get_global_position(self, s, ey):
        """coordinate transformation from curvilinear reference frame (e, ey) to inertial reference frame (X, Y)
        (s, ey): position in the curvilinear reference frame
        """
        # wrap s along the track
        s_tolerance = 0.001
        while s > self.lap_length:
            s = s - self.lap_length
        while s < 0:
            s = s + self.lap_length

        # Compute the segment in which system is evolving
        point_and_tangent = self.point_and_tangent
        index = np.all(
            [[s >= point_and_tangent[:, 3]], [s < point_and_tangent[:, 3] + point_and_tangent[:, 4] + s_tolerance]],
            axis=0,
        )
        i = np.asscalar(np.where(np.squeeze(index))[0][0])

        if point_and_tangent[i, 5] == 0.0:  # If segment is a straight line
            # Extract the first final and initial point of the segment
            xf = point_and_tangent[i, 0]
            yf = point_and_tangent[i, 1]
            xs = point_and_tangent[i - 1, 0]
            ys = point_and_tangent[i - 1, 1]
            psi = point_and_tangent[i, 2]

            # Compute the segment length
            deltaL = point_and_tangent[i, 4]
            reltaL = s - point_and_tangent[i, 3]

            # Do the linear combination
            x = (1 - reltaL / deltaL) * xs + reltaL / deltaL * xf + ey * np.cos(psi + np.pi / 2)
            y = (1 - reltaL / deltaL) * ys + reltaL / deltaL * yf + ey * np.sin(psi + np.pi / 2)
        else:
            r = 1 / point_and_tangent[i, 5]  # Extract curvature
            ang = point_and_tangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)
            # Compute the center of the arc
            if r >= 0:
                direction = 1
            else:
                direction = -1

            CenterX = point_and_tangent[i - 1, 0] + np.abs(r) * np.cos(
                ang + direction * np.pi / 2
            )  # x coordinate center of circle
            CenterY = point_and_tangent[i - 1, 1] + np.abs(r) * np.sin(
                ang + direction * np.pi / 2
            )  # y coordinate center of circle
            spanAng = (s - point_and_tangent[i, 3]) / (np.pi * np.abs(r)) * np.pi
            angleNormal = wrap((direction * np.pi / 2 + ang))
            angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))

            x = CenterX + (np.abs(r) - direction * ey) * np.cos(
                angle + direction * spanAng
            )  # x coordinate of the last point of the segment
            y = CenterY + (np.abs(r) - direction * ey) * np.sin(
                angle + direction * spanAng
            )  # y coordinate of the last point of the segment

        return x, y

    def get_orientation(self, s, ey):
        # wrap s along the track
        s_tolerance = 0.001
        while s > self.lap_length:
            s = s - self.lap_length
        while s < 0:
            s = s + self.lap_length

        point_and_tangent = self.point_and_tangent
        index = np.all(
            [[s >= point_and_tangent[:, 3]], [s < point_and_tangent[:, 3] + point_and_tangent[:, 4] + s_tolerance]],
            axis=0,
        )
        i = np.asscalar(np.where(np.squeeze(index))[0][0])

        if point_and_tangent[i, 5] == 0.0:  # If segment is a straight line
            # Extract the first final and initial point of the segment
            xf = point_and_tangent[i, 0]
            yf = point_and_tangent[i, 1]
            xs = point_and_tangent[i - 1, 0]
            ys = point_and_tangent[i - 1, 1]
            psi = point_and_tangent[i, 2]

            # Compute the segment length
            deltaL = point_and_tangent[i, 4]
            reltaL = s - point_and_tangent[i, 3]

            # Do the linear combination
            x = (1 - reltaL / deltaL) * xs + reltaL / deltaL * xf + ey * np.cos(psi + np.pi / 2)
            y = (1 - reltaL / deltaL) * ys + reltaL / deltaL * yf + ey * np.sin(psi + np.pi / 2)
        else:
            r = 1 / point_and_tangent[i, 5]  # Extract curvature
            ang = point_and_tangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)
            # Compute the center of the arc
            if r >= 0:
                direction = 1
            else:
                direction = -1

            CenterX = point_and_tangent[i - 1, 0] + np.abs(r) * np.cos(
                ang + direction * np.pi / 2
            )  # x coordinate center of circle
            CenterY = point_and_tangent[i - 1, 1] + np.abs(r) * np.sin(
                ang + direction * np.pi / 2
            )  # y coordinate center of circle

            spanAng = (s - point_and_tangent[i, 3]) / (np.pi * np.abs(r)) * np.pi

            angleNormal = wrap((direction * np.pi / 2 + ang))
            angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))

            x = CenterX + (np.abs(r) - direction * ey) * np.cos(
                angle + direction * spanAng
            )  # x coordinate of the last point of the segment
            y = CenterY + (np.abs(r) - direction * ey) * np.sin(
                angle + direction * spanAng
            )  # y coordinate of the last point of the segment
            psi = angle + direction * spanAng + np.pi / 2

        return psi

    def get_local_position(self, x, y, psi):
        """coordinate transformation from inertial reference frame (X, Y) to curvilinear reference frame (s, ey)
        (X, Y): position in the inertial reference frame
        """
        point_and_tangent = self.point_and_tangent
        CompletedFlag = 0

        for i in range(0, point_and_tangent.shape[0]):
            if CompletedFlag == 1:
                break

            if point_and_tangent[i, 5] == 0.0:  # If segment is a straight line
                # Extract the first final and initial point of the segment
                xf = point_and_tangent[i, 0]
                yf = point_and_tangent[i, 1]
                xs = point_and_tangent[i - 1, 0]
                ys = point_and_tangent[i - 1, 1]

                psi_unwrap = np.unwrap([point_and_tangent[i - 1, 2], psi])[1]
                epsi = psi_unwrap - point_and_tangent[i - 1, 2]
                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    s = point_and_tangent[i, 3]
                    ey = 0
                    CompletedFlag = 1

                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = point_and_tangent[i, 3] + point_and_tangent[i, 4]
                    ey = 0
                    CompletedFlag = 1
                else:
                    if (
                        np.abs(computeAngle([x, y], [xs, ys], [xf, yf])) <= np.pi / 2
                        and np.abs(computeAngle([x, y], [xf, yf], [xs, ys])) <= np.pi / 2
                    ):
                        v1 = np.array([x, y]) - np.array([xs, ys])
                        angle = computeAngle([xf, yf], [xs, ys], [x, y])
                        s_local = la.norm(v1) * np.cos(angle)
                        s = s_local + point_and_tangent[i, 3]
                        ey = la.norm(v1) * np.sin(angle)

                        if np.abs(ey) <= self.width:
                            CompletedFlag = 1

            else:
                xf = point_and_tangent[i, 0]
                yf = point_and_tangent[i, 1]
                xs = point_and_tangent[i - 1, 0]
                ys = point_and_tangent[i - 1, 1]

                r = 1 / point_and_tangent[i, 5]  # Extract curvature
                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                ang = point_and_tangent[i - 1, 2]  # Extract angle of the tangent at the initial point (i-1)

                # Compute the center of the arc
                CenterX = xs + np.abs(r) * np.cos(ang + direction * np.pi / 2)  # x coordinate center of circle
                CenterY = ys + np.abs(r) * np.sin(ang + direction * np.pi / 2)  # y coordinate center of circle

                # Check if on the segment using angles
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    ey = 0
                    psi_unwrap = np.unwrap([ang, psi])[1]
                    epsi = psi_unwrap - ang
                    s = point_and_tangent[i, 3]
                    CompletedFlag = 1
                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = point_and_tangent[i, 3] + point_and_tangent[i, 4]
                    ey = 0
                    psi_unwrap = np.unwrap([point_and_tangent[i, 2], psi])[1]
                    epsi = psi_unwrap - point_and_tangent[i, 2]
                    CompletedFlag = 1
                else:
                    arc1 = point_and_tangent[i, 4] * point_and_tangent[i, 5]
                    arc2 = computeAngle([xs, ys], [CenterX, CenterY], [x, y])
                    if np.sign(arc1) == np.sign(arc2) and np.abs(arc1) >= np.abs(arc2):
                        v = np.array([x, y]) - np.array([CenterX, CenterY])
                        s_local = np.abs(arc2) * np.abs(r)
                        s = s_local + point_and_tangent[i, 3]
                        ey = -np.sign(direction) * (la.norm(v) - np.abs(r))
                        psi_unwrap = np.unwrap([ang + arc2, psi])[1]
                        epsi = psi_unwrap - (ang + arc2)

                        if np.abs(ey) <= self.width:
                            CompletedFlag = 1

        if epsi > 1.0:
            pdb.set_trace()

        if CompletedFlag == 0:
            s = 10000
            ey = 10000
            epsi = 10000

            print("Error!! POINT OUT OF THE TRACK!!!! <==================")
            pdb.set_trace()

        return s, ey, epsi, CompletedFlag

    def get_curvature(self, s):
        """curvature computation
        s: curvilinear abscissa at which the curvature has to be evaluated
        point_and_tangent: points and tangent vectors defining the map (these quantities are initialized in the map object)
        """
        # In case on a lap after the first one
        while s > self.lap_length:
            s = s - self.lap_length
        while s < 0:
            s = s + self.lap_length
        # Given s \in [0, lap_length] compute the curvature
        # Compute the segment in which system is evolving
        index = np.all(
            [[s >= self.point_and_tangent[:, 3]], [s < self.point_and_tangent[:, 3] + self.point_and_tangent[:, 4]]],
            axis=0,
        )
        i = int(np.where(np.squeeze(index))[0])
        curvature = self.point_and_tangent[i, 5]
        return curvature


### Internal Utilities Functions


def computeAngle(point1, origin, point2):
    # The orientation of this angle matches that of the coordinate system. Tha is why a minus sign is needed
    v1 = np.array(point1) - np.array(origin)
    v2 = np.array(point2) - np.array(origin)
    #
    # cosang = np.dot(v1, v2)
    # sinang = la.norm(np.cross(v1, v2))
    #
    # dp = np.dot(v1, v2)
    # laa = la.norm(v1)
    # lba = la.norm(v2)
    # costheta = dp / (laa * lba)

    dot = v1[0] * v2[0] + v1[1] * v2[1]  # dot product between [x1, y1] and [x2, y2]
    det = v1[0] * v2[1] - v1[1] * v2[0]  # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

    return angle  # np.arctan2(sinang, cosang)


def wrap(angle):
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle
    return w_angle


def sign(a):
    if a >= 0:
        res = 1
    else:
        res = -1
    return res