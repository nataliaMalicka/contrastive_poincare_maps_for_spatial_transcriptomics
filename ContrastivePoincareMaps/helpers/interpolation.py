"""
Geodesic interpolation on Poincare Maps.
Implementation adapted from https://github.com/facebookresearch/PoincareMaps
"""

import numpy as np


fs = 9
lw = 2
cpal = ['#4daf4a', '#e41a1c', '#377eb8', '#abd9e9']


def get_geodesic_parameters(u, v):
    nu = u[0] ** 2 + u[1] ** 2
    nv = v[0] ** 2 + v[1] ** 2
    a = (u[1] * nv - v[1] * nu + u[1] - v[1]) / (u[0] * v[1] - u[1] * v[0])
    b = (v[0] * nu - u[0] * nv + v[0] - u[0]) / (u[0] * v[1] - u[1] * v[0])
    return a, b


def geodesic_x_range(a, b, u, v):
    # Calculate the radius squared (r^2) for the geodesic circle
    r_sq = (a ** 2 / 4) + (b ** 2 / 4) - 1

    # Calculate the full range of x values for the circle
    x_min_circle = -a / 2 - np.sqrt(r_sq)
    x_max_circle = -a / 2 + np.sqrt(r_sq)

    # Ensure the range is within the Poincaré disk bounds (-1, 1)
    x_min = max(x_min_circle, -1)
    x_max = min(x_max_circle, 1)

    return x_min, x_max


def poincare_linspace(u, v, num=75, space='lin'):
    a, b = get_geodesic_parameters(u, v)
    x_min, x_max = geodesic_x_range(a, b, u, v)
    center = [-a / 2, -b / 2]
    r = np.sqrt((a ** 2 / 4 + b ** 2 / 4 - 1))

    angle_u = np.arctan2(u[1] - center[1], u[0] - center[0])
    angle_v = np.arctan2(v[1] - center[1], v[0] - center[0])
    angle = np.linspace(angle_u, angle_v, num=num)

    if abs(angle_u - angle_v) > np.pi:
        theta_1 = np.min([angle_u, angle_v])
        theta_2 = np.max([angle_u, angle_v])
        angle1 = np.linspace(theta_2, np.pi, num=num)
        angle2 = np.linspace(-np.pi, theta_1, num=num)
        angle = np.concatenate((angle1, angle2))

    x_pos = center[0] + r * np.cos(angle)
    y_pos = center[1] + r * np.sin(angle)

    interpolated = np.array([x_pos, y_pos]).T
    # check if the interpolated points are within the Poincaré disk
    # interpolated = interpolated[np.linalg.norm(interpolated, axis=1) < 1]
    return interpolated