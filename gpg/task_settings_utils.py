# Created by X at 13.02.25

import math
import random

import numpy as np
from scipy.spatial.distance import cdist

from scipy.interpolate import make_interp_spline, interp1d

from src import bk


class kandinskyShape:
    def __init__(self, shape="", color="", x=0.5, y=0.5, size=0.5, line_width=1.0,
                 solid=False, start_angle=0, end_angle=math.pi * 2 / 3, ):
        self.shape = shape
        self.color = color
        self.x = x
        self.y = y
        self.size = size
        self.line_width = line_width
        self.solid = solid
        self.start_angle = start_angle
        self.end_angle = end_angle

    def __str__(self):
        return self.color + " " + self.shape + " (" + \
            str(self.size) + "," + str(self.x) + "," + str(self.y) + ")"


def generate_points(center, radius, n, min_distance):
    points = []
    attempts = 0
    max_attempts = n * 300  # To prevent infinite loops

    while len(points) < n and attempts < max_attempts:
        # Generate random point in polar coordinates
        r = radius * math.sqrt(random.uniform(0, 1))  # sqrt for uniform distribution in the circle
        theta = random.uniform(0, 2 * math.pi)

        # Convert polar to Cartesian coordinates
        x = center[0] + r * math.cos(theta)
        y = center[1] + r * math.sin(theta)

        new_point = (x, y)

        # Check distance from all existing points
        if all(math.hypot(x - px, y - py) >= min_distance for px, py in points):
            points.append(new_point)

        attempts += 1

    if len(points) < n:
        print(f"Warning: Only generated {len(points)} points after {max_attempts} attempts.")

    return points


def feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy):
    objs = []
    # draw circles
    objs.append(kandinskyShape(color=colors[0],
                               shape="square", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

    objs.append(kandinskyShape(color=colors[1],
                               shape="square", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

    objs.append(kandinskyShape(color=colors[2],
                               shape="square", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

    objs.append(kandinskyShape(color=colors[3],
                               shape="square", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

    objs.append(kandinskyShape(color="lightgray", shape="circle", size=so, x=xs, y=ys,
                               line_width=-1, solid=True))

    return objs


def feature_closure_square(colors, cir_so, xs, ys, dx, dy):
    objs = []
    # draw circles
    objs.append(kandinskyShape(color=colors[0],
                               shape="pac_man", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True,
                               start_angle=90, end_angle=360))

    objs.append(kandinskyShape(color=colors[1],
                               shape="pac_man", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True,
                               start_angle=270, end_angle=540))

    objs.append(kandinskyShape(color=colors[2],
                               shape="pac_man", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True,
                               start_angle=0, end_angle=270))

    objs.append(kandinskyShape(color=colors[3],
                               shape="pac_man", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True,
                               start_angle=180, end_angle=450))

    return objs


def feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s):
    objs = []
    objs.append(kandinskyShape(color=colors[0],
                               shape="pac_man", size=cir_so, x=xs, y=ys - s, line_width=-1, solid=True,
                               start_angle=120,
                               end_angle=420, ))

    objs.append(kandinskyShape(color=colors[1],
                               shape="pac_man", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True,
                               start_angle=240, end_angle=540))

    objs.append(kandinskyShape(color=colors[2],
                               shape="pac_man", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True,
                               start_angle=0, end_angle=300))

    return objs


def random_colors(random_color_num, specific_colors=None):
    color = random.sample(bk.color_large_exclude_gray, random_color_num)
    if specific_colors is not None:
        color += specific_colors
    random.shuffle(color)
    return color


import matplotlib.pyplot as plt


def get_spline_points(points, n):
    # Separate the points into x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    # Generate a smooth spline curve (use k=3 for cubic spline interpolation)
    # Spline interpolation
    spl_x = make_interp_spline(np.linspace(0, 1, len(x)), x, k=2)
    spl_y = make_interp_spline(np.linspace(0, 1, len(y)), y, k=2)

    # Dense sampling to approximate arc-length
    dense_t = np.linspace(0, 1, 1000)
    dense_x, dense_y = spl_x(dense_t), spl_y(dense_t)

    # Calculate cumulative arc length
    arc_lengths = np.sqrt(np.diff(dense_x) ** 2 + np.diff(dense_y) ** 2)
    cum_arc_length = np.insert(np.cumsum(arc_lengths), 0, 0)

    # Interpolate to find points equally spaced by arc-length
    equal_distances = np.linspace(0, cum_arc_length[-1], n)
    interp_t = interp1d(cum_arc_length, dense_t)(equal_distances)

    # Get equally spaced points
    equal_x, equal_y = spl_x(interp_t), spl_y(interp_t)
    # # Plot results
    # plt.figure(figsize=(8, 5))
    # plt.plot(x, y, 'o', label='Original Points')
    # plt.plot(dense_x, dense_y, '-', alpha=0.4, label='Spline Curve')
    # plt.plot(equal_x, equal_y, 'ro', label='Equally Spaced Points')
    # plt.legend()
    # plt.title(f'Spline Curve with {n} Equally Spaced Points')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()
    positions = np.stack([equal_x, equal_y], axis=-1)
    return positions


def get_spline_objs(line_key_points, color, shape, so, sample_num=8):
    line_points = get_spline_points(line_key_points, sample_num)
    objs = []
    for i in range(len(line_points)):
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=float(line_points[i, 0]),
                                   y=float(line_points[i, 1]),
                                   line_width=-1, solid=True))

    return objs


def get_shaded_spline_objs(line_key_points, color, shape, so, dx, dy, d_color, sample_num=8):
    line_points = get_spline_points(line_key_points, sample_num)
    objs = []
    for i in range(len(line_points)):
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=float(line_points[i, 0]),
                                   y=float(line_points[i, 1]),
                                   line_width=-1, solid=True))
        objs.append(kandinskyShape(color=d_color, shape=shape, size=so,
                                   x=float(line_points[i, 0]) + dx,
                                   y=float(line_points[i, 1]) + dy,
                                   line_width=-1, solid=True))

    return objs


def generate_positions(n, m, t=0.1, min_range=0.1, max_range=0.9):
    num_points = n * m
    positions = []

    while len(positions) < num_points:
        candidate = np.random.uniform(min_range, max_range, size=(1, 2))

        if len(positions) == 0 or np.min(cdist(positions, candidate)) >= t:
            positions.append(candidate[0])

    positions = np.array(positions)

    return positions.reshape(n, m, 2)


def get_symmetry_surrounding_positions(angle, radius,dtype, num_points=2):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    positions = []
    for p_i in range(num_points):
        angle_offset = 0.3 * p_i
        shifted_angle = angle + angle_offset
        x = 0.5 + radius * math.cos(shifted_angle)
        y = 0.5 + radius * math.sin(shifted_angle)
        if dtype:
            x_symmetry = 0.5 - radius * math.cos(shifted_angle)

        else:
            x_symmetry = 0.5 - radius * math.sin(shifted_angle)
        positions.append((x, y))
        positions.append((x_symmetry, y))

    return positions
