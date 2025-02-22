# Created by X at 13.02.25
import random

import numpy as np

from src import bk
from gpg.task_settings_utils import *


def proximity_one_shape(so, dtype):
    """
    positive pattern :
    + 2 clusters of objects, each cluster has size 2~4
    + each cluster has at least one red triangle

    negative pattern:
    - 2 clusters of objects, each cluster has size 2~4
    - at most one cluster as at least one red triangle
    """

    objs = []
    cluster_dist = 0.2
    neighbour_dist = 0.1
    group_sizes = [2, 3]
    group_nums = random.choice([2, 3])
    group_anchors = [[0.3, 0.3], [0.7, 0.7], [0.8, 0.3]]
    group_anchors = group_anchors[:group_nums]
    group_radius = 0.1

    # positive
    if dtype:
        for a_i in range(len(group_anchors)):
            group_size = random.choice(group_sizes)
            neighbour_points = [generate_points(anchor, group_radius, group_size, neighbour_dist) for
                                anchor in group_anchors]
            for i in range(group_size):
                if i == 0:
                    shape = "triangle"
                    color = "red"
                else:
                    shape = random.choice(bk.bk_shapes[1:])
                    color = random.choice(bk.color_large_exclude_gray)
                x = neighbour_points[a_i][i][0]
                y = neighbour_points[a_i][i][1]
                objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))
    # negative
    else:
        for a_i in range(len(group_anchors)):
            group_size = random.choice(group_sizes)
            neighbour_points = [generate_points(anchor, group_radius, group_size, neighbour_dist) for anchor in
                                group_anchors]
            for i in range(group_size):
                shape = random.choice(bk.bk_shapes[1:])
                color = random.choice(bk.color_large_exclude_gray)
                if (i == 0 and a_i == 0) or (group_nums == 3 and a_i == 1 and i == 0):
                    shape = "triangle"
                    color = "red"
                x = neighbour_points[a_i][i][0]
                y = neighbour_points[a_i][i][1]
                objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))

    return objs


# def proximity_red_triangle(so, dtype, cluster_num=1):
#
#     objs = []
#     cluster_dist = 0.2
#     neighbour_dist = 0.05
#     group_sizes = [2, 3]
#     group_nums = random.choice([2, 3])
#     group_anchors = [[0.3, 0.3], [0.7, 0.7], [0.8, 0.3]]
#     group_anchors = group_anchors[:group_nums]
#     group_radius = 0.1
#     if dtype:
#         for a_i in range(cluster_num):
#             group_size = random.choice(group_sizes)
#             neighbour_points = [generate_points(anchor, group_radius, group_size, neighbour_dist) for
#                                 anchor in group_anchors]
#             for i in range(group_size):
#                 if i == 0:
#                     shape = "triangle"
#                     color = "red"
#                 else:
#                     shape = random.choice(bk.bk_shapes[1:])
#                     color = random.choice(bk.color_large_exclude_gray)
#                 x = neighbour_points[a_i][i][0]
#                 y = neighbour_points[a_i][i][1]
#                 objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))
#     else:
#         for a_i in range(len(group_anchors)):
#             group_size = random.choice(group_sizes)
#             neighbour_points = [generate_points(anchor, group_radius, group_size, neighbour_dist) for anchor in
#                                 group_anchors]
#             for i in range(group_size):
#                 shape = random.choice(bk.bk_shapes[1:])
#                 color = random.choice(bk.color_large_exclude_gray)
#                 if (i == 0 and a_i == 0) or (group_nums == 3 and a_i == 1 and i == 0):
#                     shape = "triangle"
#                     color = "red"
#                 x = neighbour_points[a_i][i][0]
#                 y = neighbour_points[a_i][i][1]
#                 objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))
#
#     return objs

def generate_random_anchor():
    return [random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)]


def proximity_red_triangle(so, dtype, cluster_num=1):
    objs = []
    cluster_dist = 0.2
    neighbour_dist = 0.05
    group_sizes = [2, 3]
    group_radius = 0.1

    # Generate random anchors for clusters
    group_anchors = [generate_random_anchor() for _ in range(cluster_num)]

    for a_i in range(cluster_num):
        group_size = random.choice(group_sizes)
        neighbour_points = generate_points(group_anchors[a_i], group_radius, group_size, neighbour_dist)

        for i in range(group_size):
            if i == 0:
                shape = "triangle"
                color = "red"
            else:
                shape = random.choice(bk.bk_shapes[1:])
                color = random.choice(bk.color_large_exclude_gray)

            x, y = neighbour_points[i]
            objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))

    return objs


def get_circumference_points(cluster_num, x, y, radius):
    """
    Generate evenly spaced points on the circumference of a circle.
    """
    points = []
    for i in range(cluster_num):
        angle = (2 * math.pi / cluster_num) * i
        cx = x + radius * math.cos(angle)
        cy = y + radius * math.sin(angle)
        points.append((cx, cy))
    return points


def get_circumference_angles(cluster_num):
    """
    Generate evenly spaced points on the circumference of a circle.
    """
    angles = []
    for i in range(cluster_num):
        angle = (2 * math.pi / cluster_num) * i
        angles.append(angle)
    return angles


def get_surrounding_positions(center, radius, num_points=2):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    positions = []
    for _ in range(num_points):
        angle_offset = random.uniform(-0.2, 0.2)  # Small random variation
        angle = math.atan2(center[1] - 0.5, center[0] - 0.5) + angle_offset
        x = 0.5 + radius * math.cos(angle)
        y = 0.5 + radius * math.sin(angle)
        positions.append((x, y))
    return positions


def symmetry_circle(so, dtype, cluster_num=1):
    objs = []

    shape = "circle"
    color = random.choice(bk.color_large_exclude_gray)
    cir_so = 0.3 + random.random() * 0.5
    objs.append(kandinskyShape(color=color, shape=shape, size=cir_so, x=0.5, y=0.5, line_width=-1, solid=True))

    # Generate evenly distributed group centers on the circumference
    group_centers = get_circumference_points(cluster_num, 0.5, 0.5, cir_so)

    for a_i in range(cluster_num):
        group_size = random.randint(1, 3)
        shape = random.choice(bk.bk_shapes[1:])

        # Get multiple nearby positions along the circumference
        positions = get_surrounding_positions(group_centers[a_i], cir_so, group_size)

        for x, y in positions:
            objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))

    return objs


def get_circumference_positions(angle, radius, num_points=2):
    """
    Generate multiple points near the given center, ensuring they remain on the circumference.
    """
    positions = []
    for p_i in range(num_points):
        angle_offset = 0.3 * p_i
        shifted_angle = angle + angle_offset
        x = 0.5 + radius * math.cos(shifted_angle)
        y = 0.5 + radius * math.sin(shifted_angle)
        positions.append((x, y))
    return positions


def proximity_circle(so, dtype, cluster_num=1):
    objs = []

    shape = "circle"
    color = random.choice(bk.color_large_exclude_gray)
    cir_so = 0.3 + random.random() * 0.5
    objs.append(kandinskyShape(color=color, shape=shape, size=cir_so, x=0.5, y=0.5, line_width=-1, solid=True))

    # Generate evenly distributed group centers on the circumference
    angles = get_circumference_angles(cluster_num)

    for a_i in range(cluster_num):
        group_size = random.randint(1, 3)
        shape = random.choice(bk.bk_shapes[1:])
        # Get multiple nearby positions along the circumference
        positions = get_circumference_positions(angles[a_i], cir_so / 2 * 0.66, group_size)
        for x, y in positions:
            objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))

    return objs


def generate_grid_anchors(cluster_num, grid_size=5, spacing=0.2):
    anchors = []
    removed_indices = random.sample(range(1, grid_size - 1), cluster_num - 1)

    for row in range(grid_size):
        for col in range(grid_size):
            if row in removed_indices or col in removed_indices:
                continue
            anchors.append([row * spacing + 0.1, col * spacing + 0.1])

    return anchors


def proximity_grid(so, dtype, cluster_num=1):
    objs = []
    grid_directions = "horizontal" if random.random() > 0.5 else "vertical"

    shape = random.choice(bk.bk_shapes[1:])
    color = random.choice(bk.color_large_exclude_gray)
    if cluster_num > 2:
        max_lines = 2
    else:
        max_lines = 3
    if not dtype:
        if random.random() > 0.5:
            cluster_num -= 1
        else:
            cluster_num += 1
    for a_i in range(cluster_num):
        grid_lines = random.randint(1, max_lines)
        for i in range(grid_lines):
            if grid_directions == "vertical":
                x = 1 / cluster_num * a_i + 1 / cluster_num / (grid_lines + 1) * (i + 1) + 0.05
                for y_i in range(5):
                    y = (y_i + 1) / 7
                    objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))
            else:
                y = 1 / cluster_num * a_i + 1 / cluster_num / (grid_lines + 1) * (i + 1) + 0.05
                for x_i in range(5):
                    x = (x_i + 1) / 7
                    objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))
    return objs


def closure_classic_square(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = (0.3 + random.random() * 0.2) * 1.2
    cir_so = so * (0.3 + random.random() * 0.2)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))

    if dtype:
        fixed_colors = ["red", "green"]
    else:
        fixed_colors = ["red", "red"] if random.random() > 0.5 else ["green", "green"]

    color = random_colors(2, fixed_colors)

    objs += feature_closure_square(color, cir_so, xs, ys, dx, dy)

    return objs


def feature_closure_two_squares(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    ys = y - r

    so = (0.3 + random.random() * 0.2) * 1.2
    cir_so = so * (0.3 + random.random() * 0.2)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))

    # first square
    xs = 0.25
    if dtype:
        fixed_colors = ["red", "green"]
    else:
        fixed_colors = ["red", "red"] if random.random() > 0.5 else ["green", "green"]

    color = random_colors(2, fixed_colors)
    objs += feature_closure_square(color, cir_so, xs, ys, dx, dy)

    # second square
    xs = 0.75
    if dtype:
        fixed_colors = ["red", "green"]
    else:
        fixed_colors = ["red", "red"] if random.random() > 0.5 else ["green", "green"]
    color = random_colors(2, fixed_colors)
    objs += feature_closure_square(color, cir_so, xs, ys, dx, dy)
    return objs


def feature_closure_four_squares(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = (0.3 + random.random() * 0.1) * 1.2
    cir_so = so * (0.3 + random.random() * 0.1)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))

    xs = 0.25
    ys = 0.25
    if dtype:
        fixed_colors = ["red", "green"]
    else:
        fixed_colors = ["red", "red"] if random.random() > 0.5 else ["green", "green"]
    color = random_colors(2, fixed_colors)
    objs += feature_closure_square(color, cir_so, xs, ys, dx, dy)

    # second square
    if dtype:
        fixed_colors = ["red", "green"]
    else:
        fixed_colors = ["red", "red"] if random.random() > 0.5 else ["green", "green"]
    color = random_colors(2, fixed_colors)
    xs = 0.75
    ys = 0.25
    objs += feature_closure_square(color, cir_so, xs, ys, dx, dy)

    # third square
    xs = 0.25
    ys = 0.75
    if dtype:
        fixed_colors = ["red", "green"]
    else:
        fixed_colors = ["red", "red"] if random.random() > 0.5 else ["green", "green"]
    color = random_colors(2, fixed_colors)
    objs += feature_closure_square(color, cir_so, xs, ys, dx, dy)

    # fourth square
    xs = 0.75
    ys = 0.75
    if dtype:
        fixed_colors = ["red", "green"]
    else:
        fixed_colors = ["red", "red"] if random.random() > 0.5 else ["green", "green"]
    color = random_colors(2, fixed_colors)
    objs += feature_closure_square(color, cir_so, xs, ys, dx, dy)

    return objs


def closure_classic_triangle_and_noise(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.4 + random.random() * 0.5
    cir_so = so * (0.3 + random.random() * 0.15)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.sin(math.radians(30))

    if dtype:
        # draw circles
        colors = random_colors(3)
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)
    else:
        if random.random() > 0.5:
            # draw circles
            objs.append(kandinskyShape(color=random.choice(["blue"]),
                                       shape="pac_man", size=cir_so, x=xs, y=ys - s, line_width=-1, solid=True,
                                       start_angle=120, end_angle=420, ))

            objs.append(kandinskyShape(color=random.choice(["blue"]),
                                       shape="pac_man", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True,
                                       start_angle=0, end_angle=300))

            objs.append(kandinskyShape(color=random.choice(["blue"]),
                                       shape="pac_man", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True,
                                       start_angle=240, end_angle=540))

            objs.append(kandinskyShape(color=random.choice(["green", "blue"]),
                                       shape="pac_man", size=cir_so, x=xs + random.random() * 0.3,
                                       y=(ys - s) + random.random() * 0.3, line_width=-1, solid=True,
                                       start_angle=120, end_angle=420))
            objs.append(kandinskyShape(color=random.choice(["green", "blue"]),
                                       shape="pac_man", size=cir_so, x=xs - random.random() * 0.3,
                                       y=(ys - s) + random.random() * 0.3, line_width=-1, solid=True,
                                       start_angle=180, end_angle=480))

        else:

            # draw circles
            for i in range(5):
                start_angle = random.randint(0, 100)
                objs.append(kandinskyShape(color=random.choice(["blue", "green"]),
                                           shape="pac_man", size=cir_so, x=random.random(),
                                           y=random.random(), line_width=-1, solid=True,
                                           start_angle=start_angle, end_angle=start_angle + 300))
    return objs


def similarity_fixed_number_two(so, dtype, grid_size=3, min_circles=3, max_circles=5, diameter=0.1, image_size=(1, 1)):
    diameter = 0.08
    centers = set()
    objs = []
    # Define grid spacing to place cluster centers evenly
    grid_spacing = image_size[0] / (grid_size + 1)
    cluster_centers = [(grid_spacing * (i + 1), grid_spacing * (j + 1)) for i in range(grid_size) for j in
                       range(grid_size)]
    random.shuffle(cluster_centers)

    yellow_counter = 0
    total = random.randint(10, 20)
    # Generate circles within each cluster
    while yellow_counter < total:
        cluster_x, cluster_y = random.choice(cluster_centers)
        cluster_color = "yellow"
        num_circles = np.random.randint(min_circles, max_circles + 1)
        cluster_points = [(cluster_x, cluster_y)]
        centers.add((cluster_x, cluster_y))
        if yellow_counter > total:
            break
        for _ in range(num_circles - 1):
            # Possible movement directions (8 directions: up, down, left, right, and diagonals)
            directions = [
                (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),  # Right, Left, Up, Down
                (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)  # Diagonals
            ]

            np.random.shuffle(directions)  # Shuffle directions to try random placements

            for dx, dy in directions:
                if yellow_counter > total:
                    break
                new_x, new_y = cluster_points[-1][0] + dx, cluster_points[-1][1] + dy
                if (0.05 < new_x < image_size[0]) and (0.05 < new_y < image_size[1]) and all(
                        (new_x - cx) ** 2 + (new_y - cy) ** 2 >= diameter ** 2 for cx, cy in centers):
                    cluster_points.append((new_x, new_y))
                    centers.add((new_x, new_y))

                    yellow_counter += 1
                    objs.append(kandinskyShape(color=cluster_color, shape="circle", size=so, x=new_x,
                                               y=new_y, line_width=-1, solid=True))
                    break
    if not dtype:
        if random.random() > 0.5:
            yellow_counter += random.randint(1, 5)
        else:
            yellow_counter -= random.randint(1, 5)
    # Generate circles within each cluster
    blue_counter = 0
    while blue_counter != yellow_counter:
        cluster_x, cluster_y = random.choice(cluster_centers)
        cluster_color = "blue"
        num_circles = np.random.randint(min_circles, max_circles + 1)
        cluster_points = [(cluster_x, cluster_y)]
        centers.add((cluster_x, cluster_y))

        for _ in range(num_circles - 1):
            if blue_counter == yellow_counter:
                break
            # Possible movement directions (8 directions: up, down, left, right, and diagonals)
            directions = [
                (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),  # Right, Left, Up, Down
                (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)  # Diagonals
            ]

            np.random.shuffle(directions)  # Shuffle directions to try random placements

            for dx, dy in directions:
                new_x, new_y = cluster_points[-1][0] + dx, cluster_points[-1][1] + dy
                if (0.05 < new_x < image_size[0]) and (0.05 < new_y < image_size[1]) and all(
                        (new_x - cx) ** 2 + (new_y - cy) ** 2 >= diameter ** 2 for cx, cy in centers):
                    cluster_points.append((new_x, new_y))
                    centers.add((new_x, new_y))
                    blue_counter += 1
                    objs.append(kandinskyShape(color=cluster_color, shape="circle", size=so, x=new_x,
                                               y=new_y, line_width=-1, solid=True))
                    break

    return objs


def similarity_fixed_number_three(so, dtype, grid_size=3, min_circles=3, max_circles=5, diameter=0.1,
                                  image_size=(1, 1)):
    diameter = 0.08
    centers = set()
    objs = []

    # Define grid spacing to place cluster centers evenly
    grid_spacing = image_size[0] / (grid_size + 1)
    cluster_centers = [(grid_spacing * (i + 1), grid_spacing * (j + 1)) for i in range(grid_size) for j in
                       range(grid_size)]
    random.shuffle(cluster_centers)

    yellow_counter = 0
    total = random.randint(10, 20)
    # Generate circles within each cluster
    while yellow_counter < total:
        cluster_x, cluster_y = random.choice(cluster_centers)
        cluster_color = "yellow"
        num_circles = np.random.randint(min_circles, max_circles + 1)
        cluster_points = [(cluster_x, cluster_y)]
        centers.add((cluster_x, cluster_y))
        if yellow_counter > total:
            break
        for _ in range(num_circles - 1):
            # Possible movement directions (8 directions: up, down, left, right, and diagonals)
            directions = [
                (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),  # Right, Left, Up, Down
                (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)  # Diagonals
            ]

            np.random.shuffle(directions)  # Shuffle directions to try random placements

            for dx, dy in directions:
                if yellow_counter > total:
                    break
                new_x, new_y = cluster_points[-1][0] + dx, cluster_points[-1][1] + dy
                if (0.05 < new_x < image_size[0]) and (0.05 < new_y < image_size[1]) and all(
                        (new_x - cx) ** 2 + (new_y - cy) ** 2 >= diameter ** 2 for cx, cy in centers):
                    cluster_points.append((new_x, new_y))
                    centers.add((new_x, new_y))

                    yellow_counter += 1
                    objs.append(kandinskyShape(color=cluster_color, shape="circle", size=so, x=new_x,
                                               y=new_y, line_width=-1, solid=True))
                    break
    if not dtype:
        if random.random() > 0.5:
            yellow_counter += random.randint(1, 5)
        else:
            yellow_counter -= random.randint(1, 5)
    # Generate circles within each cluster
    blue_counter = 0
    while blue_counter != yellow_counter:
        cluster_x, cluster_y = random.choice(cluster_centers)
        cluster_color = "blue"
        num_circles = np.random.randint(min_circles, max_circles + 1)
        cluster_points = [(cluster_x, cluster_y)]
        centers.add((cluster_x, cluster_y))

        for _ in range(num_circles - 1):
            if blue_counter == yellow_counter:
                break
            # Possible movement directions (8 directions: up, down, left, right, and diagonals)
            directions = [
                (diameter, 0), (-diameter, 0), (0, diameter), (0, -diameter),  # Right, Left, Up, Down
                (diameter, diameter), (-diameter, -diameter), (diameter, -diameter), (-diameter, diameter)  # Diagonals
            ]

            np.random.shuffle(directions)  # Shuffle directions to try random placements

            for dx, dy in directions:
                new_x, new_y = cluster_points[-1][0] + dx, cluster_points[-1][1] + dy
                if (0.05 < new_x < image_size[0]) and (0.05 < new_y < image_size[1]) and all(
                        (new_x - cx) ** 2 + (new_y - cy) ** 2 >= diameter ** 2 for cx, cy in centers):
                    cluster_points.append((new_x, new_y))
                    centers.add((new_x, new_y))
                    blue_counter += 1
                    objs.append(kandinskyShape(color=cluster_color, shape="circle", size=so, x=new_x,
                                               y=new_y, line_width=-1, solid=True))
                    break

    return objs


from scipy.spatial.distance import cdist


def generate_positions(n, m, t=0.1, min_range=0.1, max_range=0.9):
    num_points = n * m
    positions = []

    while len(positions) < num_points:
        candidate = np.random.uniform(min_range, max_range, size=(1, 2))

        if len(positions) == 0 or np.min(cdist(positions, candidate)) >= t:
            positions.append(candidate[0])

    positions = np.array(positions)

    return positions.reshape(n, m, 2)


def similarity_pacman(so, dtype, clu_num=1):
    color = random.choice(bk.color_large_exclude_gray)
    so = 0.4 + random.random() * 0.5
    cir_so = 0.1
    objs = []
    angles = [0, 90, 180, 270]
    positions = generate_positions(clu_num, 5, cir_so)
    # draw circles
    for clu_i in range(clu_num):
        for i in range(5):
            objs.append(kandinskyShape(color=color, shape="pac_man", size=cir_so,
                                       x=positions[clu_i, i][0],
                                       y=positions[clu_i, i][1],
                                       line_width=-1, solid=True, start_angle=angles[clu_i],
                                       end_angle=angles[clu_i] + 300))
    return objs


def get_task_names():
    return {
        # good figure
        # "good_figure_two_groups": "good_figure",
        # "good_figure_three_groups": "good_figure",
        # "good_figure_always_three": "good_figure",

        # proximity
        # "position_proximity_red_triangle_one": "proximity",
        # "position_proximity_red_triangle_two": "proximity",
        # "position_proximity_red_triangle_three": "proximity",
        # "grid_2": "proximity",
        # "grid_3": "proximity",
        # "grid_4": "proximity",

        # feature proximity
        # "feature_proximity_circle_two": "feature_proximity",
        # "feature_proximity_circle_three": "feature_proximity",
        # "feature_proximity_circle_four": "feature_proximity",

        # similarity shape
        # "similarity_triangle_circle": "similarity_shape",

        # similarity color
        # "fixed_number_two": "similarity_color",
        # "fixed_number_three": "similarity_color",
        # "fixed_number_four": "similarity_color",

        # feature similarity
        # "similarity_pacman_one": "feature_similarity",
        # "similarity_pacman_two": "feature_similarity",
        # "similarity_pacman_three": "feature_similarity"

        # feature closure
        # "feature_closure_one_square": "feature_closure",
        # "feature_closure_two_squares": "feature_closure",
        # "feature_closure_four_squares": "feature_closure",

        # # position closure
        # "tri_group": "position_closure",
        # "square_group": "position_closure",
        # "triangle_square": "position_closure",

        # # continuity
        # "continuity_one_splits_two": "continuity",
        # "continuity_one_splits_three": "continuity",

        # symmetry
        # "symmetry_pattern": "symmetry"

    }
