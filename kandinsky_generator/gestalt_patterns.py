# Created by X at 10.01.25
import random

from kandinsky_generator.src.kp import KandinskyUniverse
from kandinsky_generator.gestalt_patterns_utils import *


def good_figure_objs(so, dtype, n=2):
    """
    positive pattern :
    + two clusters of circles/square/triangle
    + all the clusters have same number of objects

    negative pattern:
    - one clusters of circles/square/triangle
    - all the clusters have same number of objects
    """

    objs = []
    cluster_dist = 0.2
    neighbour_dist = 0.1
    group_sizes = [2, 3, 4]
    if dtype:
        shape = random.choice(["circle", "square", "triangle"])
        color = random.choice(bk.color_large_exclude_gray)
        group_anchors = generate_spaced_points(n, cluster_dist)
        group_size = random.choice(group_sizes)
        neighbour_points = [generate_evenly_distributed_points(anchor, group_size, neighbour_dist) for anchor in
                            group_anchors]
        for a_i in range(len(group_anchors)):
            for i in range(group_size):
                x = neighbour_points[a_i][i][0]
                y = neighbour_points[a_i][i][1]
                objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))

    else:
        mode = random.choice(["one_cluster", "different_number"])
        if mode == "one_cluster":
            shape = random.choice(["circle", "square", "triangle"])
            color = random.choice(bk.color_large_exclude_gray)
            group_anchor = (random.uniform(0.4, 0.6), random.uniform(0.4, 0.6))
            group_size = random.choice(group_sizes * 2)
            neighbour_points = generate_evenly_distributed_points(group_anchor, group_size, neighbour_dist)
            for i in range(group_size):
                x = neighbour_points[i][0]
                y = neighbour_points[i][1]
                objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))
        elif mode == "different_number":
            shape = random.choice(["circle", "square", "triangle"])
            color = random.choice(bk.color_large_exclude_gray)
            group_anchors = generate_spaced_points(n, cluster_dist)
            group_size = random.sample(group_sizes, n)
            neighbour_points = [generate_evenly_distributed_points(group_anchors[a_i], group_size[a_i], neighbour_dist)
                                for a_i in range(n)]
            for a_i in range(len(group_anchors)):
                for i in range(group_size[a_i]):
                    x = neighbour_points[a_i][i][0]
                    y = neighbour_points[a_i][i][1]
                    objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))
    return objs


def good_figure_one_group_noise(so, dtype, n=2):
    """
        positive pattern :
        + one cluster have three objects: one triangle and two circles

        negative pattern:
        - one clusters of circles/square/triangle
        - all the clusters have same number of objects
        """
    objs = []
    cluster_dist = 0.25
    neighbour_dist = 0.1
    group_sizes = [2, 3, 4]
    random_points_num = 2
    if dtype:
        shape = ["triangle", "circle", "circle"]
        color = random.choice(bk.color_large_exclude_gray)
        group_anchors = generate_spaced_points(2, cluster_dist)
        try:
            neighbour_points = generate_evenly_distributed_points(group_anchors[0], len(shape), neighbour_dist)
        except TypeError:
            print("")
        for i in range(len(neighbour_points)):
            x = neighbour_points[i][0]
            y = neighbour_points[i][1]
            objs.append(kandinskyShape(color=color, shape=shape[i], size=so, x=x, y=y, line_width=-1, solid=True))

        for i in range(random_points_num):
            max_offset = 0.1
            offset_x = random.uniform(-max_offset, max_offset)
            offset_y = random.uniform(-max_offset, max_offset)

            # Ensure the new point stays within the 1x1 boundary
            x = min(max(group_anchors[1][0] + offset_x, 0), 1)
            y = min(max(group_anchors[1][1] + offset_y, 0), 1)

            objs.append(kandinskyShape(color=color, shape=shape[i], size=so, x=x, y=y, line_width=-1, solid=True))

    else:
        shape = [random.choice(["circle", "square", "triangle"]),
                 random.choice(["circle", "square", "triangle"]),
                 "square"]
        color = random.choice(bk.color_large_exclude_gray)
        group_anchors = generate_spaced_points(2, cluster_dist)
        neighbour_points = generate_evenly_distributed_points(group_anchors[0], len(shape), neighbour_dist)
        for i in range(len(neighbour_points)):
            x = neighbour_points[i][0]
            y = neighbour_points[i][1]
            objs.append(kandinskyShape(color=color, shape=shape[i], size=so, x=x, y=y, line_width=-1, solid=True))

        for i in range(random_points_num):
            max_offset = 0.1
            offset_x = random.uniform(-max_offset, max_offset)
            offset_y = random.uniform(-max_offset, max_offset)

            # Ensure the new point stays within the 1x1 boundary
            x = min(max(group_anchors[1][0] + offset_x, 0), 1)
            y = min(max(group_anchors[1][1] + offset_y, 0), 1)

            objs.append(kandinskyShape(color=color, shape=shape[i], size=so, x=x, y=y, line_width=-1, solid=True))
    return objs


def proximity_red_triangle(so, dtype):
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
    neighbour_dist = 0.05
    group_sizes = [2, 3]
    group_nums = random.choice([2, 3])
    group_anchors = [[0.3, 0.3], [0.7, 0.7], [0.8, 0.3]]
    group_anchors = group_anchors[:group_nums]
    group_radius = 0.1
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


def generate_random_clustered_circles(so, dtype,
                                      grid_size=3, min_circles=3, max_circles=5, diameter=0.1, image_size=(1, 1)):
    diameter = 0.08
    radius = diameter / 2
    centers = set()
    objs = []

    # Define grid spacing to place cluster centers evenly
    grid_spacing = image_size[0] / (grid_size + 1)
    cluster_centers = [(grid_spacing * (i + 1), grid_spacing * (j + 1)) for i in range(grid_size) for j in
                       range(grid_size)]
    random.shuffle(cluster_centers)

    yellow_clusters = cluster_centers[:4]
    blue_clusters = cluster_centers[4:8]

    yellow_counter = 0
    blue_counter = 0
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
    # aa = list(centers)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_aspect('equal')
    # ax.set_facecolor('gray')
    # for obj in objs:
    #     circle = plt.Circle((obj.x, obj.y), 0.03, color=obj.color)
    #     ax.add_patch(circle)
    # plt.show()

    return objs


def fixed_number(so, dtype):
    objs = []
    color = ["yellow", "blue"]
    radius = so / 10
    centers = []
    num_circles = 30
    non_overlapping_clustered_centers = generate_random_clustered_circles()

    if dtype:

        row_space = 1 / (row_num + 1)
        col_space = 1 / (col_num + 1)
        for x in range(row_num):
            for y in range(col_num):
                color = random.choice(["blue", "red"])
                shape = "triangle" if color == "red" else "square"
                objs.append(kandinskyShape(color=color, shape=shape, size=so, x=(x + 1) * row_space,
                                           y=(y + 1) * col_space, line_width=-1, solid=True))

    else:
        row_num = random.randint(3, 5)
        col_num = random.randint(3, 5)
        diff_row_id = random.randint(0, row_num - 1)
        row_space = 1 / (row_num + 1)
        col_space = 1 / (col_num + 1)
        colors = random.choice([["red", "yellow"], ["blue", "yellow"]])
        for x in range(row_num):
            for y in range(col_num):
                color = random.choice(colors)
                if color == "red":
                    shape = "triangle"
                elif color == "blue":
                    shape = "square"
                elif colors == ["red", "yellow"]:
                    shape = "square"
                else:
                    shape = "triangle"
                objs.append(kandinskyShape(color=color, shape=shape, size=so, x=(x + 1) * row_space,
                                           y=(y + 1) * col_space, line_width=-1, solid=True))

    return objs

    pass


def similarity_two_colors(so, dtype):
    """
        positive pattern :
        + n x m objects, shapes are randomly drawn from a uniform distribution,
        colors are randomly choose between red and blue
        ideal clause: G_1 is color red, G2 is color blue

        negative pattern:
        - shapes are randomly drawn from a uniform distribution,
        - colors are not alwasy red and blue, but also with yellow


        """
    objs = []
    color = random.choice(bk.color_large_exclude_gray)

    if dtype:
        cluster = random.randint(1, 3)
        if cluster == 1:
            obj_num = 10
            color = random.choice(bk.color_large_exclude_gray)
            shape = random.choice(["triangle", "square", "circle"])
            for i in range(obj_num):
                objs.append(kandinskyShape(color=color, shape=shape, size=so, x=random.random(), y=random.random(),
                                           line_width=-1, solid=True))
            objs.append(kandinskyShape(color="red", shape=shape, size=so, x=random.random(), y=random.random(),
                                       line_width=-1, solid=True))

        elif cluster == 2:
            obj_num = 6
            color = random.sample(bk.color_large_exclude_gray, 2)
            shape = random.sample(["triangle", "square", "circle"], 2)
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[0], shape=shape[0], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
            objs.append(kandinskyShape(color="red", shape=shape[0], size=so, x=random.random(), y=random.random(),
                                       line_width=-1, solid=True))
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[1], shape=shape[1], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
            objs.append(kandinskyShape(color="red", shape=shape[1], size=so, x=random.random(), y=random.random(),
                                       line_width=-1, solid=True))
        elif cluster == 3:
            obj_num = 3
            color = random.sample(bk.color_large_exclude_gray, 3)
            shape = random.sample(["triangle", "square", "circle"], 3)
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[0], shape=shape[0], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
            objs.append(kandinskyShape(color="red", shape=shape[0], size=so, x=random.random(), y=random.random(),
                                       line_width=-1, solid=True))
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[1], shape=shape[1], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
            objs.append(kandinskyShape(color="red", shape=shape[1], size=so, x=random.random(), y=random.random(),
                                       line_width=-1, solid=True))
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[2], shape=shape[2], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
            objs.append(kandinskyShape(color="red", shape=shape[2], size=so, x=random.random(), y=random.random(),
                                       line_width=-1, solid=True))
    else:
        cluster = random.randint(1, 3)
        if cluster == 1:
            obj_num = 10
            color = random.choice(bk.color_large_exclude_gray)
            shape = random.choice(["triangle", "square", "circle"])
            for i in range(obj_num):
                objs.append(kandinskyShape(color=color, shape=shape, size=so, x=random.random(), y=random.random(),
                                           line_width=-1, solid=True))
        elif cluster == 2:
            obj_num = 5
            color = random.sample(bk.color_large_exclude_gray, 2)
            shape = random.sample(["triangle", "square", "circle"], 2)
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[0], shape=shape[0], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
            objs.append(kandinskyShape(color="red", shape=shape[0], size=so, x=random.random(), y=random.random(),
                                       line_width=-1, solid=True))
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[1], shape=shape[1], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
        elif cluster == 3:
            obj_num = 3
            color = random.sample(bk.color_large_exclude_gray, 3)
            shape = random.sample(["triangle", "square", "circle"], 3)
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[0], shape=shape[0], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[1], shape=shape[1], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
            if random.random() > 0.5:
                objs.append(kandinskyShape(color="red", shape=shape[1], size=so, x=random.random(), y=random.random(),
                                           line_width=-1, solid=True))
            for i in range(obj_num):
                objs.append(
                    kandinskyShape(color=color[2], shape=shape[2], size=so, x=random.random(), y=random.random(),
                                   line_width=-1, solid=True))
            if random.random() > 0.5:
                objs.append(kandinskyShape(color="red", shape=shape[2], size=so, x=random.random(), y=random.random(),
                                           line_width=-1, solid=True))
    return objs


def similarity_two_pairs(so, dtype):
    """
        positive pattern :
        + n x m objects, shapes are randomly drawn from a uniform distribution,
        colors are randomly choose between red and blue
        ideal clause: G_1 is color red, G2 is color blue

        negative pattern:
        - shapes are randomly drawn from a uniform distribution,
        - colors are not alwasy red and blue, but also with yellow


        """
    objs = []
    color = random.choice(bk.color_large_exclude_gray)
    if dtype:
        positions = [
            [0.3, 0.3], [0.5, 0.4],
            [0.6, 0.7], [0.8, 0.4]
        ]
        for p_i in range(2):
            color = random.choice(bk.color_large_exclude_gray)
            shape = random.choice(bk.bk_shapes[1:])
            for o_i in range(2):
                objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                           x=positions[p_i * 2 + o_i][0],
                                           y=positions[p_i * 2 + o_i][1], line_width=-1, solid=True))

    else:
        positions = [
            [0.3, 0.3], [0.5, 0.4],
            [0.6, 0.7], [0.8, 0.4]
        ]
        for p_i in range(2):
            mode = random.choice([1, 2])
            if mode == 1:
                # same shape, not same color
                color = random.sample(bk.color_large_exclude_gray, 2)
                shape = random.choice(bk.bk_shapes[1:])
                for o_i in range(2):
                    objs.append(kandinskyShape(color=color[o_i], shape=shape, size=so,
                                               x=positions[p_i * 2 + o_i][0],
                                               y=positions[p_i * 2 + o_i][1], line_width=-1, solid=True))
            else:
                # same shape, not same color
                color = random.choice(bk.color_large_exclude_gray)
                shape = random.sample(bk.bk_shapes[1:], 2)
                for o_i in range(2):
                    objs.append(kandinskyShape(color=color, shape=shape[o_i], size=so,
                                               x=positions[p_i * 2 + o_i][0],
                                               y=positions[p_i * 2 + o_i][1], line_width=-1, solid=True))
    return objs


def closure_classic_triangle(so, dtype):
    """
    variation: group_shape, size,
    """
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

    start_angles = [120, 240, 0]
    end_angles = [angle + 300 for angle in start_angles]
    if dtype:
        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="pac_man", size=cir_so * random.uniform(0.8, 1.2), x=xs, y=ys - s,
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[0], end_angle=end_angles[0]))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="pac_man", size=cir_so * random.uniform(0.8, 1.2), x=xs + dx, y=ys + dy,
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[1], end_angle=end_angles[1]))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="pac_man", size=cir_so * random.uniform(0.8, 1.2), x=xs - dx, y=ys + dy,
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[2], end_angle=end_angles[2]))

    else:
        start_angles = random.sample(range(0, 360), 3)
        end_angles = [angle + 300 for angle in start_angles]

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="pac_man", size=cir_so * random.uniform(0.8, 1.2), x=xs, y=ys - s,
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[0], end_angle=end_angles[0]))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="pac_man", size=cir_so * random.uniform(0.8, 1.2), x=xs + dx, y=ys + dy,
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[1], end_angle=end_angles[1]))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="pac_man", size=cir_so * random.uniform(0.8, 1.2), x=xs - dx, y=ys + dy,
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[2], end_angle=end_angles[2]))

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
        objs.append(kandinskyShape(color=random.choice(["blue"]),
                                   shape="pac_man", size=cir_so, x=xs, y=ys - s, line_width=-1, solid=True,
                                   start_angle=120,
                                   end_angle=420, ))

        objs.append(kandinskyShape(color=random.choice(["blue"]),
                                   shape="pac_man", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True,
                                   start_angle=240, end_angle=540))

        objs.append(kandinskyShape(color=random.choice(["blue"]),
                                   shape="pac_man", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True,
                                   start_angle=0, end_angle=300))

        objs.append(kandinskyShape(color=random.choice(["green", "blue"]),
                                   shape="pac_man", size=cir_so, x=xs + random.random() * 0.3,
                                   y=(ys - s) + random.random() * 0.3, line_width=-1, solid=True,
                                   start_angle=120, end_angle=420))
        objs.append(kandinskyShape(color=random.choice(["green", "blue"]),
                                   shape="pac_man", size=cir_so, x=xs - random.random() * 0.3,
                                   y=(ys - s) + random.random() * 0.3, line_width=-1, solid=True,
                                   start_angle=180, end_angle=480))

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


def closure_square_red_yellow(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.3 + random.random() * 0.2
    cir_so = so * (0.3 + random.random() * 0.2)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.sin(math.radians(30))
    start_angles = [90, 270, 0, 180]
    end_angles = [angle + 270 for angle in start_angles]
    cluster_dist = 0.2
    x_min = 0.25
    x_max = 0.75
    y_min = 0.5
    y_max = 0.95
    clu_num = 2

    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(generate_random_anchor(group_anchors, cluster_dist, x_min, x_max, y_min, y_max))

    clu_size = ({1: 0.3 + random.random() * 0.2,
                 2: 0.3 + random.random() * 0.1,
                 3: 0.3 + random.random() * 0.1,
                 4: 0.2 + random.random() * 0.1
                 }.get(clu_num, 0.3))
    obj_size = clu_size * (0.3 + random.random() * 0.1)

    if dtype:
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        # first square
        random.shuffle(color)
        so *= 1.2
        positions = get_feature_square_positions(group_anchors[0], clu_size)
        objs.append(kandinskyShape(color=color[0],
                                   shape="pac_man", size=cir_so, x=positions[0][0], y=positions[0][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[0], end_angle=end_angles[0]))

        objs.append(kandinskyShape(color=color[1],
                                   shape="pac_man", size=cir_so, x=positions[1][0], y=positions[1][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[1], end_angle=end_angles[1]))

        objs.append(kandinskyShape(color=color[2],
                                   shape="pac_man", size=cir_so, x=positions[2][0], y=positions[2][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[2], end_angle=end_angles[2]))

        objs.append(kandinskyShape(color=color[3],
                                   shape="pac_man", size=cir_so, x=positions[3][0], y=positions[3][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[3], end_angle=end_angles[3]))

        # second square
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        # shuffle the list: color
        random.shuffle(color)
        positions = get_feature_square_positions(group_anchors[1], clu_size)
        objs.append(kandinskyShape(color=color[0],
                                   shape="pac_man", size=cir_so, x=positions[0][0], y=positions[0][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[0], end_angle=end_angles[0]))

        objs.append(kandinskyShape(color=color[1],
                                   shape="pac_man", size=cir_so, x=positions[1][0], y=positions[1][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[1], end_angle=end_angles[1]))

        objs.append(kandinskyShape(color=color[2],
                                   shape="pac_man", size=cir_so, x=positions[2][0], y=positions[2][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[2], end_angle=end_angles[2]))

        objs.append(kandinskyShape(color=color[3],
                                   shape="pac_man", size=cir_so, x=positions[3][0], y=positions[3][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[3], end_angle=end_angles[3]))
    else:
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["green", "green"]
        # first square
        random.shuffle(color)
        so *= 1.2
        positions = get_feature_square_positions(group_anchors[0], clu_size)
        objs.append(kandinskyShape(color=color[0],
                                   shape="pac_man", size=cir_so, x=positions[0][0], y=positions[0][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[0], end_angle=end_angles[0]))

        objs.append(kandinskyShape(color=color[1],
                                   shape="pac_man", size=cir_so, x=positions[1][0], y=positions[1][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[1], end_angle=end_angles[1]))

        objs.append(kandinskyShape(color=color[2],
                                   shape="pac_man", size=cir_so, x=positions[2][0], y=positions[2][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[2], end_angle=end_angles[2]))

        objs.append(kandinskyShape(color=color[3],
                                   shape="pac_man", size=cir_so, x=positions[3][0], y=positions[3][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[3], end_angle=end_angles[3]))

        # second square
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 3) + ["green"]
        # shuffle the list: color
        random.shuffle(color)
        xs = 0.75
        positions = get_feature_square_positions(group_anchors[1], clu_size)

        objs.append(kandinskyShape(color=color[0],
                                   shape="pac_man", size=cir_so, x=positions[0][0], y=positions[0][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[0], end_angle=end_angles[0]))

        objs.append(kandinskyShape(color=color[1],
                                   shape="pac_man", size=cir_so, x=positions[1][0], y=positions[1][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[1], end_angle=end_angles[1]))

        objs.append(kandinskyShape(color=color[2],
                                   shape="pac_man", size=cir_so, x=positions[2][0], y=positions[2][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[2], end_angle=end_angles[2]))

        objs.append(kandinskyShape(color=color[3],
                                   shape="pac_man", size=cir_so, x=positions[3][0], y=positions[3][1],
                                   line_width=-1, solid=True,
                                   start_angle=start_angles[3], end_angle=end_angles[3]))

    return objs


def closure_unit_square(group_anchor, clu_size, color, cir_so, start_angles, end_angles):
    objs = []
    positions = get_feature_square_positions(group_anchor, clu_size)
    objs.append(kandinskyShape(color=color[0],
                               shape="pac_man", size=cir_so, x=positions[0][0], y=positions[0][1],
                               line_width=-1, solid=True,
                               start_angle=start_angles[0], end_angle=end_angles[0]))

    objs.append(kandinskyShape(color=color[1],
                               shape="pac_man", size=cir_so, x=positions[1][0], y=positions[1][1],
                               line_width=-1, solid=True,
                               start_angle=start_angles[1], end_angle=end_angles[1]))

    objs.append(kandinskyShape(color=color[2],
                               shape="pac_man", size=cir_so, x=positions[2][0], y=positions[2][1],
                               line_width=-1, solid=True,
                               start_angle=start_angles[2], end_angle=end_angles[2]))

    objs.append(kandinskyShape(color=color[3],
                               shape="pac_man", size=cir_so, x=positions[3][0], y=positions[3][1],
                               line_width=-1, solid=True,
                               start_angle=start_angles[3], end_angle=end_angles[3]))
    return objs


def closure_four_squares(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5

    so = 0.3 + random.random() * 0.1
    cir_so = so * (0.3 + random.random() * 0.1)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    start_angles = [90, 270, 0, 180]
    end_angles = [angle + 270 for angle in start_angles]
    cluster_dist = 0.2
    x_min = 0.25
    x_max = 0.75
    y_min = 0.5
    y_max = 0.95
    clu_num = 4

    group_anchors = []
    for _ in range(clu_num):
        group_anchors.append(generate_random_anchor(group_anchors, cluster_dist, x_min, x_max, y_min, y_max))

    clu_size = ({1: 0.3 + random.random() * 0.2,
                 2: 0.3 + random.random() * 0.1,
                 3: 0.3 + random.random() * 0.1,
                 4: 0.2 + random.random() * 0.1
                 }.get(clu_num, 0.3))
    cir_so = clu_size * (0.3 + random.random() * 0.1)
    if dtype:
        so *= 1.2

        # 1st square
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        random.shuffle(color)
        objs += closure_unit_square(group_anchors[0], clu_size, color, cir_so, start_angles, end_angles)

        # 2nd square
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        random.shuffle(color)
        objs += closure_unit_square(group_anchors[1], clu_size, color, cir_so, start_angles, end_angles)

        # 3rd square
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        random.shuffle(color)
        objs += closure_unit_square(group_anchors[2], clu_size, color, cir_so, start_angles, end_angles)

        # 4th square
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        random.shuffle(color)
        objs += closure_unit_square(group_anchors[3], clu_size, color, cir_so, start_angles, end_angles)

    else:
        so *= 1.2
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["green", "green"]

        # first square
        random.shuffle(color)
        objs += closure_unit_square(group_anchors[0], clu_size, color, cir_so, start_angles, end_angles)

        # second square
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["green", "green"]
        # shuffle the list: color
        random.shuffle(color)
        objs += closure_unit_square(group_anchors[1], clu_size, color, cir_so, start_angles, end_angles)

        # second square
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 3) + ["green", "green"]
        # shuffle the list: color
        random.shuffle(color)
        objs += closure_unit_square(group_anchors[2], clu_size, color, cir_so, start_angles, end_angles)

        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["green", "green"]
        # first square
        random.shuffle(color)
        objs += closure_unit_square(group_anchors[3], clu_size, color, cir_so, start_angles, end_angles)

    return objs


def closure_classic_square(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.4 + random.random() * 0.5
    cir_so = so * (0.3 + random.random() * 0.2)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.sin(math.radians(30))

    if not dtype:
        # draw circles
        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs, y=ys - s, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))
        # draw triangle
        objs.append(kandinskyShape(color="lightgray",
                                   shape="triangle", size=so, x=xs, y=ys, line_width=-1, solid=True))
    else:
        dy = s * math.cos(math.radians(30))
        so *= 1.2
        # draw circles
        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw triangle
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

    return objs


def closure_classic_circle(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.4 + random.random() * 0.5
    cir_so = so * (0.3 + random.random() * 0.2)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.sin(math.radians(30))

    if dtype:
        # draw circles
        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="square", size=cir_so, x=xs, y=ys - s, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="square", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="square", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))
        # draw triangle
        objs.append(kandinskyShape(color="lightgray",
                                   shape="circle", size=so, x=xs, y=ys, line_width=-1, solid=True))
    else:
        # draw circles
        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="square", size=cir_so, x=xs, y=ys - s, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="square", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="square", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))
        # draw triangle
        objs.append(kandinskyShape(color="lightgray",
                                   shape="triangle", size=so, x=xs, y=ys, line_width=-1, solid=True))

    return objs


def closure_big_triangle(so, dtype):
    objs = []
    x = 0.4 + random.random() * 0.2
    y = 0.4 + random.random() * 0.2
    positions = get_triangle_positions("m", x, y)
    obj_num = len(positions)

    if not dtype and random.random() < 0.3:
        positions = get_random_positions(obj_num, so)
        dtype = True
    for i in range(len(positions)):
        color = random.choice(["yellow", "green"])
        if dtype:
            shape = "square" if color == "yellow" else "circle"
        else:
            shape = "circle" if color == "yellow" else "square"

        objs.append(kandinskyShape(color=color, shape=shape, size=so * random.uniform(0.8, 1.2),
                                   x=positions[i][0], y=positions[i][1], line_width=-1, solid=True))
    return objs


def closure_big_square(so, dtype):
    objs = []
    x = 0.4 + random.random() * 0.2
    y = 0.4 + random.random() * 0.2
    r = 0.4 - min(abs(0.5 - x), abs(0.5 - y))
    m = 20

    minx = x - r / 2
    maxx = x + r / 2
    miny = y - r / 2
    maxy = y + r / 2
    n = int(m / 4)
    dx = r / n
    for i in range(n + 1):
        color = random.choice(["blue", "red"])
        if dtype:
            shape = "circle" if color == "blue" else "triangle"
        else:
            shape = "triangle" if color == "blue" else "circle"
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=minx + i * dx, y=miny, line_width=-1, solid=True))

        color = random.choice(["blue", "red"])
        if dtype:
            shape = "circle" if color == "blue" else "triangle"
        else:
            shape = "triangle" if color == "blue" else "circle"
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=minx + i * dx, y=maxy, line_width=-1, solid=True))

    for i in range(n - 1):
        color = random.choice(["blue", "red"])
        if dtype:
            shape = "circle" if color == "blue" else "triangle"
        else:
            shape = "triangle" if color == "blue" else "circle"
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=minx, y=miny + (i + 1) * dx, line_width=-1, solid=True))

        color = random.choice(["blue", "red"])
        if dtype:
            shape = "circle" if color == "blue" else "triangle"
        else:
            shape = "triangle" if color == "blue" else "circle"
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=maxx, y=miny + (i + 1) * dx, line_width=-1, solid=True))

    return objs


def continuity_one_splits_n(so, dtype, n):
    objs = []
    main_road_length = 5
    split_road_length = 4
    dx = 0.08
    dy = 0.08
    main_y = 0.5

    # draw the main road
    colors = random.sample(bk.color_large_exclude_gray, 3)
    shape = random.choice(bk.bk_shapes[1:])
    minx = 0.1
    for i in range(main_road_length):
        objs.append(kandinskyShape(color=colors[0], shape=shape, size=so,
                                   x=minx + i * dx, y=main_y, line_width=-1, solid=True))

    # draw the split roads
    minx = minx + main_road_length * dx
    mode = random.choice([0, 1])
    for i in range(split_road_length):
        if dtype:
            if mode == 0:
                # same branch
                objs.append(kandinskyShape(color=colors[0], shape=shape, size=so,
                                           x=minx + i * dx, y=main_y + (i + 1) * dy, line_width=-1, solid=True))
                # different branch
                objs.append(kandinskyShape(color=colors[1], shape=shape, size=so,
                                           x=minx + i * dx, y=main_y - (i + 1) * dy, line_width=-1, solid=True))
            else:
                # diff branch
                objs.append(kandinskyShape(color=colors[1], shape=shape, size=so,
                                           x=minx + i * dx, y=main_y + (i + 1) * dy, line_width=-1, solid=True))
                # same branch
                objs.append(kandinskyShape(color=colors[0], shape=shape, size=so,
                                           x=minx + i * dx, y=main_y - (i + 1) * dy, line_width=-1, solid=True))
        else:
            if mode == 0:
                # diff branch
                objs.append(kandinskyShape(color=colors[1], shape=shape, size=so,
                                           x=minx + i * dx, y=main_y + (i + 1) * dy, line_width=-1, solid=True))
                # different branch
                objs.append(kandinskyShape(color=colors[2], shape=shape, size=so,
                                           x=minx + i * dx, y=main_y - (i + 1) * dy, line_width=-1, solid=True))
            else:
                # same branch
                objs.append(kandinskyShape(color=colors[0], shape=shape, size=so,
                                           x=minx + i * dx, y=main_y + (i + 1) * dy, line_width=-1, solid=True))
                # same branch
                objs.append(kandinskyShape(color=colors[0], shape=shape, size=so,
                                           x=minx + i * dx, y=main_y - (i + 1) * dy, line_width=-1, solid=True))
    return objs


def symmetry_pattern(so, dtype):
    objs = []
    n = random.choice([3, 4, 5, 6])
    ys = [0.5, 0.6, 0.4, 0.7, 0.3, 0.8]
    ys = ys[:n]
    for i in range(n):
        x_shift = random.uniform(0.1, 0.4)
        colors = random.sample(bk.color_large_exclude_gray, 2)
        shape = random.choice(bk.bk_shapes[1:])
        # left part
        objs.append(kandinskyShape(color=colors[0], shape=shape, size=so,
                                   x=0.5 - x_shift, y=ys[i], line_width=-1, solid=True))
        if dtype:
            # right part symmetry: identical
            objs.append(kandinskyShape(color=colors[0], shape=shape, size=so,
                                       x=0.5 + x_shift, y=ys[i], line_width=-1, solid=True))
        else:
            # right part symmetry: change shape or color
            objs.append(kandinskyShape(color=colors[1], shape=shape, size=so,
                                       x=0.5 + x_shift, y=ys[i], line_width=-1, solid=True))
    return objs


def gen_patterns(pattern_name, dtype):
    so = 0.1
    overlap_patterns = []
    if pattern_name == "proximity_red_triangle":
        g = lambda so, truth: proximity_red_triangle(so, dtype)
    elif pattern_name == "proximity_one_shape":
        g = lambda so, truth: proximity_one_shape(so, dtype)
    elif pattern_name == "similarity_triangle_circle":
        g = lambda so, truth: similarity_two_colors(so, dtype)
    elif pattern_name == "fixed_number":
        g = lambda so, truth: generate_random_clustered_circles(so, dtype)
    elif pattern_name == "similarity_two_pairs":
        g = lambda so, truth: similarity_two_pairs(so, dtype)
    elif pattern_name == "gestalt_triangle":
        g = lambda so, truth: closure_classic_triangle(so, dtype)
    elif pattern_name == "closure_square_red_yellow":
        g = lambda so, truth: closure_square_red_yellow(so, dtype)
    elif pattern_name == "closure_four_squares":
        g = lambda so, truth: closure_four_squares(so, dtype)
    elif pattern_name == "gestalt_triangle_and_noise":
        g = lambda so, truth: closure_classic_triangle_and_noise(so, dtype)
    elif pattern_name == "gestalt_square":
        g = lambda so, truth: closure_classic_square(so, dtype)
    elif pattern_name == "gestalt_circle":
        g = lambda so, truth: closure_classic_circle(so, dtype)
    elif pattern_name == "tri_group":
        so = 0.1
        g = lambda so, truth: closure_big_triangle(so, dtype)
    elif pattern_name == "square_group":
        so = 0.1
        g = lambda so, truth: closure_big_square(so, dtype)
    elif pattern_name == "triangle_square":
        so = 0.1
        g = lambda so, truth: closure_big_square(so, dtype) + closure_big_triangle(so, dtype)
    elif pattern_name == "continuity_one_splits_two":
        g = lambda so, truth: continuity_one_splits_n(so, dtype, n=2)
    elif pattern_name == "continuity_one_splits_three":
        g = lambda so, truth: continuity_one_splits_n(so, dtype, n=3)
    elif pattern_name == "symmetry_pattern":
        g = lambda so, truth: symmetry_pattern(so, dtype)

    else:
        raise ValueError
    kf = g(so, dtype)
    t = 0
    tt = 0
    max_try = 1000
    if pattern_name not in overlap_patterns:
        while (KandinskyUniverse.overlaps(kf) or KandinskyUniverse.overflow(kf)) and (t < max_try):
            kf = g(so, dtype)
            if tt > 10:
                tt = 0
                so = so * 0.90
            tt = tt + 1
            t = t + 1
    return kf
