# Created by jing at 10.01.25
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
    neighbour_dist = 0.1
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
        row_num = random.randint(3, 5)
        col_num = random.randint(3, 5)
        diff_row_id = random.randint(0, row_num - 1)
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
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.4 + random.random() * 0.6
    cir_so = so * (0.2 + random.random() * 0.2)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.sin(math.radians(30))

    if dtype:
        # draw circles
        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs, y=ys - s, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        if random.random() > 0.5:
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
        if random.random() > 0.5:
            # draw triangle
            objs.append(kandinskyShape(color="lightgray",
                                       shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

    return objs


def closure_big_triangle(so, dtype):
    objs = []
    x = 0.4 + random.random() * 0.2
    y = 0.4 + random.random() * 0.2
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
    # n = int(2 * r * math.pi / 0.25)
    n = 15
    innerdegree = math.radians(30)
    dx = r * math.cos(innerdegree)
    dy = r * math.sin(innerdegree)

    n = round(n / 3)

    xs = x
    ys = y - r
    xe = x + dx
    ye = y + dy
    dxi = (xe - xs) / n
    dyi = (ye - ys) / n

    for i in range(n + 1):
        color = random.choice(["yellow", "green"])
        if dtype:
            shape = "square" if color == "yellow" else "circle"
        else:
            shape = "circle" if color == "yellow" else "square"
        objs.append(kandinskyShape(color=color, shape=shape, size=so, x=xs + i * dxi,
                                   y=ys + i * dyi, line_width=-1, solid=True))

    xs = x + dx
    ys = y + dy
    xe = x - dx
    ye = y + dy
    dxi = (xe - xs) / n
    dyi = (ye - ys) / n
    for i in range(n):
        color = random.choice(["yellow", "green"])
        if dtype:
            shape = "square" if color == "yellow" else "circle"
        else:
            shape = "circle" if color == "yellow" else "square"
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=xs + (i + 1) * dxi, y=ys + (i + 1) * dyi, line_width=-1, solid=True))

    xs = x - dx
    ys = y + dy
    xe = x
    ye = y - r
    dxi = (xe - xs) / n
    dyi = (ye - ys) / n
    for i in range(n - 1):
        color = random.choice(["yellow", "green"])
        if dtype:
            shape = "square" if color == "yellow" else "circle"
        else:
            shape = "circle" if color == "yellow" else "square"
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=xs + (i + 1) * dxi, y=ys + (i + 1) * dyi, line_width=-1, solid=True))
    return objs


def closure_big_square(so, dtype):
    objs = []
    x = 0.4 + random.random() * 0.2
    y = 0.4 + random.random() * 0.2
    r = 0.4 - min(abs(0.5 - x), abs(0.5 - y))
    m = 15

    minx = x - r / 2
    maxx = x + r / 2
    miny = y - r / 2
    maxy = y + r / 2
    n = int(m / 4)
    dx = r / n
    for i in range(n + 1):
        color = random.choice(["blue", "red"])
        if dtype:
            shape = "circle" if color == "blue" else "square"
        else:
            shape = "square" if color == "blue" else "circle"
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=minx + i * dx, y=miny, line_width=-1, solid=True))

        color = random.choice(["blue", "red"])
        if dtype:
            shape = "circle" if color == "blue" else "square"
        else:
            shape = "square" if color == "blue" else "circle"
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=minx + i * dx, y=maxy, line_width=-1, solid=True))

    for i in range(n - 1):
        color = random.choice(["blue", "red"])
        if dtype:
            shape = "circle" if color == "blue" else "square"
        else:
            shape = "square" if color == "blue" else "circle"
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=minx, y=miny + (i + 1) * dx, line_width=-1, solid=True))

        color = random.choice(["blue", "red"])
        if dtype:
            shape = "circle" if color == "blue" else "square"
        else:
            shape = "square" if color == "blue" else "circle"
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
    overlap_patterns = ["gestalt_triangle"]
    if pattern_name == "proximity_red_triangle":
        g = lambda so, truth: proximity_red_triangle(so, dtype)
    elif pattern_name == "similarity_triangle_circle":
        g = lambda so, truth: similarity_two_colors(so, dtype)
    elif pattern_name == "similarity_two_pairs":
        g = lambda so, truth: similarity_two_pairs(so, dtype)
    elif pattern_name == "gestalt_triangle":
        g = lambda so, truth: closure_classic_triangle(so, dtype)
    elif pattern_name == "tri_group":
        so = 0.1
        g = lambda so, truth: closure_big_triangle(so, dtype)
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
