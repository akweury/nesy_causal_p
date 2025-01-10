# Created by jing at 10.01.25
from kandinsky_generator.src.kp import KandinskyUniverse
from kandinsky_generator.gestalt_patterns_utils import *


def proximity_n_groups(so, dtype, n=2):
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


def proximity_always_n(so, dtype, n=2):
    """
        positive pattern :
        + one cluster have three objects: one triangle and two circles

        negative pattern:
        - one clusters of circles/square/triangle
        - all the clusters have same number of objects
        """
    objs = []
    cluster_dist = 0.3
    neighbour_dist = 0.1
    group_sizes = [2, 3, 4]
    random_points_num = 2
    if dtype:
        shape = ["triangle", "circle", "circle"]
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


def similarity_two_shapes(so, dtype, shape_a, shape_b, n=2):
    """
        positive pattern :
        + n x m objects, where a random row has shape_a, and the rest objects are shape_b

        negative pattern:
        - one clusters of circles/square/triangle
        - all the clusters have same number of objects
        """
    objs = []
    cluster_dist = 0.3
    neighbour_dist = 0.1
    group_sizes = [2, 3, 4]
    random_points_num = 2
    color = random.choice(bk.color_large)
    if dtype:
        row_num = random.randint(3, 5)
        col_num = random.randint(3, 5)
        diff_row_id = random.randint(0, row_num - 1)
        row_space = 1 / (row_num + 1)
        col_space = 1 / (col_num + 1)
        for x in range(row_num):
            for y in range(col_num):
                if x != diff_row_id:
                    objs.append(kandinskyShape(color=color, shape=shape_a, size=so, x=(x + 1) * row_space,
                                               y=(y + 1) * col_space, line_width=-1, solid=True))
                else:
                    objs.append(kandinskyShape(color=color, shape=shape_b, size=so, x=(x + 1) * row_space,
                                               y=(y + 1) * col_space, line_width=-1, solid=True))

    else:
        row_num = random.randint(3, 5)
        col_num = random.randint(3, 5)
        diff_row_id = random.randint(0, row_num - 1)
        row_space = 1 / (row_num + 1)
        col_space = 1 / (col_num + 1)
        for x in range(row_num):
            for y in range(col_num):
                if x != diff_row_id:
                    objs.append(kandinskyShape(color=color, shape=shape_b, size=so, x=(x + 1) * row_space,
                                               y=(y + 1) * col_space, line_width=-1, solid=True))
                else:
                    objs.append(kandinskyShape(color=color, shape=shape_a, size=so, x=(x + 1) * row_space,
                                               y=(y + 1) * col_space, line_width=-1, solid=True))
    return objs


def gen_patterns(pattern_name, dtype, example_idx):
    so = 0.1

    if pattern_name == "proximity_two_groups":
        g = lambda so, truth: proximity_n_groups(so, dtype, n=2)
    elif pattern_name == "proximity_three_groups":
        g = lambda so, truth: proximity_n_groups(so, dtype, n=3)
    elif pattern_name == "proximity_always_three":
        g = lambda so, truth: proximity_always_n(so, dtype, n=3)
    elif pattern_name == "similarity_triangle_circle":
        g = lambda so, truth: similarity_two_shapes(so, dtype, "triangle", "circle", n=3)
    elif pattern_name == "gestalt_triangle":
        pass
    elif pattern_name == "tri_group":
        pass
    elif pattern_name == "triangle_square":
        pass
    else:
        raise ValueError
    kf = g(so, dtype)
    t = 0
    tt = 0
    max_try = 1000
    while (KandinskyUniverse.overlaps(kf) or KandinskyUniverse.overflow(kf)) and (t < max_try):
        kf = g(so, dtype)
        if tt > 10:
            tt = 0
            so = so * 0.90
        tt = tt + 1
        t = t + 1
    return kf
