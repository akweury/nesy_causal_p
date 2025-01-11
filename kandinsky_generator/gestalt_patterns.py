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


def closure_classic_triangle(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.4 + random.random() * 0.6
    cir_so = so * (0.3 + random.random() * 0.2)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.sin(math.radians(30))

    # draw circles
    objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                               shape="circle", size=cir_so, x=xs, y=ys - s, line_width=-1, solid=True))

    objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                               shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

    objs.append(kandinskyShape(color=random.choice(["blue", "green", "yellow"]),
                               shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))
    if dtype:
        # draw triangle
        objs.append(kandinskyShape(color="lightgray",
                                   shape="triangle", size=so, x=xs, y=ys, line_width=-1, solid=True))

    return objs


def closure_big_triangle(so, dtype):
    objs = []
    x = 0.4 + random.random() * 0.2
    y = 0.4 + random.random() * 0.2
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
    # n = int(2 * r * math.pi / 0.25)
    n = 20
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


def gen_patterns(pattern_name, dtype):
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
        g = lambda so, truth: closure_classic_triangle(so, dtype)
    elif pattern_name == "tri_group":
        so = 0.05
        g = lambda so, truth: closure_big_triangle(so, dtype)
    elif pattern_name == "triangle_square":
        so = 0.05
        g = lambda so, truth: closure_big_square(so, dtype) + closure_big_triangle(so, dtype)
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
