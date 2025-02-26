# Created by X at 13.02.25

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
        if random.random() < 0.5:
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
        else:
            x = 0.5 - radius * math.cos(angle)
            y = 0.5 - radius * math.sin(angle)
        positions.append((x, y))
    return positions





def get_symmetry_on_cir_positions(angle, radius, num_points=2):
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


def symmetry_solar_sys(so, dtype, cluster_num=1):
    objs = []

    shape = "circle"
    color = random.choice(bk.color_large_exclude_gray)
    cir_so = 0.3 + random.random() * 0.5
    objs.append(kandinskyShape(color=color, shape=shape, size=cir_so, x=0.5, y=0.5, line_width=-1, solid=True))

    # Generate evenly distributed group centers on the circumference
    group_centers = get_circumference_points(cluster_num, 0.5, 0.5, cir_so)

    for a_i in range(cluster_num):
        shape = random.choice(bk.bk_shapes[1:])
        if dtype:
            group_size = random.randint(1, 3)
            positions = get_symmetry_on_cir_positions(group_centers[a_i], cir_so, group_size)
        else:
            group_size = random.randint(2, 4)
            # Get multiple nearby positions along the circumference
            positions = get_surrounding_positions(group_centers[a_i], cir_so, group_size)

        for x, y in positions:
            objs.append(kandinskyShape(color=color, shape=shape, size=so, x=x, y=y, line_width=-1, solid=True))

    return objs


def feature_symmetry_circle(so, dtype, cluster_num=1):
    objs = []
    shape = "circle"
    color = random.choice(bk.color_large_exclude_gray)
    cir_so = 0.3 + random.random() * 0.5
    objs.append(kandinskyShape(color=color, shape=shape, size=cir_so, x=0.5, y=0.5, line_width=-1, solid=True))

    # Generate evenly distributed group centers on the circumference
    angles = get_circumference_angles(cluster_num)

    for a_i in range(cluster_num):
        shape = random.choice(bk.bk_shapes[1:])
        group_size = random.randint(2, 4)

        positions = get_symmetry_surrounding_positions(angles[a_i], cir_so / 2 * 0.66,dtype, group_size)
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


def closure_big_triangle(so, dtype):
    objs = []
    x = 0.4 + random.random() * 0.2
    y = 0.4 + random.random() * 0.2
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
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
    x = 0.4 + random.random() * 0.5
    y = 0.4 + random.random() * 0.5
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


def closure_big_circle(so, t):
    objs = []
    x = 0.4 + random.random() * 0.2
    y = 0.4 + random.random() * 0.2
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
    n = int(2 * r * math.pi / 0.2)

    random_rotate_rad = random.random()
    for i in range(n):
        d = (i + random_rotate_rad) * 2 * math.pi / n
        if t:
            color = random.choice(["blue", "yellow"])
            shape = random.choice(["square", "triangle"])
        else:
            color = random.choice(bk.color_large_exclude_gray)
            shape = random.choice(bk.bk_shapes[1:])
        objs.append(kandinskyShape(color=color, shape=shape, size=so,
                                   x=x + r * math.cos(d), y=y + r * math.sin(d), line_width=-1, solid=True))

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


def feature_closure_circle_one(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.5 + random.random() * 0.3
    cir_so = so * (0.6 + random.random() * 0.2)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))
    colors = random.sample(bk.color_large_exclude_gray, 4)
    if dtype:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)
    else:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)
    return objs


def feature_closure_circle_two(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    ys = y - r

    so = random.random() * 0.1 + 0.3
    cir_so = random.random() * 0.1 + 0.1

    # correct the size to  the same area as an square
    s = so * 0.3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))
    colors = random.sample(bk.color_large_exclude_gray, 4)
    # second square
    xs = 0.25
    if dtype:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)
    else:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)

    xs = 0.75
    if dtype:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)
    else:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)
    return objs


def feature_closure_circle_three(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    ys = y - r

    so = random.random() * 0.1 + 0.3
    cir_so = random.random() * 0.1 + 0.1

    # correct the size to  the same area as an square
    s = so * 0.3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))
    colors = random.sample(bk.color_large_exclude_gray, 4)

    # second square
    xs = 0.5
    ys = 0.25
    if dtype:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)
    else:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)

    xs = 0.25
    ys = 0.75
    if dtype:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)
    else:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)

    xs = 0.75
    ys = 0.75
    if dtype:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)
    else:
        objs += feature_closure_circle(colors, so, cir_so, xs, ys, dx, dy)

    return objs


def feature_closure_triangle_one(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.4 + random.random() * 0.3
    cir_so = so * (0.4 + random.random() * 0.2)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))
    colors = random.sample(bk.color_large_exclude_gray, 4)
    if dtype:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)
    else:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)
    return objs


def feature_closure_triangle_two(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.4 + random.random() * 0.1
    cir_so = so * (0.4 + random.random() * 0.2)

    #
    # so = random.random()*0.1 + 0.3
    # cir_so = random.random()*0.1 + 0.1

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))
    colors = random.sample(bk.color_large_exclude_gray, 4)

    xs = 0.25
    if dtype:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)
    else:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)

    xs = 0.75
    if dtype:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)
    else:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)

    return objs


def feature_closure_triangle_three(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.4 + random.random() * 0.1
    cir_so = so * (0.4 + random.random() * 0.1)

    #
    # so = random.random()*0.1 + 0.3
    # cir_so = random.random()*0.1 + 0.1

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))
    colors = random.sample(bk.color_large_exclude_gray, 4)

    xs = 0.5
    ys = 0.25
    if dtype:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)
    else:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)

    xs = 0.25
    ys = 0.75
    if dtype:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)
    else:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)

    xs = 0.75
    ys = 0.75
    if dtype:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)
    else:
        objs += feature_closure_triangle(colors, cir_so, xs, ys, dx, dy, s)

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


def continuity_one_splits_n(so, dtype):
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


def position_continuity_two_splines(so, dtype):
    objs = []

    # draw the main road
    colors = random.sample(bk.color_large_exclude_gray, 3)
    shape = random.choice(bk.bk_shapes[1:])
    center_point = [random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)]

    line1_key_points = np.array([
        [random.random() * 0.1 + 0.1, random.random() * 0.1 + 0.7],  # start point
        center_point,
        [random.uniform(0.7, 0.8), random.uniform(0.1, 0.3)]
    ])

    line2_key_points = np.array([
        [random.random() * 0.1 + 0.1, random.uniform(0.1, 0.3)],  # start point
        center_point,
        [random.uniform(0.7, 0.8), random.uniform(0.7, 0.8)]
    ])

    line1_points = get_spline_points(line1_key_points, 7)
    line2_points = get_spline_points(line2_key_points, 7)

    for i in range(len(line1_points)):
        objs.append(kandinskyShape(color=colors[0], shape=shape, size=so, x=line1_points[i][0], y=line1_points[i][1],
                                   line_width=-1, solid=True))
    for i in range(len(line2_points)):
        objs.append(kandinskyShape(color=colors[0], shape=shape, size=so, x=line2_points[i][0], y=line2_points[i][1],
                                   line_width=-1, solid=True))
    return objs


def position_continuity_a_splines(so, dtype):
    objs = []

    # draw the main road
    colors = random.sample(bk.color_large_exclude_gray, 3)
    shape = random.choice(bk.bk_shapes[1:])
    center_point = [random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)]

    line1_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.4, 0.6)],  # start point
        center_point,
        [random.uniform(0.8, 0.9), random.uniform(0.4, 0.6)]
    ])

    line2_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.7, 0.8)],  # start point
        [random.uniform(0.4, 0.6), random.uniform(0.1, 0.2)],  # start point,
        [random.uniform(0.8, 0.9), random.uniform(0.8, 0.9)]
    ])
    objs += get_spline_objs(line1_key_points, colors[0], shape, so, 8)
    objs += get_spline_objs(line2_key_points, colors[0], shape, so, 12)

    return objs


def position_continuity_u_splines(so, dtype):
    objs = []

    # draw the main road
    colors = random.sample(bk.color_large_exclude_gray, 3)
    shape = random.choice(bk.bk_shapes[1:])
    center_point = [random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)]

    line1_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.4, 0.6)],  # start point
        center_point,
        [random.uniform(0.8, 0.9), random.uniform(0.4, 0.6)]
    ])

    line2_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.1, 0.2)],  # start point
        [random.uniform(0.4, 0.6), random.uniform(0.8, 0.9)],  # start point,
        [random.uniform(0.8, 0.9), random.uniform(0.1, 0.2)]
    ])
    objs += get_spline_objs(line1_key_points, colors[0], shape, so, 8)
    objs += get_spline_objs(line2_key_points, colors[0], shape, so, 12)

    return objs


def feature_continuity_x_splines(so, dtype):
    objs = []

    # draw the main road
    colors = random.sample(bk.color_large_exclude_gray, 3)
    shape = random.choice(bk.bk_shapes[1:])
    center_point = [random.uniform(0.4, 0.6), random.uniform(0.4, 0.6)]

    line1_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.1, 0.2)],  # start point
        center_point,
        [random.uniform(0.8, 0.9), random.uniform(0.8, 0.9)]
    ])

    line2_key_points = np.array([
        [random.uniform(0.1, 0.2), random.uniform(0.8, 0.9)],  # start point
        center_point,
        [random.uniform(0.8, 0.9), random.uniform(0.1, 0.2)]
    ])
    dx = random.uniform(0.005, 0.03)
    dy = random.uniform(-0.03, -0.005)

    objs += get_shaded_spline_objs(line1_key_points, colors[0], shape, so, dx, dy, colors[1], 8)
    objs += get_spline_objs(line2_key_points, colors[0], shape, so, 12)

    return objs


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

        # symbolic proximity
        # "position_proximity_red_triangle_one": "position_proximity",
        # "position_proximity_red_triangle_two": "position_proximity",
        # "position_proximity_red_triangle_three": "position_proximity",
        # "grid_2": "position_proximity",
        # "grid_3": "position_proximity",
        # "grid_4": "position_proximity",

        # neural feature proximity
        # "feature_proximity_circle_two": "feature_proximity",
        # "feature_proximity_circle_three": "feature_proximity",
        # "feature_proximity_circle_four": "feature_proximity",

        # symbolic feature similarity
        "fixed_number_two": "position_similarity",
        "fixed_number_three": "position_similarity",
        "fixed_number_four": "position_similarity",

        # neural feature similarity
        # "similarity_pacman_one": "feature_similarity",
        # "similarity_pacman_two": "feature_similarity",
        # "similarity_pacman_three": "feature_similarity"

        # symbolic feature closure
        # "tri_group_one": "position_closure",
        # "tri_group_two": "position_closure",
        # "tri_group_three": "position_closure",
        # "square_group_one": "position_closure",
        # "square_group_two": "position_closure",
        # "square_group_three": "position_closure",
        # "circle_group_one": "position_closure",
        # "circle_group_two": "position_closure",
        # "circle_group_three": "position_closure",

        # neural feature closure
        # "feature_closure_one_square": "feature_closure",
        # "feature_closure_two_squares": "feature_closure",
        # "feature_closure_four_squares": "feature_closure",
        # "feature_closure_one_circle": "feature_closure",
        # "feature_closure_two_circles": "feature_closure",
        # "feature_closure_three_circles": "feature_closure",
        # "feature_closure_one_triangle": "feature_closure",
        # "feature_closure_two_triangles": "feature_closure",
        # "feature_closure_three_triangles": "feature_closure",

        # position continuity
        # "continuity_one_splits_two": "position_continuity",
        # "position_continuity_x_splines": "position_continuity",
        # "position_continuity_a_splines": "position_continuity",
        # "position_continuity_u_splines": "position_continuity",

        # feature continuity
        # "feature_continuity_x_splines": "feature_continuity",

        # symmetry
        # "position_symmetry_solar": "position_symmetry",
        # "feature_symmetry_circle": "feature_symmetry"

    }
