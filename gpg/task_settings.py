# Created by X at 13.02.25

import random
import math

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

    if dtype:
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        # first square
        random.shuffle(color)
        xs = 0.25
        dy = s * math.cos(math.radians(30))
        so *= 1.2
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

        # second square
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        # shuffle the list: color
        random.shuffle(color)
        xs = 0.75
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

    else:
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["green", "green"]
        # first square
        random.shuffle(color)
        xs = 0.25
        dy = s * math.cos(math.radians(30))
        so *= 1.2
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

        # second square
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 3) + ["green"]
        # shuffle the list: color
        random.shuffle(color)
        xs = 0.75
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

    return objs

def closure_four_squares(so, dtype):
    objs = []
    x = 0.5  # + random.random() * 0.8
    y = 0.8  # + random.random() * 0.8
    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    so = 0.3 + random.random() * 0.1
    cir_so = so * (0.3 + random.random() * 0.1)

    # correct the size to  the same area as an square
    s = 0.7 * math.sqrt(3) * so / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.sin(math.radians(30))

    if dtype:
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        # first square
        random.shuffle(color)
        xs = 0.25
        ys = 0.25
        dy = s * math.cos(math.radians(30))
        so *= 1.2
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

        # second square
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        # shuffle the list: color
        random.shuffle(color)
        xs = 0.75
        ys = 0.25
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

        # second square
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        # shuffle the list: color
        random.shuffle(color)
        xs = 0.25
        ys = 0.75
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

        # second square
        color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "green"]
        # shuffle the list: color
        random.shuffle(color)
        xs = 0.75
        ys = 0.75
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

    else:
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["green", "green"]
        # first square
        random.shuffle(color)
        xs = 0.25
        ys = 0.25
        dy = s * math.cos(math.radians(30))
        so *= 1.2
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))

        # second square
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["green" ,"green"]
        # shuffle the list: color
        random.shuffle(color)
        xs = 0.75
        ys = 0.25
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))
        # second square
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 3) + ["green", "green"]
        # shuffle the list: color
        random.shuffle(color)
        xs = 0.25
        ys = 0.75
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))
        if random.random() > 0.5:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["red", "red"]
        else:
            color = random.sample(bk.color_large_exclude_gray, 2) + ["green", "green"]
        # first square
        random.shuffle(color)
        xs = 0.75
        ys = 0.75
        dy = s * math.cos(math.radians(30))
        objs.append(kandinskyShape(color=color[0],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys - dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[1],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[2],
                                   shape="circle", size=cir_so, x=xs - dx, y=ys + dy, line_width=-1, solid=True))

        objs.append(kandinskyShape(color=color[3],
                                   shape="circle", size=cir_so, x=xs + dx, y=ys - dy, line_width=-1, solid=True))

        # draw square
        objs.append(kandinskyShape(color="lightgray",
                                   shape="square", size=so, x=xs, y=ys, line_width=-1, solid=True))
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


def get_task_names():
    return {
        # good figure
        # "good_figure_two_groups": "good_figure",
        # "good_figure_three_groups": "good_figure",
        # "good_figure_always_three": "good_figure",

        # proximity
        # "proximity_red_triangle": "proximity",

        # similarity shape
        # "similarity_triangle_circle": "similarity_shape",

        # similarity color
        # "fixed_number": "similarity_color",

        # feature closure
        "closure_square_red_yellow":"feature_closure",
        "closure_four_squares": "feature_closure",
        "gestalt_triangle": "feature_closure",

        # # position closure
        # "tri_group": "position_closure",
        # "square_group": "position_closure",
        # "triangle_square": "position_closure",

        # # continuity
        # "continuity_one_splits_two": "continuity",
        # "continuity_one_splits_three": "continuity",

        # symmetry
        "symmetry_pattern": "symmetry"

    }