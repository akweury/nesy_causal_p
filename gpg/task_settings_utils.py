# Created by X at 13.02.25

import math
import random
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
