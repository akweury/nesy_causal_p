# Created by X at 13.02.25

import math
import random


class kandinskyShape:
    def __init__(self, shape="", color="", x=0.5, y=0.5, size=0.5, line_width=1.0,
                 solid=False, start_angle=0, end_angle=math.pi*2 / 3,):
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