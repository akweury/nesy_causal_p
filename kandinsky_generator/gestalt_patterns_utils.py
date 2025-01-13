# Created by jing at 10.01.25
import random
import math
from tqdm import tqdm
import numpy as np

from kandinsky_generator.src.kp import KandinskyUniverse
from kandinsky_generator.src.kp.KandinskyUniverse import kandinskyShape
from kandinsky_generator.src.kp.RandomKandinskyFigure import Random
from src import bk


def generate_spaced_points(n, d):
    """
    Generate n points within a 1x1 area where:
    - The distance between any two points is greater than d.
    - The distance from each point to the border is greater than d.

    Parameters:
    - n: Number of points to generate.
    - d: Minimum distance between points and from the border.

    Returns:
    - List of tuples representing the generated points.
    """
    points = []
    max_attempts = 10000  # Limit to prevent infinite loops in case of impossible conditions
    attempts = 0

    while len(points) < n and attempts < max_attempts:
        # Generate a random point within the safe area (distance d from the border)
        x = random.uniform(d, 1 - d)
        y = random.uniform(d, 1 - d)
        new_point = (x, y)

        # Check if the new point is at least distance d from all existing points
        is_valid = all(math.hypot(x - px, y - py) > d for px, py in points)

        if is_valid:
            points.append(new_point)

        attempts += 1

    return points if len(points) == n else None  # Return None if unable to generate valid points


def generate_evenly_distributed_points(p, n, r):
    """
    Generate n points evenly distributed around point p at a distance r.

    Parameters:
    - p: Tuple (x, y) representing the center point.
    - n: Number of points to generate.
    - r: Radius/distance from point p.

    Returns:
    - List of tuples representing the generated points.
    """
    points = []
    angle_step = 2 * math.pi / n  # Divide the full circle into n equal parts

    for i in range(n):
        angle = i * angle_step
        x = p[0] + r * math.cos(angle)
        y = p[1] + r * math.sin(angle)
        points.append((x, y))

    return points


def generate_points(center, radius, n, min_distance):
    points = []
    attempts = 0
    max_attempts = n * 100  # To prevent infinite loops

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