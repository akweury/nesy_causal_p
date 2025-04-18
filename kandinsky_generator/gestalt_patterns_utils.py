# Created by X at 10.01.25
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


def generate_clusters(th, min_cluster_center_distance=0.2):
    """
    Generate clusters of points in the unit rectangle [0,1] x [0,1] with:
      - Each point at least 'th' away from the border.
      - Each cluster (1 to 3 points) generated around a cluster center.
      - Cluster centers are forced to be at least 'min_cluster_center_distance' apart,
        ensuring that clusters are well separated.

    Parameters:
      th: float
         Minimum distance from any point to the border.
      min_cluster_center_distance: float
         Minimum allowed distance between any two cluster centers.

    Returns:
      clusters: list of lists, where each inner list contains the points (as NumPy arrays)
                belonging to one cluster.
    """
    clusters = []
    safe_length = 1 - 2 * th  # length of the safe interval in each dimension

    # Choose the maximum spread for points within a cluster.
    # We want this spread to be small compared to both the safe region and the separation between clusters.
    max_cluster_spread = min(safe_length / 10.0, min_cluster_center_distance / 3.0)

    # Decide on a random number of clusters (between 1 and 5)
    num_clusters = random.randint(2, 4)
    centers = []

    # To ensure that the cluster (with its maximum spread) does not violate the border constraint,
    # we choose centers from the reduced safe region:
    safe_lower = th + max_cluster_spread
    safe_upper = 1 - th - max_cluster_spread

    # Use rejection sampling to generate cluster centers that are at least min_cluster_center_distance apart.
    attempts = 0
    max_attempts = 1000
    while len(centers) < num_clusters and attempts < max_attempts:
        candidate = np.array([
            random.uniform(safe_lower, safe_upper),
            random.uniform(safe_lower, safe_upper)
        ])
        if all(np.linalg.norm(candidate - c) >= min_cluster_center_distance for c in centers):
            centers.append(candidate)
        attempts += 1

    if len(centers) < num_clusters:
        print("Warning: Only generated", len(centers), "clusters instead of", num_clusters,
              "with the specified separation.")

    # For each cluster center, generate 1-3 points close to the center.
    for center in centers:
        num_points = random.randint(2, 3)
        cluster_points = []
        # For this cluster, choose a spread r that is at most max_cluster_spread.
        r = random.uniform(0, max_cluster_spread)
        for _ in range(num_points):
            # Each coordinate offset is chosen uniformly from [-r, r]
            offset = np.array([random.uniform(-r * 0.5, r * 0.5), random.uniform(-r * 0.5, r * 0.5)])
            point = center + offset
            # Clamp the point (as a precaution) to the safe region [th, 1-th]
            point[0] = min(max(point[0], th), 1 - th)
            point[1] = min(max(point[1], th), 1 - th)
            cluster_points.append(point)
        clusters.append(cluster_points)

    return clusters


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


def euclidean_distance(anchor, existing):
    return math.sqrt((anchor[0] - existing[0]) ** 2 + (anchor[1] - existing[1]) ** 2)


def generate_random_anchor(existing_anchors, cluster_dist=0.1, x_min=0.4, x_max=0.7, y_min=0.4, y_max=0.7):
    # Increased to ensure clear separation
    while True:
        anchor = [random.uniform(x_min, x_max), random.uniform(y_min, y_max)]
        if all(euclidean_distance(anchor, existing) > cluster_dist for existing in existing_anchors):
            return anchor


def get_feature_rectangle_positions(anchor, clu_size):
    positions = []

    x = anchor[0]
    y = anchor[1]

    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y)) * 0.5
    xs = x
    ys = y - r

    # correct the size to  the same area as an rectangle
    s = 0.7 * math.sqrt(3) * clu_size / 3
    dx = s * math.cos(math.radians(30))
    dy = s * math.cos(math.radians(30))

    positions.append([xs - dx, ys - dy])
    positions.append([xs + dx, ys + dy])
    positions.append([xs - dx, ys + dy])
    positions.append([xs + dx, ys - dy])

    return positions


def get_random_positions(obj_quantity, obj_size):
    group_anchors = []
    for _ in range(obj_quantity):
        group_anchors.append(
            generate_random_anchor(group_anchors, cluster_dist=obj_size, x_min=0.1, x_max=0.9, y_min=0.1, y_max=0.9))

    return group_anchors


def get_triangle_positions(obj_quantity, x, y):
    positions = []

    r = 0.3 - min(abs(0.5 - x), abs(0.5 - y))
    n = {"s": 6, "m": 15, "l": 20}.get(obj_quantity, 2)
    r = {"s": r * 0.8, "m": r, "l": r * 1.2}.get(obj_quantity, 2)
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
        positions.append([xs + i * dxi, ys + i * dyi])

    xs = x + dx
    ys = y + dy
    xe = x - dx
    ye = y + dy
    dxi = (xe - xs) / n
    dyi = (ye - ys) / n
    for i in range(n):
        positions.append([xs + (i + 1) * dxi, ys + (i + 1) * dyi])

    xs = x - dx
    ys = y + dy
    xe = x
    ye = y - r
    dxi = (xe - xs) / n
    dyi = (ye - ys) / n
    for i in range(n - 1):
        positions.append([xs + (i + 1) * dxi, ys + (i + 1) * dyi])

    return positions


def get_rectangle_positions(obj_quantity, x, y):
    r = 0.4 - min(abs(0.5 - x), abs(0.5 - y))

    m = {"s": 8, "m": 16, "l": 24}.get(obj_quantity, 2)
    r = {"s": r * 0.8, "m": r, "l": r * 1.2}.get(obj_quantity, 2)

    minx = x - r / 2
    maxx = x + r / 2
    miny = y - r / 2
    maxy = y + r / 2
    n = int(m / 4)
    dx = r / n

    positions = []
    for i in range(n + 1):
        positions.append([minx + i * dx, miny])
        positions.append([minx + i * dx, maxy])

    for i in range(n - 1):
        positions.append([minx, miny + (i + 1) * dx])
        positions.append([maxx, miny + (i + 1) * dx])

    return positions
