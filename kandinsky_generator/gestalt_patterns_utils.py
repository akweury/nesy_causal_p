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


def generate_clusters(th, min_cluster_center_distance=0.2):
    """
    Generate clusters of points in the unit square [0,1] x [0,1] with:
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
            offset = np.array([random.uniform(-r*0.5, r*0.5), random.uniform(-r*0.5, r*0.5)])
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