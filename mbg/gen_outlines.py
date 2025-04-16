# Created by MacBook Pro at 15.04.25


import numpy as np
import os
import config
from math import cos, sin, radians
from itertools import combinations


def rotate(points, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return points @ rot_matrix.T


def normalize_shape(points):
    """ Normalize shape to fit inside [-0.5, 0.5] x [-0.5, 0.5] box. """
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_extent = np.max(np.linalg.norm(points, axis=1))
    return points / (2 * max_extent)  # normalize into radius-0.5

def generate_triangle_points_in_unit_circle(step_deg=10):
    angles = np.arange(0, 360, step_deg)
    points = [(cos(radians(a)), sin(radians(a))) for a in angles]
    return np.array(points)

def is_valid_triangle(A, B, C, epsilon=0.8):
    AB = B - A
    AC = C - A
    cross_product = np.abs(np.cross(AB, AC))
    return cross_product > epsilon  # avoid collinear points


# Updated triangle generator
def generate_all_triangles(min_angle=15, max_angle=150, step=1):
    # Generate all valid triangles from points on unit circle
    unit_circle_points = generate_triangle_points_in_unit_circle(step_deg=10)
    indices = range(len(unit_circle_points))
    triangle_set = []

    for i, j, k in combinations(indices, 3):
        A, B, C = unit_circle_points[i], unit_circle_points[j], unit_circle_points[k]
        if is_valid_triangle(A, B, C):
            triangle = np.array([A, B, C])
            triangle_normalized = normalize_shape(triangle)
            triangle_set.append(triangle_normalized)
    return triangle_set

def generate_rectangle(aspect_ratio, angle_deg):
    """Generate rectangle centered at origin with aspect ratio and rotation."""
    w = np.sqrt(1 / aspect_ratio)
    h = w * aspect_ratio
    rect = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])
    rect_rotated = rotate(rect, angle_deg)
    return normalize_shape(rect_rotated)


def generate_ellipse(aspect_ratio, angle_deg, num_points=100):
    """Generate ellipse perimeter points."""
    t = np.linspace(0, 2 * np.pi, num_points)
    x = 0.5 * np.cos(t)
    y = 0.5 * aspect_ratio * np.sin(t)
    ellipse = np.stack([x, y], axis=1)
    ellipse_rotated = rotate(ellipse, angle_deg)
    return normalize_shape(ellipse_rotated)


# ---------- Unified Generator Function ----------

def generate_and_save_shapes(output_dir):
    shape_data = {"triangle": generate_all_triangles(), "rectangle": [], "ellipse": []}
    # Generate and save updated triangle shapes

    # # Triangles
    # for a in range(20, 160, 5):  # coarse first
    #     for b in range(20, 160 - a, 5):
    #         tri = generate_triangle(a, b)
    #         if tri is not None:
    #             shape_data["triangle"].append(tri)

    # Rectangles
    aspect_ratios = np.arange(0.2, 5.01, 0.05)
    for r in aspect_ratios:
        for angle in range(0, 180, 1):
            rect = generate_rectangle(r, angle)
            shape_data["rectangle"].append(rect)

    # Ellipses
    for r in aspect_ratios:
        for angle in range(0, 180, 1):
            ell = generate_ellipse(r, angle)
            shape_data["ellipse"].append(ell)

    # Save as .npy files
    for shape, data in shape_data.items():
        arr = np.array(data, dtype=object)
        np.save(os.path.join(output_dir, f"{shape}_shapes.npy"), arr)

    return shape_data


# Generate and save shapes
shape_data = generate_and_save_shapes(config.mb_outlines)
shape_data.keys()  # Return keys to confirm shape types
