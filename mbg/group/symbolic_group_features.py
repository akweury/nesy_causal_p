# Created by MacBook Pro at 28.04.25
# mbg/symbolic_group_features.py

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

import mbg.mbg_config as param


@dataclass
class SymbolicGroupFeatures:
    cardinality: float  # |G|
    shape_dist: np.ndarray  # (C,) distribution over C shape‐classes
    avg_rgb: np.ndarray  # (3,) mean [r,g,b] in [0,1]
    centroid: np.ndarray  # (2,) [cx,cy] normalized
    bbox: np.ndarray  # (4,) [min_x,min_y,max_x,max_y] normalized
    dispersion: float  # max distance from centroid

    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            [self.cardinality],
            self.shape_dist,
            self.avg_rgb,
            self.centroid,
            self.bbox,
            [self.dispersion]
        ])

    def to_dict(self):
        return {
            "cardinality": self.cardinality,
            "shape_dist": self.shape_dist,
            "avg_rgb": self.avg_rgb,
            "centroid": self.centroid,
            "bbox": self.bbox,
            "dispersion": self.dispersion
        }


def compute_symbolic_group_features(
        objects: List[Dict],
        image_width: int,
        image_height: int
) -> SymbolicGroupFeatures:
    """
    objects: list of dicts, each with:
        'x','y' = top-left pixel coords of bbox,
        'w','h' = bbox width/height in pixels,
        'color_r','color_g','color_b', 'shape' (int)
    image_width, image_height: for normalization.
    """
    n = len(objects)
    # ---- shape distribution ----
    C = len(param.LABEL_NAMES)
    shape_counts = np.zeros(C, dtype=float)
    for o in objects:
        shape_counts += o["s"]['shape'][1:].numpy()
    shape_dist = shape_counts / n

    # ---- average color ----
    rgb = np.array([o["s"]["color"] for o in objects], float)
    avg_rgb = rgb.mean(axis=0) / 255.0

    # ---- object centroids (pixel) ----
    centroids_x = np.array([o["s"]['x'] + o["s"]['w'] / 2 for o in objects], float)
    centroids_y = np.array([o["s"]['y'] + o["s"]['h'] / 2 for o in objects], float)

    # ---- normalized group centroid ----
    cx = centroids_x.mean() / image_width
    cy = centroids_y.mean() / image_height
    centroid = np.array([cx, cy], float)

    # ---- normalized bbox corners of group ----
    x_mins = np.array([o["s"]['x'] for o in objects], float) / image_width
    y_mins = np.array([o["s"]['y'] for o in objects], float) / image_height
    x_maxs = (np.array([o["s"]['x'] for o in objects], float) + np.array([o["s"]['w'] for o in objects],
                                                                         float)) / image_width
    y_maxs = (np.array([o["s"]['y'] for o in objects], float) + np.array([o["s"]['h'] for o in objects],
                                                                         float)) / image_height
    bbox = np.array([x_mins.min(), y_mins.min(), x_maxs.max(), y_maxs.max()], float)

    # ---- dispersion (max normalized dist from group centroid) ----
    dists = np.sqrt(((centroids_x / image_width) - cx) ** 2 +
                    ((centroids_y / image_height) - cy) ** 2)
    dispersion = float(dists.max())

    return SymbolicGroupFeatures(
        cardinality=float(n),
        shape_dist=shape_dist,
        avg_rgb=avg_rgb,
        centroid=centroid,
        bbox=bbox,
        dispersion=dispersion
    )
