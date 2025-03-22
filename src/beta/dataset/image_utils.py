# Created by X at 12.03.25

import json
import numpy as np
from PIL import Image


def load_image(image_path):
    """
    Loads an image from the given file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image: The loaded image in RGB mode.
    """
    return Image.open(image_path).convert("RGB")


def load_annotation(annotation_path):
    """
    Loads annotation data from a JSON file.

    Args:
        annotation_path (str): Path to the JSON annotation file.

    Returns:
        dict: The annotation data.
    """
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    return annotation
