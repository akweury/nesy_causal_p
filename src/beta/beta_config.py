# Created by X at 11.03.25
import os
from pathlib import Path

root = Path(__file__).parents[2]
storage = root / 'storage'
output = storage / 'output'
os.makedirs(output, exist_ok=True)

DATASET_DIR = storage / "raw_patterns"
SPLITS = ["train", "test"]
OUTPUT_DIR = storage / "output"
GESTALT_PRINCIPLES = ["proximity", "similarity", "closure", "symmetry", "continuity"]


# Define the encoding configuration for each attribute.
# Each entry in the list represents one column in the encoded feature vector.
# - name: a human-readable attribute name.
# - key: the key used to retrieve the raw value from the object's dictionary.
# - default: a default value if the key is not present.
# - transform: a function that converts the raw value into the encoded form.
ENCODE_ATTRIBUTES = [
    {
        "name": "x",
        "key": "x",
        "default": 0,
        "transform": lambda x: x
    },
    {
        "name": "y",
        "key": "y",
        "default": 0,
        "transform": lambda y: y
    },
    {
        "name": "size",
        "key": "size",
        "default": 0,
        "transform": lambda s: s
    },
    {
        "name": "obj_number",
        "key": "obj_number",
        "default": 1,
        "transform": lambda n: n
    },
    {
        "name": "color_r",
        "key": "color_r",
        "default": 0,
        "transform": lambda col: col
    },
    {
        "name": "color_g",
        "key": "color_g",
        "default": 0,
        "transform": lambda col: col
    },
    {
        "name": "color_b",
        "key": "color_b",
        "default": 0,
        "transform": lambda col: col
    },
    {
        "name": "is_triangle",
        "key": "shape",
        "default": 0,
        "transform": lambda s: 1 if str(s).lower() == "triangle" else 0
    },
    {
        "name": "is_square",
        "key": "shape",
        "default": 0,
        "transform": lambda s: 1 if str(s).lower() == "square" else 0
    },
    {
        "name": "is_circle",
        "key": "shape",
        "default": 0,
        "transform": lambda s: 1 if str(s).lower() == "circle" else 0
    },
]

def get_attribute_index(attribute_name):
    """
    Returns the index of the attribute in the encoded feature vector
    by searching the configuration.
    """
    for idx, attr in enumerate(ENCODE_ATTRIBUTES):
        if attr["name"] == attribute_name:
            return idx
    return None