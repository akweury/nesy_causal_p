# Created by X at 12.03.25

import numpy as np
from src.beta.beta_config import ENCODE_ATTRIBUTES

def encode_features(symbolic, neuro):
    """
    Encodes the symbolic features into a fixed-size object-centric matrix using a configurable schema.

    The encoding scheme is defined in 'encode_config.py' (ENCODE_ATTRIBUTES) and specifies:
      - The order of attributes.
      - How to extract and transform each attribute from the object dictionary.

    For example, attributes can include: x, y, size, object number, color (r, g, b), and shape indicators.

    Args:
        symbolic (list): A list of dictionaries. Each dictionary represents one object and
                         should contain keys like "x", "y", "size", "color", "shape", etc.
        neuro: Additional neuro features (passed through unchanged).

    Returns:
        dict: A dictionary with:
              - "object_matrix": a NumPy array of shape (max_objects, feature_dim) containing the encoded features.
              - "neuro_features": the neuro features (unchanged).
    """
    # Load the attribute configuration.
    attributes = ENCODE_ATTRIBUTES
    feature_dim = len(attributes)
    max_objects = 20  # Fixed number of objects per image.

    object_features = []
    for obj in symbolic:
        feature_vector = []
        for attr in attributes:
            key = attr["key"]
            default_val = attr.get("default", 0)
            transform = attr.get("transform", lambda x: x)
            # Retrieve the raw value using the specified key, or use the default.
            raw_value = obj.get(key, default_val)
            # Apply transformation.
            value = transform(raw_value)
            feature_vector.append(value)
        object_features.append(feature_vector)

    # Pad with zeros if there are fewer than max_objects; trim if more.
    num_objects = len(object_features)
    if num_objects < max_objects:
        padding = [[0] * feature_dim for _ in range(max_objects - num_objects)]
        object_features.extend(padding)
    else:
        object_features = object_features[:max_objects]

    # Convert to NumPy array.
    object_matrix = np.array(object_features)

    return {"symbolic_features": object_matrix, "neuro_features": neuro}

def beta_encoder(features):
    sample_num = len(features["positive"]["symbolic"])

    encoded_features = {}
    for polarity in ["positive", "negative"]:
        encoded_features[polarity] = {"symbolic": [], "neuro": []}
        for sample_i in range(sample_num):
            symbolic_features = features[polarity]["symbolic"][sample_i]
            neuro_features = features[polarity]["neuro"][sample_i]
            encoded_polarity_features = encode_features(symbolic_features, neuro_features)
            encoded_features[polarity]["symbolic"].append(encoded_polarity_features["symbolic_features"])
            encoded_features[polarity]["neuro"].append(encoded_polarity_features["neuro_features"])

    return encoded_features
