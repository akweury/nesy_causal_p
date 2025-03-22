# Created by X at 12.03.25

import torch
import numpy as np

from src.beta.dataset.image_utils import load_image, load_annotation


def detect_features(image, annotation):
    """
    Detects both symbolic and neuro features from the image and its annotation.

    Symbolic features are obtained from the annotation's "img_data" field,
    which contains a list of object properties. Neuro features are derived by
    resizing the image to the specified resolution (default 224) and converting
    it into a NumPy array.

    Args:
        image (PIL.Image): The input image.
        annotation (dict): The annotation data.

    Returns:
        tuple: (symbolic_features, neuro_features)
            symbolic_features: A list of object properties from the annotation.
            neuro_features: A NumPy array representing the resized image.
    """
    # Extract symbolic features from the annotation.
    symbolic_features = annotation.get("img_data", [])

    # Resize image based on the provided resolution (default: 224).
    resolution = annotation.get("resolution", 224)
    image_resized = image.resize((resolution, resolution))
    neuro_features = np.array(image_resized)

    return symbolic_features, neuro_features


def beta_detector(task):
    positive_examples, negative_examples = task["positive"], task["negative"]
    sample_num = len(positive_examples["images"])
    features = {}
    for polarity in ["positive", "negative"]:
        features[polarity] = {"symbolic": [], "neuro": []}
        for sample_i in range(sample_num):
            image = load_image(positive_examples["images"][sample_i])
            annotation = load_annotation(positive_examples["annotations"][sample_i])
            symbolic, neuro = detect_features(image, annotation)
            features[polarity]["symbolic"].append(symbolic)
            features[polarity]["neuro"].append(neuro)
    return features
