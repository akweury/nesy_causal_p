# Created by MacBook Pro at 17.04.25


import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
from patch_set_model import PatchSetClassifier


def generate_random_patch_sets(self, contour):
    N = contour.size(0)
    if N < self.patch_size:
        return []

    patch_start_indices = torch.linspace(0, N - self.patch_size, steps=self.num_patches, dtype=torch.long)
    patches = [contour[i:i + self.patch_size] for i in patch_start_indices]

    patch_sets = []
    for _ in range(self.num_sets):
        if len(patches) < self.set_size:
            continue
        idxs = torch.randperm(len(patches))[:self.set_size]
        patch_set = torch.stack([patches[i] for i in idxs])
        patch_sets.append(patch_set)
    return patch_sets


def load_classifier(model_path, input_dim, hidden_dim, num_classes, device):
    model = PatchSetClassifier(input_dim, hidden_dim, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_objects_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.where(gray == 211, 0, 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return labels, stats, centroids

def extract_contour_tensor(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    contour = max(contours, key=len).squeeze()
    if contour.ndim != 2 or contour.shape[0] < 5:
        return None
    contour = contour - contour.mean(axis=0)
    max_dist = np.linalg.norm(contour, axis=1).max()
    contour = contour / (2 * max_dist + 1e-6)
    return torch.tensor(contour, dtype=torch.float32)

def predict_shape_from_contour(contour_tensor, model, device, patch_size=5, set_size=3, num_sets=10):
    patch_sets = generate_random_patch_sets(contour_tensor, patch_size=patch_size, set_size=set_size, num_sets=num_sets)
    if len(patch_sets) == 0:
        return None
    batch = torch.stack(patch_sets).to(device)  # [num_sets, set_size, patch_size, 2]
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).mean(dim=0)  # average voting
    return probs

def test_on_image(image_path, model_path, device='cpu'):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    labels, stats, _ = extract_objects_from_image(image)

    input_dim = 3 * 5 * 2
    model = load_classifier(model_path, input_dim, hidden_dim=128, num_classes=3, device=device)

    shape_names = ["triangle", "rectangle", "ellipse"]
    pred_labels = []

    for i in range(1, np.max(labels)+1):  # skip background
        mask = (labels == i).astype(np.uint8)
        contour_tensor = extract_contour_tensor(mask)
        if contour_tensor is None:
            pred_labels.append("unknown")
            continue
        probs = predict_shape_from_contour(contour_tensor, model, device)
        if probs is None:
            pred_labels.append("unknown")
        else:
            pred_labels.append(shape_names[probs.argmax().item()])

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    for i, label in enumerate(pred_labels):
        x, y, w, h, _ = stats[i]
        ax.text(x, y - 5, label, color="red", fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
    ax.set_title("Patch Classifier Prediction")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


