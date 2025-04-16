# Created by MacBook Pro at 01.04.25


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
from src.neural import models
from src.utils import chart_utils
import config
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


##############################
# 1. Data Generation and Dataset
##############################
class AngleSegmentationDataset(Dataset):
    def __init__(self, angle_sequences, label_sequences):
        """
        angle_sequences: list of 1-D torch.Tensor of tangent angles (in degrees)
        label_sequences: list of 1-D torch.Tensor of labels:
                         0 for "circle" (arc),
                         1 for one radial (line) segment,
                         2 for the other radial (line) segment.
        """
        self.angle_sequences = angle_sequences
        self.label_sequences = label_sequences

    def __len__(self):
        return len(self.angle_sequences)

    def __getitem__(self, idx):
        return self.angle_sequences[idx], self.label_sequences[idx]


def generate_synthetic_sample(min_r, max_r):
    """
    1. Create an empty black image (1024x1024).
    2. Draw a white filled circle at the center.
    3. Draw a black filled triangle with one vertex at the circle center
       and two vertices chosen randomly outside the circle.
    4. Find the contour of the remaining white area.
    5. Compute the tangent angles along the contour using finite differences.
    6. Label each contour point: if the distance to the center is near the circle radius, label it 0 ("circle"), otherwise 1 ("line").
    7. Randomly rotate the contour (and labels) to remove ordering bias.

    Returns:
       img: The original 1024x1024 image (numpy array, uint8) showing the circle and triangle.
       contour: The extracted contour (numpy array of shape [N,2]).
       angles_tensor: 1-D torch.Tensor of tangent angles (in degrees).
       labels_tensor: 1-D torch.Tensor of labels (0 for circle, 1 for line).
    """
    # Image parameters.
    img_size = 1024
    center = (img_size // 2, img_size // 2)

    # Create a black image.
    img = np.zeros((img_size, img_size), dtype=np.uint8)

    # Choose a random radius (ensuring the circle fits well inside the image).
    r = random.randint(min_r, max_r)

    # Draw a white filled circle.
    cv2.circle(img, center, r, 255, thickness=-1)

    # Helper to pick a random vertex outside the circle.
    def random_vertex():
        angle = random.uniform(0, 360)
        # Choose a distance greater than r (with some margin).
        dist = random.uniform(r + 50, r + 150)
        x = int(center[0] + dist * math.cos(math.radians(angle)))
        y = int(center[1] + dist * math.sin(math.radians(angle)))
        return (x, y)

    vertex1 = random_vertex()
    vertex2 = random_vertex()
    triangle_pts = np.array([center, vertex1, vertex2], dtype=np.int32)

    # Draw the black filled triangle.
    cv2.fillPoly(img, [triangle_pts], 0)

    # Find contours of the white area.
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise RuntimeError("No contour found!")
    # Assume the largest contour is our object.
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze(1)  # shape: (N,2)

    # Compute tangent angles using the neighboring points (finite differences).
    N = contour.shape[0]
    tangent_angles = []
    for i in range(N):
        p_prev = contour[(i - 1) % N]
        p_next = contour[(i + 1) % N]
        dx = float(p_next[0] - p_prev[0])
        dy = float(p_next[1] - p_prev[1])
        angle = math.degrees(math.atan2(dy, dx)) % 360
        tangent_angles.append(angle)
    tangent_angles = np.array(tangent_angles, dtype=np.float32)

    # Label each point:
    labels = []
    for (x, y) in contour:
        dist = math.hypot(x - center[0], y - center[1])
        # If the distance is within 15% of r (i.e. near the circle boundary), label as circle (0)
        if dist >= 0.85 * r:
            labels.append(0)
        else:
            labels.append(1)
    labels = np.array(labels, dtype=np.int64)

    # Randomly rotate (circular shift) the contour and labels.
    shift = random.randint(0, N - 1)
    tangent_angles = np.roll(tangent_angles, shift)
    labels = np.roll(labels, shift)

    # Convert angles and labels to torch tensors.
    angles_tensor = torch.tensor(tangent_angles, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return img, contour, angles_tensor, labels_tensor


def draw_train_image(img, contour, angles_tensor):
    # 1. Show the original image (circle with triangle cut-out).
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title("Original Image (Circle and Triangle)")
    plt.axis("off")

    # 2. Show the contour image.
    # Convert the original grayscale image to BGR so we can draw in color.
    img_contour = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    # Draw the extracted contour in red.
    cv2.drawContours(img_contour, [contour], -1, (0, 0, 255), thickness=2)
    # Convert BGR to RGB for matplotlib.
    img_contour_rgb = cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_contour_rgb)
    plt.title("Contour Image")
    plt.axis("off")

    # 3. Plot a line chart of the tangent angles.
    plt.figure(figsize=(10, 4))
    plt.plot(angles_tensor.numpy(), marker='o', linestyle='-')
    plt.title("Tangent Angles Along the Contour")
    plt.xlabel("Contour Point Index")
    plt.ylabel("Tangent Angle (degrees)")
    plt.grid(True)

    plt.show()


def gen_dataset(num_samples=10000):
    """
    For each example, simulate an object whose boundary is composed of:
      - A superior arc (circle segment) from the circle,
      - Two radial segments (line segments) from the circumference to the center.

    The center is fixed at (512,512) in a 1024×1024 image.

    Tangent angles are computed as follows:
      - For the arc: tangent = (θ + 90) mod 360.
      - For each radial segment: use the constant direction.

    Labels are assigned with three classes:
      - 0: arc (circle)
      - 1: radial segment from center to P2 (line)
      - 2: radial segment from P1 to center (line)

    **Random Rotation:**
    Since the outline is a closed loop, we randomly rotate the full sequence
    so that the starting point is random. This prevents the model from always
    seeing the arc first.
    """
    angle_sequences = []
    label_sequences = []
    for _ in range(num_samples):
        img, contour, angles_tensor, labels_tensor = generate_synthetic_sample(min_r=30, max_r=150)
        # draw_train_image(img, contour, angles_tensor
        angle_sequences.append(angles_tensor)
        label_sequences.append(labels_tensor)

    return AngleSegmentationDataset(angle_sequences, label_sequences)


def compute_curvature(contour):
    """
    Compute curvature at each contour point using a finite-difference approximation.
    For each point, the curvature is computed as:
      curvature = |delta_tangent_angle| / (average arc length)
    where delta_tangent_angle is the difference between the tangent angles at neighboring points.

    Args:
      contour: numpy array of shape (N,2) representing the contour points.
    Returns:
      curvature: numpy array of shape (N,) with curvature values.
    """
    N = contour.shape[0]
    curvature = np.zeros(N, dtype=np.float32)
    for i in range(N):
        p_prev = contour[(i - 1) % N]
        p = contour[i]
        p_next = contour[(i + 1) % N]

        # Vectors from previous to current and current to next.
        v1 = p - p_prev
        v2 = p_next - p

        # Compute tangent angles.
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        dtheta = angle2 - angle1
        # Normalize angle difference to [-pi, pi].
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi

        # Compute average arc length.
        arc_length = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2.0
        if arc_length > 1e-5:
            curvature[i] = abs(dtheta) / arc_length
        else:
            curvature[i] = 0.0
    return curvature


##############################
# 2. Model and Helper Functions
##############################
def angle_to_features(angles):
    """
    Convert a 1-D tensor of angles (in degrees) to features [sin(angle), cos(angle)].
    Output shape: [T, 2]
    """
    angles_rad = angles * math.pi / 180.0
    return torch.stack((torch.sin(angles_rad), torch.cos(angles_rad)), dim=-1)


class AngleSegmenter(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, num_classes=3):
        """
        A BiLSTM network for per-time-step classification.
        We now predict 3 classes: 0 (circle), 1 (line segment #1), 2 (line segment #2).
        """
        super(AngleSegmenter, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)


    def forward(self, x):
        # x: [B, T, input_dim]
        print(f"deivce of lstm:{next(self.lstm.parameters()).device}")
        print(f"deivce of x:{x.device}")
        out, _ = self.lstm(x)
        logits = self.fc(out)  # [B, T, num_classes]
        return logits


def collate_fn(batch):
    """
    Collate function to pad sequences.
    Uses -100 as the padding value for labels (ignored in loss computation).
    """
    angles_list, labels_list = zip(*batch)
    angles_padded = torch.nn.utils.rnn.pad_sequence(angles_list, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    return angles_padded, labels_padded


##############################
# 3. Training, Inference, and Testing
##############################
def train_model(model, dataloader, num_epochs=10, lr=1e-3, device='cpu'):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for angles, labels in dataloader:
            angles = angles.to(device)
            labels = labels.to(device)
            features = angle_to_features(angles)  # [B, T, 2]
            optimizer.zero_grad()
            logits = model(features)  # [B, T, 3]
            loss = criterion(logits.view(-1, 3), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
    return model


def predict_segments(model, angles, device='cpu'):
    """
    Given a 1-D tensor of tangent angles (in degrees), predict labels for each index.
    Then group contiguous indices into segments.
    In the post-processing, labels 1 and 2 are both mapped to "line" (but remain separate if predicted discontinuously).
    Returns:
      segments: list of lists (each sublist contains indices belonging to a segment)
      segment_labels: list of string labels ("circle" or "line") for each segment.
    """
    model.eval()
    with torch.no_grad():
        angles = angles.to(device)
        features = angle_to_features(angles).unsqueeze(0)  # [1, T, 2]
        logits = model(features.to(device))  # [1, T, 3]
        predictions = torch.argmax(logits, dim=-1).squeeze(0)  # [T]
    segments = []
    segment_labels = []
    current_segment = [0]
    current_label = predictions[0].item()
    for i in range(1, len(predictions)):
        if predictions[i].item() == current_label:
            current_segment.append(i)
        else:
            # Map label 0 -> "circle", 1 or 2 -> "line"
            seg_label = "circle" if current_label == 0 else "line"
            segments.append(current_segment)
            segment_labels.append(seg_label)
            current_segment = [i]
            current_label = predictions[i].item()
    seg_label = "circle" if current_label == 0 else "line"
    segments.append(current_segment)
    segment_labels.append(seg_label)
    return segments, segment_labels


def train_line_arc_detectors(dataset, num_epochs=10, lr=1e-3, batch_size=8, device='cpu', model_path=None):
    """
    Train the angle segmentation model on the provided dataset.
    If model_path is provided and a file exists, load the model instead of training.
    Returns a detector function that takes a 1-D tensor of tangent angles as input
    and outputs (segments, segment_labels).
    """
    model = AngleSegmenter().to(device)
    if model_path is not None and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        model = train_model(model, dataloader, num_epochs=num_epochs, lr=lr, device=device)
        if model_path is not None:
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")


    def detector(angles, device):
        return predict_segments(model, angles, device=device)

    detector.model = model
    return detector


def test_model_accuracy(detector, dataset, device='cpu'):
    """
    Evaluate per-time-step accuracy of the detector’s underlying model on a test dataset.
    Returns the accuracy as a fraction.
    """
    total_correct = 0
    total_count = 0
    model = detector.model.to(device)
    model.eval()
    with torch.no_grad():
        for angles, labels in dataset:
            angles = angles.to(device)
            features = angle_to_features(angles).unsqueeze(0).to(device)  # [1, T, 2]
            logits = model(features.to(device))  # [1, T, 3]
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu()  # [T]
            total_correct += (preds == labels).sum().item()
            total_count += len(labels)
    return total_correct / total_count


def get_detector(args):
    data = gen_dataset()
    # print(f"device is {args.device}")
    detector = train_line_arc_detectors(data, num_epochs=10, lr=1e-3, batch_size=8, device=args.device,
                                        model_path=config.models / "line_arc_detector.pth")

    # Generate a test dataset.
    test_dataset = gen_dataset(num_samples=20)
    accuracy = test_model_accuracy(detector, test_dataset, device=args.device)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Test on a new example.
    # For demonstration, generate a new synthetic outline.
    test_angles, _ = test_dataset[0]
    segments, segment_labels = detector(test_angles, args.device)
    chart_utils.show_line_chart(test_angles, file_name=config.output / "angles.png")

    print("Predicted segments and labels:")
    for seg, label in zip(segments, segment_labels):
        print(f"Label: {label}, Indices: {len(seg)}")

    return detector


def rgb2bw(img):
    bw_img = np.array(Image.fromarray(img.to("cpu").numpy().astype('uint8')).convert("L"))
    bw_img[bw_img == 211] = 0
    bw_img[bw_img > 0] = 1
    return bw_img


def test_bw_img(bw_img):
    """
    Given a black-and-white image (bw_img) with a white object on a black background,
    this function:
      1. Extracts the largest contour.
      2. Computes tangent angles at each contour point via finite differences.
      3. Computes the curvature at each contour point.
      4. Plots:
         - The input BW image.
         - The image with the extracted contour drawn over it.
         - A line chart of tangent angles.
         - A line chart of curvature.

    Args:
      bw_img: a numpy array (uint8) representing the BW image.

    Returns:
      contour: numpy array of shape (N,2) of the extracted contour points.
      tangent_angles: numpy array of shape (N,) containing the tangent angles (in degrees).
      curvature: numpy array of shape (N,) containing the curvature values.
    """
    # Find contours in the BW image.
    contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise RuntimeError("No contour found in the input image!")
    # Choose the largest contour.
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze(1)  # shape: (N, 2)

    N = contour.shape[0]
    tangent_angles = []
    for i in range(N):
        p_prev = contour[(i - 1) % N]
        p_next = contour[(i + 1) % N]
        dx = float(p_next[0] - p_prev[0])
        dy = float(p_next[1] - p_prev[1])
        angle = math.degrees(math.atan2(dy, dx)) % 360
        tangent_angles.append(angle)
    tangent_angles = np.array(tangent_angles, dtype=np.float32)

    # Compute curvature.
    curvature = compute_curvature(contour)

    # # Plot 1: The original BW image.
    # plt.figure(figsize=(8, 8))
    # plt.imshow(bw_img, cmap='gray')
    # plt.title("Input BW Image")
    # plt.axis("off")
    #
    # # Plot 2: The extracted contour overlaid on the original image.
    # img_contour = cv2.cvtColor(bw_img.copy(), cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img_contour, [contour], -1, (0, 0, 255), thickness=2)
    # img_contour_rgb = cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(img_contour_rgb)
    # plt.title("Extracted Contour")
    # plt.axis("off")
    #
    # # Plot 3: Tangent angles along the contour.
    # plt.figure(figsize=(10, 4))
    # plt.plot(tangent_angles, marker='o', linestyle='-')
    # plt.title("Tangent Angles Along the Contour")
    # plt.xlabel("Contour Point Index")
    # plt.ylabel("Tangent Angle (degrees)")
    # plt.grid(True)
    #
    # # Plot 4: Curvature along the contour.
    # plt.figure(figsize=(10, 4))
    # plt.plot(curvature, marker='x', linestyle='-', color='purple')
    # plt.title("Curvature Along the Contour")
    # plt.xlabel("Contour Point Index")
    # plt.ylabel("Curvature (1/pixels)")
    # plt.grid(True)
    #
    # plt.show()

    return contour, tangent_angles, curvature


def post_process_segments(segments, segment_labels, curvature, line_split_threshold=0.05, min_points=5):
    """
    Given:
      - segments: list of lists of indices (each list corresponds to a contiguous segment)
      - segment_labels: list of labels for each segment ("line" or "circle")
      - curvature: numpy array (length = total number of contour points) with curvature values
      - line_split_threshold: curvature threshold above which a "line" segment is considered to contain
                              more than one line.
      - min_points: minimum number of points a segment must have to be kept.

    The function does the following:
      1. For each segment labeled "line", it checks for high curvature peaks. If such peaks exist,
         the segment is split at the middle of each contiguous high-curvature run.
      2. Then, adjacent segments labeled "circle" are merged.
      3. After splitting and merging, segments with fewer than min_points are discarded.

    Returns:
      new_segments, new_segment_labels
    """
    new_segments = []
    new_labels = []

    # Step 1: Process "line" segments by splitting if necessary.
    for seg, label in zip(segments, segment_labels):
        if label == "line":
            seg_curv = np.array([curvature[i] for i in seg])
            high_indices = np.where(seg_curv > line_split_threshold)[0]
            if len(high_indices) == 0:
                if len(seg) >= min_points:
                    new_segments.append(seg)
                    new_labels.append(label)
            else:
                # Group consecutive high curvature indices.
                groups = []
                current_group = [high_indices[0]]
                for idx in high_indices[1:]:
                    if idx == current_group[-1] + 1:
                        current_group.append(idx)
                    else:
                        groups.append(current_group)
                        current_group = [idx]
                groups.append(current_group)
                # Use the middle index of each group as a split point.
                split_points = [group[len(group) // 2] for group in groups]
                subsegments = []
                start = 0
                for sp in split_points:
                    subsegments.append(seg[start:sp + 1])
                    start = sp + 1
                if start < len(seg):
                    subsegments.append(seg[start:])
                # Append subsegments only if they have at least min_points.
                for sub in subsegments:
                    if len(sub) >= min_points:
                        new_segments.append(sub)
                        new_labels.append(label)
        else:
            # For circle segments, keep them only if they have enough points.
            if len(seg) >= min_points:
                new_segments.append(seg)
                new_labels.append(label)

    # Step 2: Merge adjacent "circle" segments.
    merged_segments = []
    merged_labels = []
    for seg, label in zip(new_segments, new_labels):
        if merged_segments and label == "circle" and merged_labels[-1] == "circle":
            merged_segments[-1] = merged_segments[-1] + seg
        else:
            merged_segments.append(seg)
            merged_labels.append(label)
    # Handle wrap-around merge if needed.
    if len(merged_segments) > 1 and merged_labels[0] == "circle" and merged_labels[-1] == "circle":
        merged_segments[0] = merged_segments[-1] + merged_segments[0]
        merged_segments.pop()
        merged_labels.pop()

    # Final filtering: remove segments with fewer than min_points.
    final_segments = []
    final_labels = []
    for seg, label in zip(merged_segments, merged_labels):
        if len(seg) >= min_points:
            final_segments.append(seg)
            final_labels.append(label)

    return final_segments, final_labels
