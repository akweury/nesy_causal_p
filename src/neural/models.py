# Created by jing at 11.12.24

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import F for functional operations
import cv2
import numpy as np
from PIL import Image, ImageDraw
import math

from torch.utils.data import DataLoader, TensorDataset
from src.neural.neural_utils import *
from src.utils import chart_utils, data_utils

import config
from src import bk


# Define the Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, in_channels):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # (in, 64, 64) -> (32, 32, 32)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (32, 32, 32) -> (16, 16, 16)
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (16, 16, 16) -> (8, 8, 8)
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Latent space
        self.latent = nn.Linear(16 * 8 * 8, 128)  # Example latent size: 128

        # Decoder
        self.decoder_fc = nn.Linear(128, 16 * 8 * 8)
        self.decoder = nn.Sequential(
            # (16, 8, 8) -> (32, 16, 16)
            nn.ConvTranspose2d(16, 32,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # (32, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(32, 64,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # (64, 32, 32) -> (in, 64, 64)
            nn.ConvTranspose2d(64, in_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.latent(x)

        # Decoder
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 16, 8, 8)
        x = self.decoder(x)
        return x


def train_autoencoder(args, bk_shapes):
    args.step_counter += 1
    args.logger.info(f"Step {args.step_counter}/{args.total_step}: "
                     f"Training Autoencoder for patterns {bk_shapes}.")

    for bk_shape in bk_shapes:
        save_path = config.output / f"{bk_shape}"
        os.makedirs(save_path, exist_ok=True)
        model_file = save_path / "fm_ae.pth"
        ae_fm_file = save_path / "fm_ae.pt"
        train_loader, fm_channels = prepare_fm_data(args)
        if not os.path.exists(model_file):
            # Initialize the model, loss, and optimizer
            model = Autoencoder(fm_channels)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            epochs = 20
            loss_history = []
            for epoch in range(epochs):
                epoch_loss = 0
                for batch, in train_loader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(batch)

                    # Compute the loss
                    loss = criterion(outputs, batch)

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                # visual
                img_file = save_path / f"train_output.png"
                original_img = batch[0].sum(dim=0)
                output_img = outputs[0].sum(dim=0)
                visual_ae_compare(original_img, output_img, img_file)
                avg_loss = epoch_loss / len(train_loader)
                loss_history.append(avg_loss)
                args.logger.debug(f"Train AE ({bk_shape}) "
                                  f"Epoch [{epoch + 1}/{epochs}], "
                                  f"Loss: {avg_loss:.4f}")

            # Save the trained model
            torch.save(model.state_dict(), save_path / "fm_ae.pth")
            args.logger.info("Feature map autoencoder is saved as 'fm_ae.pth'")

            # Visualize the training history
            visual_train_history(save_path, epochs, loss_history)

            args.logger.debug(f"Training {bk_shape} autoencoder completed!")
        if not os.path.exists(ae_fm_file):
            # Test the trained model
            model = Autoencoder(fm_channels)
            model.load_state_dict(torch.load(model_file))
            model.eval()
            for b_i, (batch,) in enumerate(train_loader):
                test_output = model(batch)

                img_file = save_path / f"ae_test_{b_i}.png"
                original_img = batch[0].sum(dim=0)
                output_img = test_output[0].sum(dim=0)
                visual_ae_compare(original_img, output_img, img_file)


def one_layer_conv(data, kernels):
    data = data.to(torch.float32)
    kernels = kernels.to(torch.float32)
    if kernels.shape[-1] == 3:
        padding = 1
    elif kernels.shape[-1] == 5:
        padding = 2
    elif kernels.shape[-1] == 7:
        padding = 3
    elif kernels.shape[-1] == 9:
        padding = 4
    else:
        raise ValueError("kernels has to be 3/5/7/9 dimensional")
    output = F.conv2d(data, kernels, stride=1, padding=padding)
    # max_value = kernels.sum(dim=[1, 2, 3])
    # max_value = max_value.unsqueeze(1).unsqueeze(2).unsqueeze(0)
    # max_value = torch.repeat_interleave(max_value, output.shape[2], dim=-2)
    # max_value = torch.repeat_interleave(max_value, output.shape[3], dim=-1)
    # mask = (max_value == output).to(torch.float32)
    output = output / 9
    return output


def crop_img(img, crop_data=None):
    rgb = img.numpy().astype(np.uint8)
    bg_mask = np.all(rgb == bk.color_matplotlib["lightgray"], axis=-1)
    rgb[bg_mask] = [0, 0, 0]
    bw_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    bw_img = torch.from_numpy(bw_img).squeeze()
    if crop_data is None:
        height, width = bw_img.shape[-2], bw_img.shape[-1]
        # Find the bounding box of the nonzero values
        nonzero_coords = torch.nonzero(bw_img)
        if nonzero_coords.numel() == 0:  # Handle completely empty images
            return bw_img, [0, 0, 0, 0]

        min_y, min_x = nonzero_coords.min(dim=0).values
        max_y, max_x = nonzero_coords.max(dim=0).values

        # Compute the side length of the square
        side_length = max(max_y - min_y + 1, max_x - min_x + 1)

        # Adjust the bounding box to make it square
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2
        half_side = side_length // 2 + 5

        # Compute the new square bounding box
        new_min_y = max(center_y - half_side, 0)
        new_max_y = min(center_y + half_side + 1, height)
        new_min_x = max(center_x - half_side, 0)
        new_max_x = min(center_x + half_side + 1, width)
    else:
        new_min_y, new_max_y, new_min_x, new_max_x = crop_data
    # Crop the image
    cropped_image = bw_img[new_min_y:new_max_y, new_min_x:new_max_x]

    # if resize is not None:
    #     cropped_image = cv2.resize(cropped_image.numpy(), (resize, resize),
    #                                interpolation=cv2.INTER_AREA)
    #     cropped_image = torch.from_numpy(cropped_image)
    cropped_image = cropped_image.unsqueeze(0)
    return cropped_image, [new_min_y, new_max_y, new_min_x, new_max_x]


def resize_img(img, resize):
    # rgb = rgb_np.numpy().astype(np.uint8)
    # bw_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # bw_img[bw_img != 211] = 1
    # bw_img[bw_img == 211] = 0
    # if crop:
    #     # bw image to cropped bw image
    #     bw_img, _ = crop_img(torch.from_numpy(bw_img).squeeze(), resize=resize)
    # else:
    #     if resize:
    img = img.squeeze().numpy()
    resized_img = cv2.resize(img, (resize, resize),
                             interpolation=cv2.INTER_LINEAR)
    resized_img = torch.from_numpy(resized_img).unsqueeze(0)
    # else:
    #     bw_img = torch.from_numpy(bw_img).unsqueeze(0)
    return resized_img


def to_bw_img(image):
    # Load an image
    image[image > 0] = 1
    return image


def img2bw(img, cropped_data=None, resize=16):
    cropped_img, cropped_data = crop_img(img.squeeze(), crop_data=cropped_data)
    resized_img = resize_img(cropped_img, resize=resize)
    bw_img = to_bw_img(resized_img)
    return bw_img, cropped_data


def img2fm(img, kernels, cropped_data=None):
    bw_img, cropped_data = img2bw(img, cropped_data)
    fms = one_layer_conv(bw_img, kernels)
    if fms.ndim == 3:
        fms = fms.unsqueeze(0)
    return fms, cropped_data


def fm_merge(fms):
    if fms.ndim == 3:
        in_fms = fms.sum(dim=0).squeeze()
    elif fms.ndim == 4:
        in_fms = fms.sum(dim=1).squeeze()
    else:
        raise ValueError
    merged_fm = (in_fms - in_fms.min()) / ((in_fms.max() - in_fms.min()) + 1e-20)
    return merged_fm


def calculate_circular_variance(angles, wrap_size=20):
    # Convert degrees to radians
    angles_rad = angles * math.pi / 180
    # Wrap the data for circular variance calculation
    wrapped_radians = torch.cat([angles_rad, angles_rad[:wrap_size - 1]])  # Wrap around

    # Function to calculate circular variance
    def circular_variance(values):
        sin_vals = torch.sin(values)
        cos_vals = torch.cos(values)
        mean_cos = torch.mean(cos_vals)
        mean_sin = torch.mean(sin_vals)
        R = torch.sqrt(mean_cos ** 2 + mean_sin ** 2)
        return 1 - R

    # Compute circular variance for each window
    variances = []
    for i in range(len(angles_rad)):
        window = wrapped_radians[i:i + wrap_size]  # Get a 5-value window
        variance = circular_variance(window)
        variances.append(variance)

    # Convert variances to a tensor
    variances = torch.tensor(variances)
    return variances


def extract_line_curves(angles, seg_var_th=1e-8):
    """
    Segments a series of angles into increasing, decreasing, or constant segments by checking the trends of future angles.

    Args:
        angles (array-like): Input angles in degrees.
        tolerance (float): Fluctuation tolerance to consider as 'constant' (in degrees).
        future_steps (tuple): Steps ahead to check (e.g., 5th and 10th angles).

    Returns:
        list of dict: Each dict contains the start index, end index, and segment type ('increasing', 'decreasing', 'constant').
    """

    circular_variances = calculate_circular_variance(angles)
    # chart_utils.show_line_chart(circular_variances, "circular_variances")

    all_seg = []
    check_list = []
    current_seg = []
    segment_labels = []
    future_step = 5
    # Find the first index where the smoothed trend stops decreasing
    for i in range(len(angles) - future_step):
        # post_variance = torch.var_mean(angles[i:i + future_step])[0]
        # if i < future_step:
        #     prev_variance = torch.var_mean(torch.cat((angles[-(future_step - i):], angles[:i])))[0]
        # else:
        #     prev_variance = torch.var_mean(angles[i - future_step:i])[0]
        if circular_variances[i] < 0.1:
            current_seg.append(i)
            check_list.append(i)
        else:
            if len(current_seg) > 10:
                seg_var = torch.var(circular_variances[current_seg[10:-10]])
                if seg_var < seg_var_th:
                    segment_labels.append("line")
                else:
                    segment_labels.append("circle")
                # chart_utils.show_line_chart(circular_variances[current_seg], "seg_var")
                # chart_utils.show_line_chart(angles[current_seg], "angles")
                # print("seg_var", seg_var)
                # print("")
                all_seg.append(current_seg)
            current_seg = []
            current_seg.append(i)
            check_list.append(i)
    if len(current_seg) > 10:
        all_seg.append(current_seg)
        seg_var = torch.var(circular_variances[current_seg[10:-10]])
        if seg_var < seg_var_th:
            segment_labels.append("line")
        else:
            segment_labels.append("circle")
    # Visual Segments
    # chart_utils.visual_multiple_segments(all_seg, angles)
    all_seg = [seg for seg in all_seg if len(seg) > 10]
    if (len(angles) - all_seg[-1][-1]) < 10 and all_seg[0][0] < 10 and segment_labels[0] == segment_labels[-1]:
        merged_seg = all_seg[-1] + all_seg[0]
        all_seg = all_seg[1:-1] + [merged_seg]
        segment_labels = segment_labels[1:-1] + [segment_labels[0]]
    if len(all_seg) != len(segment_labels):
        print("")
    return all_seg, segment_labels


def find_contours(input_array):
    # Find contours with all points along the edges
    contours, _ = cv2.findContours(input_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Extract contours as Nx2 arrays of pixel positions
    contour_coordinates = []
    for contour in contours:
        # Convert (x, y) to (row, col) matching the original NumPy array indexing
        coords = contour.squeeze(axis=1)  # Convert (N, 1, 2) to (N, 2)
        coords[:, [0, 1]] = coords[:, [1, 0]]  # Swap columns to match NumPy indexing
        contour_coordinates.append(coords)

    contour_points = [point[:, [1, 0]] for point in contour_coordinates]

    all_points = np.concatenate(contour_points)

    contour_img = np.zeros((input_array.shape[0], input_array.shape[0], 3))
    from src import bk
    for i in range(len(all_points)):
        pos = all_points[i]
        contour_img[pos[1], pos[0]] = [255, 255, 255]
    # chart_utils.van(contour_img)
    return contour_points


def calculate_dvs(contour_points):
    dvs = []
    for contour_list in contour_points:
        direction_vector = data_utils.contour_to_direction_vector(contour_list)
        direction_vector = torch.tensor(direction_vector)
        dvs.append(direction_vector)
    return dvs


def get_contour_segs(img):
    bw_img = np.array(Image.fromarray(img.numpy().astype('uint8')).convert("L"))
    bw_img[bw_img == 211] = 0
    bw_img[bw_img > 0] = 1

    # bw_img = resize_img(bw_img, 64)
    contour_points = find_contours(bw_img)

    dvs = calculate_dvs(contour_points)
    all_segments = []
    all_labels = []
    # find out the segments, and their labels (curve or line)
    for dv in dvs:
        segments, seg_labels = extract_line_curves(dv, seg_var_th=1e-8)
        all_segments.append(segments)
        all_labels.append(seg_labels)

    # visualize the segments on the original contour image
    # chart_utils.visual_labeled_contours(bw_img.shape[0], all_segments, contour_points, all_labels)
    return contour_points, all_segments, all_labels


def calculate_line_properties(positions):
    """
    Calculate the slope, start position, and end position of a line segment
    given a list of 2D positions.

    Args:
        positions (list of tuple): List of (x, y) positions.

    Returns:
        slope (float): Slope of the line segment.
        start_pos (tuple): Start position (x, y).
        end_pos (tuple): End position (x, y).
    """
    # Start and end positions
    start_pos = positions[0]
    end_pos = positions[-1]

    # Extract coordinates
    x1, y1 = start_pos
    x2, y2 = end_pos

    # Calculate slope
    if x2 - x1 == 0:  # Vertical line
        slope = 10  # Infinite slope
    else:
        slope = (y2 - y1) / (x2 - x1)
    if slope > 10:
        slope = 10
    return slope, start_pos, end_pos


def are_slopes_similar(slope1, slope2, slope_tolerance):
    """Check if two slopes are similar within a tolerance."""
    return abs(slope1 - slope2) <= slope_tolerance


def are_lines_collinear(line1, line2, slope_tolerance, distance_tolerance):
    """Check if two lines are collinear based on slope and distance from one line's points to the other."""
    if not are_slopes_similar(line1["slope"], line2["slope"], slope_tolerance):
        return False

    # Unpack line points
    x1, y1 = line1["start_point"]
    x2, y2 = line1["end_point"]
    x3, y3 = line2["start_point"]
    x4, y4 = line2["end_point"]

    # Calculate distance from line1 to points of line2
    def point_to_line_distance(px, py, x1, y1, x2, y2):
        numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return numerator / denominator

    dist1 = point_to_line_distance(x3, y3, x1, y1, x2, y2)
    dist2 = point_to_line_distance(x4, y4, x1, y1, x2, y2)

    return dist1 <= distance_tolerance and dist2 <= distance_tolerance


def draw_line_on_array(array, start, end, color):
    """
    Draw a line on a 2D PyTorch array between the start and end points.

    Args:
        array (torch.Tensor): A 2D tensor representing the grid.
        start (tuple): Start point (x1, y1) of the line.
        end (tuple): End point (x2, y2) of the line.
        value (int or float): The value to assign to the pixels of the line.

    Returns:
        torch.Tensor: The modified array with the line drawn.
    """
    x1, y1 = start
    x2, y2 = end

    # Bresenham's Line Algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        array[y1 - 1:y1 + 2, x1 - 1:x1 + 2] = torch.tensor(color)  # Set the pixel value at the current point

        if (x1, y1) == (x2, y2):
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return array


def is_collinear(p1, p2, p3, tolerance):
    """
    Check if three points are collinear (lie on the same line).

    Args:
        p1, p2, p3: Points as tuples (x, y).

    Returns:
        bool: True if the points are collinear.
    """
    x1, y1 = p1 / 1024
    x2, y2 = p2 / 1024
    x3, y3 = p3 / 1024
    # Compute the determinant for collinearity
    determinant = abs((y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1))
    return determinant <= tolerance


def fit_circle(points):
    """
    Fit a circle to a set of points using least squares.

    Args:
        points (list of tuple): List of (x, y) coordinates.

    Returns:
        center (tuple): (x, y) coordinates of the circle's center.
        radius (float): Radius of the circle.
    """
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Solve the circle equation: (x - x_c)^2 + (y - y_c)^2 = r^2
    A = np.c_[2 * x, 2 * y, np.ones(len(points))]
    b = x ** 2 + y ** 2
    sol = np.linalg.lstsq(A, b, rcond=None)[0]

    x_center, y_center, c = sol
    radius = np.sqrt(c + x_center ** 2 + y_center ** 2)
    return (x_center, y_center), radius


def calculate_arc_properties(points):
    """
    Calculate arc properties from a set of positions.

    Args:
        points (list of tuple): List of (x, y) coordinates defining the arc.

    Returns:
        dict: Arc properties including center, radius, start_angle, end_angle, and direction.
    """
    # Fit a circle to the points
    center, radius = fit_circle(points)

    # Calculate angles for start and end points
    start_angle = np.arctan2(points[0][1] - center[1], points[0][0] - center[0])
    end_angle = np.arctan2(points[-1][1] - center[1], points[-1][0] - center[0])

    # Convert to degrees
    start_angle_deg = np.degrees(start_angle) % 360
    end_angle_deg = np.degrees(end_angle) % 360

    # Determine direction
    angles = []
    for p in points:
        angle = np.arctan2(p[1] - center[1], p[0] - center[0])
        angles.append(np.degrees(angle) % 360)
    direction = "clockwise" if (np.diff(angles) < 0).mean() > 0.5 else "counterclockwise"

    return {
        "center": tuple(center),
        "radius": radius,
        "start_angle": start_angle_deg,
        "end_angle": end_angle_deg,
        "direction": direction,
    }


def draw_arc_on_image(image, center, radius, start_angle, end_angle, color=(255, 0, 0), thickness=2,
                      direction="counterclockwise"):
    """
    Draw an arc on an RGB image.

    Args:
        image (np.ndarray): RGB image as a NumPy array.
        center (tuple): Center of the arc (x, y).
        radius (int): Radius of the arc.
        start_angle (float): Start angle of the arc in degrees.
        end_angle (float): End angle of the arc in degrees.
        color (tuple): Color of the arc in RGB format.
        thickness (int): Thickness of the arc outline.

    Returns:
        np.ndarray: Image with the arc drawn.
    """
    # Use OpenCV's ellipse function to draw an arc
    x_center, y_center = center
    if direction == "clockwise":
        start_angle, end_angle = end_angle, start_angle
    start_angle_rad = np.radians(start_angle)  # Convert start angle to radians
    end_angle_rad = np.radians(end_angle)  # Convert end angle to radians

    # Generate angles for the arc
    if start_angle < end_angle:
        angles = np.linspace(start_angle_rad, end_angle_rad, 1000)
    else:
        angles = np.linspace(start_angle_rad, end_angle_rad + 2 * np.pi, 1000)

    # Calculate x and y coordinates of the arc
    x_coords = np.round(x_center + radius * np.cos(angles)).astype(int)
    y_coords = np.round(y_center + radius * np.sin(angles)).astype(int)

    # Draw the arc on the array
    for x, y in zip(x_coords, y_coords):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check bounds
            image[y - 1:y + 2, x - 1:x + 2] = color
    # chart_utils.van(image)
    return image


def merge_similar_lines(lines, slope_tolerance, distance_tolerance, vertical_th=8):
    used = np.zeros(len(lines))
    similar_lines = []


    for l_i, line in enumerate(lines):
        current_line = line
        if used[l_i]:
            continue
        if l_i == len(lines):
            if used[l_i] == 0:
                used[l_i] = 1
                similar_lines.append(line)
        for l2_i in range(l_i + 1, len(lines)):
            l2 = lines[l2_i]
            k1 = line[0]
            k2 = l2[0]
            slope_similar = np.abs(k1 - k2) < slope_tolerance
            all_vertical = np.abs(k2) >= vertical_th and np.abs(k1) >= vertical_th
            collinearity = is_collinear(current_line[1], current_line[2], l2[1], distance_tolerance)
            if (slope_similar or all_vertical) and collinearity:
                used[l_i] = 1
                used[l2_i] = 1
                # Update the start and end to cover both lines
                current_start = min(current_line[1], l2[1], key=lambda p: (p[0], p[1]))
                current_end = max(current_line[2], l2[2], key=lambda p: (p[0], p[1]))
                current_line = [current_line[0], current_start, current_end]
                similar_lines.append(current_line)
    return similar_lines


def get_curves(contour_points, contour_segs, contour_seg_labels, width):
    cir_segs = []
    for contour_i, contour_seg in enumerate(contour_segs):
        for seg_i, seg in enumerate(contour_seg):
            if contour_seg_labels[contour_i][seg_i] == "circle":
                cir_segs.append(contour_points[contour_i][seg])

    circles = []
    for cir_seg in cir_segs:
        cir_dict = calculate_arc_properties(cir_seg)
        circles.append(cir_dict)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 0, 128), (255, 0, 128)]
    # visual circles
    cir_img = np.zeros((width, width, 3), dtype=np.uint8)
    for c_i, cir in enumerate(circles):
        cir_img = draw_arc_on_image(cir_img, cir["center"], cir["radius"], cir["start_angle"], cir["end_angle"],
                                    colors[1], 2, cir["direction"])
    # chart_utils.van(cir_img)

    return circles


def get_line_groups(contour_points, contour_segs, contour_seg_labels, width):
    line_segs = []
    for contour_i, contour_seg in enumerate(contour_segs):
        for seg_i, seg in enumerate(contour_seg):
            if contour_seg_labels[contour_i][seg_i] == "line":
                line_segs.append(contour_points[contour_i][seg])
    lines = []
    for line_seg in line_segs:
        slope, start, end = calculate_line_properties(line_seg)
        lines.append([slope, start, end])

    # visual lines
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 0, 128), (255, 0, 128)]
    line_img = torch.zeros(width, width, 3)
    for l_i, line in enumerate(lines):
        line_img = draw_line_on_array(line_img, line[1], line[2], colors[0])
    # chart_utils.van(line_img)

    # merge all the line_segs
    merged_lines = merge_similar_lines(lines, slope_tolerance=0.9, distance_tolerance=1e-2, vertical_th=8)
    # visual lines
    merged_line_img = torch.zeros(width, width, 3)
    for l_i, line in enumerate(merged_lines):
        merged_line_img = draw_line_on_array(merged_line_img, line[1], line[2], colors[l_i])
    # chart_utils.van(merged_line_img)

    return merged_lines


def distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def forms_closed_figure(line_ends, max_distance=10):
    """
    Check if a set of lines forms a closed figure allowing a distance up to max_distance.

    Each line is represented as a tuple (slope, start_point, end_point)
    start_point and end_point are tuples representing coordinates (x, y)
    """
    # for each point, find a similar point in the rest point
    all_points = line_ends.reshape(-1, 2)
    closed = True
    for l_i in range(len(line_ends)):
        line = line_ends[l_i]
        dists = []
        for point in all_points:
            dists.append(distance(line[0], point))
        if not sorted(dists)[1] < max_distance:
            print(f"point1: {line[0]}, {sorted(dists)[1]}")
            closed = False
            break
        dists = []
        for point in all_points:
            dists.append(distance(line[1], point))
        if not sorted(dists)[1] < max_distance:
            print(f"point1: {line[0]}, {sorted(dists)[1]}")
            closed = False
            break

    return closed


def find_triangles(lines):
    # check closed lines
    # check slopes
    end_points = []
    for line in lines:
        end_points.append([line[1], line[2]])
    end_points = np.array(end_points)
    close = forms_closed_figure(end_points, max_distance=120)
    triangle = None

    if close and len(lines) == 3:
        triangle = True
    return triangle


def find_squares(lines):
    # check slopes
    end_points = []
    for line in lines:
        end_points.append([line[1], line[2]])
    end_points = np.array(end_points)
    close = forms_closed_figure(end_points, max_distance=120)
    square = None

    if close and len(lines) == 4:
        square = True
    return square


def find_circles(curves):
    raise NotImplementedError
    # check slopes
    end_points = []
    for line in curves:
        end_points.append([line[1], line[2]])
    end_points = np.array(end_points)
    close = forms_closed_figure(end_points, max_distance=120)
    circle = None


    return circle
