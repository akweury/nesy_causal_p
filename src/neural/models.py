# Created by Xxx at 11.12.24

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Import F for functional operations
import cv2
import numpy as np
from PIL import Image, ImageDraw
import math
from scipy.spatial import ConvexHull
from itertools import combinations
import itertools
from torch.utils.data import DataLoader, TensorDataset
from src.neural.neural_utils import *
from src.utils import chart_utils, data_utils

import config
from src import bk
from collections import defaultdict
from sklearn.cluster import DBSCAN


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


def calculate_circular_difference(angles, wrap_size=20):
    """
    Computes the circular difference (range) for each sliding window of angles.

    The circular difference is defined as the minimal arc that covers all the
    angles in the window. This is computed as 2π minus the largest gap between
    consecutive angles (after sorting), with the wrap-around gap included.

    Args:
        angles (torch.Tensor): A 1D tensor of angles in degrees.
        wrap_size (int): Number of consecutive angles to consider in each window.

    Returns:
        torch.Tensor: A tensor of circular differences (in radians) for each window.
    """
    # Convert degrees to radians
    angles_rad = angles * math.pi / 180

    # Wrap the data for windowing (to account for wrapping around the end)
    wrapped_radians = torch.cat([angles_rad, angles_rad[:wrap_size - 1]])

    def circular_difference(window):
        # Sort the angles in the window
        sorted_angles, _ = torch.sort(window)
        # Compute differences between adjacent sorted angles
        diffs = sorted_angles[1:] - sorted_angles[:-1]
        # Include the wrap-around difference: from the last angle to the first angle plus 2π
        wrap_diff = (sorted_angles[0] + 2 * math.pi) - sorted_angles[-1]
        # Combine all differences
        all_diffs = torch.cat([diffs, wrap_diff.unsqueeze(0)])
        # The circular difference (range) is the complement of the largest gap in a full circle
        circ_range = 2 * math.pi - torch.max(all_diffs)
        return circ_range

    differences = []
    for i in range(len(angles_rad)):
        window = wrapped_radians[i:i + wrap_size]
        diff_val = circular_difference(window)
        differences.append(diff_val)

    # Convert the list of differences into a tensor and return it
    differences = torch.tensor(differences)
    return differences


def middle_values(value_list, cut_percent=0.1):
    length = len(value_list)
    head_cut_index = math.floor(length * cut_percent)
    tail_cut_index = math.ceil(length * (1 - cut_percent))
    return value_list[head_cut_index:tail_cut_index]


def circular_var(values):
    values[values > 10] = values[values > 10] - 360
    return torch.var(values)


#####################################################

def smooth_angles(angles, window_size=11):
    """
    Smooth a 1-D tensor of angles in degrees using a circular moving average.
    Angles are converted to radians for averaging, then converted back.
    """
    if window_size % 2 == 0:
        window_size += 1
    pad = window_size // 2
    angles_padded = torch.cat([angles[-pad:], angles, angles[:pad]])
    smoothed = []
    for i in range(pad, len(angles) + pad):
        window = angles_padded[i - pad: i + pad + 1]
        window_rad = window * math.pi / 180.0
        sin_mean = torch.sin(window_rad).mean()
        cos_mean = torch.cos(window_rad).mean()
        mean_angle_rad = torch.atan2(sin_mean, cos_mean)
        mean_angle_deg = (mean_angle_rad * 180.0 / math.pi) % 360
        smoothed.append(mean_angle_deg)
    return torch.tensor(smoothed, dtype=angles.dtype)


def circular_diff(angles):
    """
    Compute differences between consecutive angles (in degrees) with wrap-around.
    The result is in the interval [-180, 180].
    """
    diff = torch.diff(angles)
    diff = (diff + 180) % 360 - 180
    return diff


def circular_mean(angles):
    """
    Compute the circular mean (in degrees) of a tensor of angles.
    """
    angles_rad = angles * math.pi / 180.0
    sin_mean = torch.sin(angles_rad).mean()
    cos_mean = torch.cos(angles_rad).mean()
    mean_angle_rad = torch.atan2(sin_mean, cos_mean)
    return (mean_angle_rad * 180.0 / math.pi) % 360


def extract_line_curves(
        angles,
        angle_var_th=1.0,  # Maximum allowed variance of the smoothed angles (for a line)
        smoothing_window=11,  # Window size for smoothing the angles
        window_size_var=11,  # Window size for computing local derivative variance
        var_threshold=0.15,  # Local derivative variance threshold (deg^2) to consider a region as stable
        deriv_mean_thresh=1.0,  # Mean derivative threshold (deg) for labeling a segment as a line
        min_seg_length=10,  # Minimum number of indices for a segment to be considered valid
        border_trim=5,  # Number of indices to trim from each end of a segment to reduce edge effects
        max_gap=3,  # only break the segment if the gap is larger than that value
):
    """
    Segment the outline (given as a 1-D tensor of tangent angles in degrees) into regions
    corresponding to "line" or "circle" segments.

    The segmentation parameters are now arguments so you can fine-tune them.
    The function expects to find exactly two segments labeled as "line" for your application.
    """
    # Step 1: Smooth the input angles.
    smoothed_angles = smooth_angles(angles, window_size=smoothing_window)

    # Step 2: Compute the circular derivative.
    d_angles = circular_diff(smoothed_angles)

    # Step 3: Compute local variance of the derivative using a sliding window.
    pad = window_size_var // 2
    padded_d = torch.cat([d_angles[-pad:], d_angles, d_angles[:pad]])
    local_var = []
    for i in range(pad, len(d_angles) + pad):
        window = padded_d[i - pad: i + pad + 1]
        local_var.append(window.var(unbiased=False))
    local_var = torch.tensor(local_var, dtype=d_angles.dtype)

    segments = []
    segment_labels = []
    current_seg = []

    # Step 4: Segment the derivative signal based on local variance.
    # We also split segments if there is a gap in consecutive indices.
    for i, lv in enumerate(local_var):
        if lv < var_threshold:
            # If a gap is detected, process the current segment first.
            if current_seg and (i - current_seg[-1] > max_gap):
                if len(current_seg) >= min_seg_length:
                    # Trim the candidate segment edges.
                    seg_indices = current_seg[border_trim:-border_trim] if len(
                        current_seg) > 2 * border_trim else current_seg
                    deriv_vals = torch.tensor([d_angles[j] for j in seg_indices])
                    mean_deriv = abs(circular_mean(deriv_vals))
                    seg_angles = smoothed_angles[seg_indices]
                    angle_variance = seg_angles.var(unbiased=False)
                    if mean_deriv < deriv_mean_thresh and angle_variance < angle_var_th:
                        segment_labels.append("line")
                    else:
                        segment_labels.append("circle")
                    segments.append(current_seg)
                current_seg = [i]
            else:
                current_seg.append(i)
        else:
            # End of a candidate segment.
            if current_seg and len(current_seg) >= min_seg_length:
                seg_indices = current_seg[border_trim:-border_trim] if len(
                    current_seg) > 2 * border_trim else current_seg
                deriv_vals = torch.tensor([d_angles[j] for j in seg_indices])
                mean_deriv = abs(circular_mean(deriv_vals))
                seg_angles = smoothed_angles[seg_indices]
                angle_variance = seg_angles.var(unbiased=False)
                if mean_deriv < deriv_mean_thresh and angle_variance < angle_var_th:
                    segment_labels.append("line")
                else:
                    segment_labels.append("circle")
                segments.append(current_seg)
            current_seg = []

    # Process any remaining segment.
    if current_seg and len(current_seg) >= min_seg_length:
        seg_indices = current_seg[border_trim:-border_trim] if len(current_seg) > 2 * border_trim else current_seg
        deriv_vals = torch.tensor([d_angles[j] for j in seg_indices])
        mean_deriv = abs(circular_mean(deriv_vals))
        seg_angles = smoothed_angles[seg_indices]
        angle_variance = seg_angles.var(unbiased=False)
        if mean_deriv < deriv_mean_thresh and angle_variance < angle_var_th:
            segment_labels.append("line")
        else:
            segment_labels.append("circle")
        segments.append(current_seg)

    # Step 5: (Optional) Merge head and tail segments if the contour is closed.
    if segments and (segments[0][0] < border_trim and segments[-1][-1] > len(angles) - border_trim):
        if segment_labels[0] == segment_labels[-1]:
            merged_seg = segments[-1] + segments[0]
            segments = segments[1:-1] + [merged_seg]
            segment_labels = segment_labels[1:-1] + [segment_labels[0]]

    # Final check: Expect exactly two segments labeled as "line".
    if segment_labels.count("line") != 2 or segment_labels.count("circle") != 1:
        #     For debugging purposes, optionally display diagnostic plots.
        chart_utils.show_line_chart(local_var, file_name=config.output / "local_variance.png")
        chart_utils.show_line_chart(angles, file_name=config.output / "angles.png")

        print(
            f"Found {segment_labels.count('line')} line(s), {segment_labels.count('circle')} circle(s).")
        detect_correct = False
    else:
        detect_correct = True
    return segments, segment_labels, detect_correct


#####################################################

# def extract_line_curves(angles, angle_var_th=1e+2):
#     circular_variances = calculate_circular_variance(angles, wrap_size=10)
#
#     all_seg = []
#     check_list = []
#     current_seg = []
#     segment_labels = []
#     future_step = 5
#     # Find the first index where the smoothed trend stops decreasing
#     for i in range(len(angles) - future_step):
#         if circular_variances[i] < 0.05:
#             current_seg.append(i)
#             check_list.append(i)
#         else:
#             if len(current_seg) > 20:
#                 seg_angle_var = circular_var(angles[current_seg[10:-10]])
#                 if seg_angle_var < angle_var_th:
#                     segment_labels.append("line")
#                 else:
#                     segment_labels.append("circle")
#                 all_seg.append(current_seg)
#             current_seg = []
#             current_seg.append(i)
#             check_list.append(i)
#     if len(current_seg) > 20:
#         all_seg.append(current_seg)
#         seg_angle_var = circular_var(angles[current_seg[10:-10]])
#         if seg_angle_var < angle_var_th:
#             segment_labels.append("line")
#         else:
#             segment_labels.append("circle")
#     all_seg = [seg for seg in all_seg if len(seg) > 10]
#     is_head_tail_connect = (all_seg[0][0] < 10) and (all_seg[-1][-1] > len(angles) - 10)
#     is_head_line = circular_var(angles[middle_values(all_seg[0])]) < 10
#     is_tail_line = circular_var(angles[middle_values(all_seg[-1])]) < 10
#     is_head_circle = circular_var(angles[middle_values(all_seg[0])]) > 10
#     is_tail_circle = circular_var(angles[middle_values(all_seg[-1])]) > 10
#     if is_head_tail_connect:
#         if is_head_line and is_tail_line or is_head_circle and is_tail_circle:
#             merged_seg = all_seg[-1] + all_seg[0]
#             if len(all_seg[-1]) > len(all_seg[0]):
#                 label = segment_labels[-1]
#             else:
#                 label = segment_labels[0]
#             all_seg = all_seg[1:-1] + [merged_seg]
#             segment_labels = segment_labels[1:-1] + [label]
#     if segment_labels.count("line") != 2:
#         chart_utils.show_line_chart(circular_variances, file_name=config.output / "circular_image.png")
#         chart_utils.show_line_chart(angles, file_name=config.output / "circular_difference_image.png")
#         raise ValueError
#     return all_seg, segment_labels


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

    contour_img = np.zeros((input_array.shape[0], input_array.shape[0], 3), dtype=np.uint8)
    line_width = 3
    from src import bk
    for i in range(len(all_points)):
        pos = all_points[i]
        contour_img[pos[1] - line_width:pos[1] + line_width, pos[0] - line_width:pos[0] + line_width] = [255, 255, 255]
    # chart_utils.van(contour_img, config.output / "contour.png")
    return contour_points


def calculate_dvs(contour_points):
    dvs = []
    shifted_contour_points = []
    for contour_list in contour_points:
        direction_vector = data_utils.contour_to_direction_vector(contour_list)
        direction_vector = torch.tensor(direction_vector)
        # direction_vector[direction_vector > 350] = 360 - direction_vector[direction_vector > 350]
        direction_vector_shifted, contour_list_shifted = data_utils.shift_by_largest_gap_tensor(direction_vector,
                                                                                                torch.from_numpy(
                                                                                                    contour_list))
        dvs.append(direction_vector_shifted)
        shifted_contour_points.append(contour_list_shifted.numpy())
    return dvs, shifted_contour_points


def get_contour_segs(img):
    bw_img = np.array(Image.fromarray(img.to("cpu").numpy().astype('uint8')).convert("L"))
    bw_img[bw_img == 211] = 0
    bw_img[bw_img > 0] = 1

    # bw_img = resize_img(bw_img, 64)
    contour_points = find_contours(bw_img)

    dvs, contour_points = calculate_dvs(contour_points)

    all_segments = []
    all_labels = []
    # find out the segments, and their labels (curve or line)
    for dv in dvs:
        angle_var_th = 5.0  # Maximum allowed variance of the smoothed angles (for a line)
        smoothing_window = 11  # Window size for smoothing the angles
        window_size_var = 11  # Window size for computing local derivative variance
        var_threshold = 0.5  # Local derivative variance threshold (deg^2) to consider a region as stable
        deriv_mean_thresh = 10000  # Mean derivative threshold (deg) for labeling a segment as a line
        min_seg_length = 5  # Minimum number of indices for a segment to be considered valid
        border_trim = 5  # Number of indices to trim from each end of a segment to reduce edge effects
        max_gap = 10
        segments, seg_labels, detect_res = extract_line_curves(dv, angle_var_th,
                                                   smoothing_window,
                                                   window_size_var, var_threshold,
                                                   deriv_mean_thresh, min_seg_length,
                                                   border_trim, max_gap)
        if not detect_res:
            print("")
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
    line_width = 5
    while True:
        array[y1 - line_width:y1 + line_width, x1 - line_width:x1 + line_width] = torch.tensor(
            color)  # Set the pixel value at the current point

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
    line_width = 5
    # Draw the arc on the array
    for x, y in zip(x_coords, y_coords):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # Check bounds
            image[y - line_width:y + line_width, x - line_width:x + line_width] = color
    chart_utils.van(image, file_name=config.output / "closure_arc_segs.png")
    return image


# def is_collinear(p1, p2, p3, distance_tolerance):
#     """
#     Checks if point p3 is collinear with line segment (p1, p2)
#     """
#     area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
#     return area < distance_tolerance


def euclidean_distance(p1, p2):
    return torch.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def cluster_lines_by_proximity(lines, eps=10):
    """
    Clusters line segments based on the proximity of their endpoints.
    """
    endpoints = [(line[1] + line[2]) / 2 / 1024 for line in lines]

    clustering = DBSCAN(eps=0.1, min_samples=1).fit(endpoints)

    clusters = defaultdict(list)
    for idx, label in enumerate(clustering.labels_):
        clusters[label].append([lines[idx]])

    return clusters


def label_corners(clusters):
    """
    Labels clusters as left_top, left_bottom, right_top, right_bottom.
    """
    labeled_clusters = {}

    for label, points in clusters.items():
        try:
            points_np = np.array([[points[0][0][1], points[0][0][2]], [
                points[1][0][1], points[1][0][2]]])
        except IndexError:
            return {}
        centers = points_np.mean(axis=1)
        if centers[:, 1].argmax() == centers[:, 0].argmax():
            # left_bottom or right top

            # left_bottom: if the horizontal line on the right side
            if np.abs(points[centers[:, 1].argmax()][0][0]) < 0.5:
                labeled_clusters[label] = ('left_bottom', {"lines": points, "pos": centers.mean(axis=0)})
            else:
                labeled_clusters[label] = ('right_top', {"lines": points, "pos": centers.mean(axis=0)})
        else:
            if np.abs(points[centers[:, 1].argmax()][0][0]) < 0.5:
                labeled_clusters[label] = ('right_bottom', {"lines": points, "pos": centers.mean(axis=0)})
            else:
                labeled_clusters[label] = ('left_top', {"lines": points, "pos": centers.mean(axis=0)})
    return labeled_clusters


def group_corners_to_rectangles(labeled_clusters):
    """
    Groups four corners into rectangle clusters.
    """
    grouped_rectangles = []
    used = set()

    left_tops = {k: v for k, v in labeled_clusters.items() if v[0] == 'left_top'}
    right_tops = {k: v for k, v in labeled_clusters.items() if v[0] == 'right_top'}
    left_bottoms = {k: v for k, v in labeled_clusters.items() if v[0] == 'left_bottom'}
    right_bottoms = {k: v for k, v in labeled_clusters.items() if v[0] == 'right_bottom'}

    for lt_label, (lt_type, lt_points) in left_tops.items():
        best_rt_label = None
        min_dist = float('inf')

        for rt_label, (rt_type, rt_points) in right_tops.items():
            if rt_label in used:
                continue
            if rt_points["pos"][0] > lt_points["pos"][0]:
                if abs(rt_points["pos"][1] - lt_points["pos"][1]) < 20:  # Same height check
                    dist = abs(rt_points["pos"][0] - lt_points["pos"][0])
                    if dist < min_dist:
                        min_dist = dist
                        best_rt_label = rt_label

        if best_rt_label is None:
            continue

        best_lb_label = None
        min_dist = float('inf')
        for lb_label, (lb_type, lb_points) in left_bottoms.items():

            if lb_label in used:
                continue
            if lb_points["pos"][1] > lt_points["pos"][1]:  # higher y-axis
                if abs(lb_points["pos"][0] - lt_points["pos"][0]) < 20:  # Same x-axis
                    dist = abs(lb_points["pos"][1] - lt_points["pos"][1])
                    if dist < min_dist:
                        min_dist = dist
                        best_lb_label = lb_label

        best_rb_label = None
        min_dist = float('inf')
        for rb_label, (rb_type, rb_points) in right_bottoms.items():
            if rb_label in used:
                continue
            if rb_points["pos"][1] > lt_points["pos"][1] and rb_points["pos"][0] > lt_points["pos"][0]:
                dist = abs(rb_points["pos"][1] - lt_points["pos"][1]) + abs(rb_points["pos"][0] - lt_points["pos"][0])
                if dist < min_dist:
                    min_dist = dist
                    best_rb_label = rb_label
        try:
            grouped_rectangles.append({
                'left_top': lt_points,
                'right_top': labeled_clusters[best_rt_label][1],
                'left_bottom': labeled_clusters[best_lb_label][1],
                'right_bottom': labeled_clusters[best_rb_label][1],
            })
        except KeyError:
            continue
        used.update([lt_label, best_rt_label, best_lb_label, best_rb_label])

    return grouped_rectangles


def merge_rectangles(lines):
    clusters = cluster_lines_by_proximity(lines)
    labeled_clusters = label_corners(clusters)
    rectangles = group_corners_to_rectangles(labeled_clusters)
    return rectangles


def label_triangle_corners(clusters):
    """
    Labels clusters as left_top, left_bottom, right_top, right_bottom.
    """
    labeled_clusters = {}

    for label, points in clusters.items():
        # xs = [p[0] for p in points]
        # ys = [p[1] for p in points]

        # min_x, max_x = min(xs), max(xs)
        # min_y, max_y = min(ys), max(ys)
        try:
            points_np = np.array([[points[0][0][1], points[0][0][2]], [
                points[1][0][1], points[1][0][2]]])
        except IndexError:
            return {}
        centers = points_np.mean(axis=1)
        if np.abs(points[0][0][0]) < 0.5:
            if points[1][0][0] > 0:
                labeled_clusters[label] = ('left_bottom', {"lines": points, "pos": centers.mean(axis=0)})
            else:
                labeled_clusters[label] = ('right_bottom', {"lines": points, "pos": centers.mean(axis=0)})
        elif np.abs(points[1][0][0]) < 0.5:
            if points[0][0][0] > 0:
                labeled_clusters[label] = ('left_bottom', {"lines": points, "pos": centers.mean(axis=0)})
            else:
                labeled_clusters[label] = ('right_bottom', {"lines": points, "pos": centers.mean(axis=0)})
        else:
            labeled_clusters[label] = ('top', {"lines": points, "pos": centers.mean(axis=0)})
    return labeled_clusters


def group_corners_to_triangles(labeled_clusters):
    """
    Groups four corners into rectangle clusters.
    """
    grouped_rectangles = []
    used = set()

    left_bottoms = {k: v for k, v in labeled_clusters.items() if v[0] == 'left_bottom'}
    right_bottoms = {k: v for k, v in labeled_clusters.items() if v[0] == 'right_bottom'}
    tops = {k: v for k, v in labeled_clusters.items() if v[0] == 'top'}

    for t_label, (t_type, t_points) in tops.items():
        best_rb_label = None
        min_dist = float('inf')

        for rb_label, (rb_type, rb_points) in right_bottoms.items():
            if rb_label in used:
                continue
            if rb_points["pos"][0] < t_points["pos"][0]:
                rb_line_index = torch.tensor((rb_points["lines"][0][0][0], rb_points["lines"][1][0][0])).argmin()
                t_line_index = torch.tensor((t_points["lines"][0][0][0], t_points["lines"][1][0][0])).argmin()
                valid = is_collinear(t_points["lines"][t_line_index][0][1].astype(np.float32),
                                     t_points["lines"][t_line_index][0][2].astype(np.float32),
                                     rb_points["lines"][rb_line_index][0][1].astype(np.float32),
                                     tolerance=1e-3)
                if valid:
                    dist = abs(rb_points["pos"][0] - t_points["pos"][0])
                    if dist < min_dist:
                        min_dist = dist
                        best_rb_label = rb_label

        if best_rb_label is None:
            continue

        best_lb_label = None
        min_dist = float('inf')
        for lb_label, (lb_type, lb_points) in left_bottoms.items():
            if lb_label in used:
                continue
            if lb_points["pos"][1] > t_points["pos"][1]:  # higher y-axis
                lb_line_index = torch.tensor((lb_points["lines"][0][0][0], lb_points["lines"][1][0][0])).argmax()
                t_line_index = torch.tensor((t_points["lines"][0][0][0], t_points["lines"][1][0][0])).argmax()
                valid = is_collinear(
                    t_points["lines"][t_line_index][0][1].astype(np.float32),
                    t_points["lines"][t_line_index][0][2].astype(np.float32),
                    lb_points["lines"][lb_line_index][0][1].astype(np.float32),
                    tolerance=1e-3)
                if valid:
                    dist = abs(lb_points["pos"][1] - t_points["pos"][1])
                    if dist < min_dist:
                        min_dist = dist
                        best_lb_label = lb_label

        grouped_rectangles.append({
            'top': t_points,
            'right_bottom': labeled_clusters[best_rb_label][1],
            'left_bottom': labeled_clusters[best_lb_label][1],

        })

        used.update([t_label, best_rb_label, best_lb_label])

    return grouped_rectangles


def merge_triangles(lines):
    clusters = cluster_lines_by_proximity(lines)
    labeled_clusters = label_triangle_corners(clusters)
    triangles = group_corners_to_triangles(labeled_clusters)
    return triangles


#
# def merge_similar_lines(lines, line_obj_indices, slope_tolerance, distance_tolerance, vertical_th=8):
#     used = np.zeros(len(lines))
#     similar_lines = []
#     line_group_data = []
#     for l_i, line in enumerate(lines):
#         merged_line_obj_indices = [
#             line_obj_indices[l_i]
#         ]
#
#         current_line = line
#         if used[l_i]:
#             continue
#         if l_i == len(lines):
#             if used[l_i] == 0:
#                 used[l_i] = 1
#                 similar_lines.append(line)
#                 merged_line_obj_indices.append(line_obj_indices[l_i])
#         for l2_i in range(l_i + 1, len(lines)):
#             l2 = lines[l2_i]
#             k1 = line[0]
#             k2 = l2[0]
#             # Convert slopes to angles (in radians)
#             theta1 = math.atan(k1)
#             theta2 = math.atan(k2)
#             # Compute the absolute difference in angles
#             angle_diff = abs(theta1 - theta2)
#             slope_similar = angle_diff < slope_tolerance
#             all_vertical = np.abs(k2) >= vertical_th and np.abs(k1) >= vertical_th
#             collinearity = is_collinear(current_line[1], current_line[2], l2[1], distance_tolerance)
#             if (slope_similar or all_vertical) and collinearity:
#                 used[l_i] = 1
#                 used[l2_i] = 1
#                 # Update the start and end to cover both lines
#                 current_start = min(current_line[1], current_line[2], l2[1], l2[2], key=lambda p: (p[0], p[1]))
#                 current_end = max(current_line[1], current_line[2], l2[1], l2[2], key=lambda p: (p[0], p[1]))
#                 current_line = [current_line[0], current_start, current_end]
#                 merged_line_obj_indices.append(line_obj_indices[l2_i])
#         similar_lines.append(current_line)
#         line_group_data.append(np.unique(merged_line_obj_indices))
#
#     return similar_lines, line_group_data


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
    # cir_img = np.zeros((width, width, 3), dtype=np.uint8)
    # for c_i, cir in enumerate(circles):
    #     cir_img = draw_arc_on_image(cir_img, cir["center"], cir["radius"], cir["start_angle"], cir["end_angle"],
    #                                 colors[1], 2, cir["direction"])
    # chart_utils.van(cir_img, config.output / "closure_arc_segs.png")

    return circles


def get_line_groups(contour_points, contour_segs, contour_seg_labels, width):
    line_segs = []
    line_obj_indices = []
    for contour_i, contour_seg in enumerate(contour_segs):
        for seg_i, seg in enumerate(contour_seg):
            if contour_seg_labels[contour_i][seg_i] == "line":
                line_segs.append(contour_points[contour_i][seg])
                line_obj_indices.append(contour_i)

    lines = []
    for line_seg in line_segs:
        slope, start, end = calculate_line_properties(line_seg)
        if np.abs(start[0] - end[0]) < 5:
            end[0] = start[0]
        if np.abs(start[1] - end[1]) < 5:
            end[1] = start[1]
        ends = sorted([start, end], key=lambda p: (p[0], p[1]))
        # start = start.astype(np.float32)
        # start/=width
        # end = end.astype(np.float32)
        # end/=width
        lines.append([slope, ends[0], ends[1]])

    # visual lines
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 0, 128), (255, 0, 128)]
    line_img = torch.zeros(width, width, 3)
    for l_i, line in enumerate(lines):
        line_img = draw_line_on_array(line_img, line[1], line[2], colors[0])
    chart_utils.van(line_img.numpy().astype(np.uint8), file_name=config.output / "closure_line_segs.png")

    # merge all the line_segs
    rectangles = merge_rectangles(lines)
    # # visual lines
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 0, 128), (255, 0, 128)]
    # merged_line_img = torch.zeros(width, width, 3)
    # for l_i, line in enumerate(merged_lines):
    #     merged_line_img = draw_line_on_array(merged_line_img, line[1], line[2], colors[l_i])
    # chart_utils.van(merged_line_img.numpy().astype(np.uint8), file_name=config.output / "closure_merged_lines.png")

    return rectangles


def get_triangle_groups(contour_points, contour_segs, contour_seg_labels, width):
    line_segs = []
    line_obj_indices = []
    for contour_i, contour_seg in enumerate(contour_segs):
        for seg_i, seg in enumerate(contour_seg):
            if contour_seg_labels[contour_i][seg_i] == "line":
                line_segs.append(contour_points[contour_i][seg])
                line_obj_indices.append(contour_i)

    lines = []
    for line_seg in line_segs:
        slope, start, end = calculate_line_properties(line_seg)
        if np.abs(start[0] - end[0]) < 5:
            end[0] = start[0]
        if np.abs(start[1] - end[1]) < 5:
            end[1] = start[1]
        ends = sorted([start, end], key=lambda p: (p[0], p[1]))
        lines.append([slope, ends[0], ends[1]])

    # visual lines
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 0, 128), (255, 0, 128)]
    line_img = torch.zeros(width, width, 3)
    for l_i, line in enumerate(lines):
        line_img = draw_line_on_array(line_img, line[1], line[2], colors[0])
    chart_utils.van(line_img.numpy().astype(np.uint8), file_name=config.output / "closure_line_segs.png")

    # merge all the line_segs
    triangles = merge_triangles(lines)
    # # visual lines
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 0, 128), (255, 0, 128)]
    # merged_line_img = torch.zeros(width, width, 3)
    # for l_i, line in enumerate(merged_lines):
    #     merged_line_img = draw_line_on_array(merged_line_img, line[1], line[2], colors[l_i])
    # chart_utils.van(merged_line_img.numpy().astype(np.uint8), file_name=config.output / "closure_merged_lines.png")

    return triangles


def distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def forms_closed_figure(line_ends, max_distance=1e-2):
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
        other_points = np.concatenate((line_ends[:l_i], line_ends[l_i + 1:])).reshape(-1, 2)
        for point in other_points:
            dists.append(distance(line[0], point))
        if not sorted(dists)[0] < max_distance:
            print(f"point1: {line[0]}, {sorted(dists)[1]}")
            closed = False
            break
        dists = []
        for point in other_points:
            dists.append(distance(line[1], point))
        if not sorted(dists)[0] < max_distance:
            print(f"point1: {line[0]}, {sorted(dists)[1]}")
            closed = False
            break

    return closed


def find_triangles(triangle_lines, center_points, obj_group_labels, group_labels):
    # if len(lines) < 4:
    #     return obj_group_labels, False, group_labels

    group_id = 0
    for tri_group in triangle_lines:
        group_id += 1
        group_labels.append(1)
        for k, v in tri_group.items():
            pos = v["pos"] / 1024
            dists = [euclidean_distance(pos, cp) for cp in center_points]
            index = dists.index(min(dists))
            obj_group_labels[index] = group_id
    #
    # if len(lines) < 3:
    #     return obj_group_labels, False, group_labels
    #
    #     # Create a list from 0 to n
    # comb_lists = list(combinations(list(range(len(lines))), 3))  # Get all combinations of length 3
    # # check closed lines
    # # check slopes
    # triangles = []
    # for comb_list in comb_lists:
    #     triangle_lines = [lines[i] for i in comb_list]
    #     end_points = []
    #     for line in triangle_lines:
    #         end_points.append([line[1], line[2]])
    #     end_points = np.array(end_points)
    #     close = forms_closed_figure(end_points, max_distance=0.1)
    #     triangle = None
    #     if close:
    #         triangles.append(comb_list)
    #
    # label_id = obj_group_labels.max()
    # hasTriangle = False
    # for triangle in triangles:
    #     hasTriangle = True
    #     group_labels.append(1)
    #     label_id += 1
    #     for line_i in triangle:
    #         obj_group_labels[list(line_groups[line_i])] = label_id
    return obj_group_labels, group_labels


def find_squares(rect_lines, center_points, obj_group_labels, group_labels):
    # if len(lines) < 4:
    #     return obj_group_labels, False, group_labels
    group_id = 0
    for rect_group in rect_lines:
        group_id += 1
        group_labels.append(2)
        for k, v in rect_group.items():
            pos = v["pos"] / 1024
            dists = [euclidean_distance(pos, cp) for cp in center_points]
            index = dists.index(min(dists))
            obj_group_labels[index] = group_id

    return obj_group_labels, group_labels


def assign_cluster(pt, clusters, tolerance):
    """
    Assigns a point to an existing cluster if it's close enough (within tolerance);
    otherwise, creates a new cluster.
    """
    for i, cl in enumerate(clusters):
        if distance(pt, cl) < tolerance:
            return i
    clusters.append(pt)
    return len(clusters) - 1


def find_position_closure_triangles(lines, line_group_data, labels, group_labels, tolerance=0.05):
    """
    Given a list of lines (each represented as [slope, (x1, y1), (x2, y2)]),
    returns a list where each element is a list of line indices that together form a triangle.

    For a set of three lines to form a triangle:
      1. The endpoints (allowing for gaps) must cluster into exactly 3 distinct vertices.
      2. Each line should connect two different clusters.
      3. Each vertex should be connected exactly twice.

    Parameters:
      lines (list): List of lines.
      tolerance (float): Maximum distance to consider two endpoints as the same vertex.

    Returns:
      List[List[int]]: Each inner list contains the indices of lines that form a triangle.
    """
    triangles = []
    n = len(lines)

    # Check every combination of three lines
    for combo in itertools.combinations(range(n), 3):
        clusters = []  # Will store representative points for vertices
        line_assignments = []  # List of tuples: (cluster_index of endpoint1, cluster_index of endpoint2)
        valid = True

        # Process each line in the combination
        for idx in combo:
            p1 = lines[idx][1]
            p2 = lines[idx][2]
            c1 = assign_cluster(p1, clusters, tolerance)
            c2 = assign_cluster(p2, clusters, tolerance)

            # If both endpoints are in the same cluster, it's not a valid edge for a triangle.
            if c1 == c2:
                valid = False
                break
            line_assignments.append((c1, c2))

        if not valid:
            continue

        # For a triangle, there should be exactly 3 distinct vertices.
        if len(clusters) != 3:
            continue

        # Count the occurrence of each vertex in the edges.
        vertex_count = {}
        for (c1, c2) in line_assignments:
            vertex_count[c1] = vertex_count.get(c1, 0) + 1
            vertex_count[c2] = vertex_count.get(c2, 0) + 1

        # In a proper triangle, each of the three vertices should appear exactly twice.
        if sorted(vertex_count.values()) != [2, 2, 2]:
            continue

        # If we passed all the tests, add this combination as a triangle.
        triangles.append(list(combo))
    for tri_line_indices in triangles:
        group_labels.append(1)
        group_id = max(labels) + 1
        for line_index in tri_line_indices:
            point_indices = line_group_data[line_index]
            for point_index in point_indices:
                labels[point_index] = group_id
    return labels, group_labels


def find_position_closure_squares(lines, line_group_data, labels, group_labels, tolerance=0.05):
    """
    Given a list of lines (each represented as [slope, (x1, y1), (x2, y2)]),
    finds all combinations of four lines that together form a square based on position closure.

    For a set of four lines to form a square:
      1. The endpoints (allowing for gaps) must cluster into exactly 4 distinct vertices.
      2. Each line should connect two different clusters.
      3. Each vertex should be connected exactly twice.

    After a square is found, the function assigns the square group label (2) to the points
    associated with the lines forming the square. The group label is updated in the provided
    labels list, and the group_labels list is appended with 2 for each square.

    Parameters:
      lines (list): List of lines.
      line_group_data (list): Mapping from each line index to the list of associated point indices.
      labels (list): List of labels for each point.
      group_labels (list): List of group labels already assigned.
      tolerance (float): Maximum distance to consider two endpoints as the same vertex.

    Returns:
      Tuple: Updated (labels, group_labels).
    """
    import itertools

    squares = []
    n = len(lines)

    # Check every combination of four lines
    for combo in itertools.combinations(range(n), 4):
        clusters = []  # Will store representative points for vertices
        line_assignments = []  # List of tuples: (cluster_index of endpoint1, cluster_index of endpoint2)
        valid = True

        # Process each line in the combination
        for idx in combo:
            p1 = lines[idx][1]
            p2 = lines[idx][2]
            c1 = assign_cluster(p1, clusters, tolerance)
            c2 = assign_cluster(p2, clusters, tolerance)

            # If both endpoints fall into the same cluster, this line is degenerate for closure.
            if c1 == c2:
                valid = False
                break
            line_assignments.append((c1, c2))

        if not valid:
            continue

        # For a square, there should be exactly 4 distinct vertices.
        if len(clusters) != 4:
            continue

        # Count the occurrence of each vertex in the edges.
        vertex_count = {}
        for (c1, c2) in line_assignments:
            vertex_count[c1] = vertex_count.get(c1, 0) + 1
            vertex_count[c2] = vertex_count.get(c2, 0) + 1

        # In a proper square, each of the four vertices should appear exactly twice.
        if sorted(vertex_count.values()) != [2, 2, 2, 2]:
            continue

        # If we passed all the tests, add this combination as a square.
        squares.append(list(combo))

    # Update the labels and group_labels for each found square.
    for square_line_indices in squares:
        group_labels.append(2)
        group_id = max(labels) + 1
        for line_index in square_line_indices:
            point_indices = line_group_data[line_index]
            for point_index in point_indices:
                labels[point_index] = group_id

    return labels, group_labels


def find_circles(curves, circle_data, labels):
    raise NotImplementedError
    # check slopes
    end_points = []
    for line in curves:
        end_points.append([line[1], line[2]])
    end_points = np.array(end_points)
    close = forms_closed_figure(end_points, max_distance=120)
    circle = None

    return circle


def compute_convex_hull(points):
    """
    Step 1: Given the objects' positions in the image,
    calculate the convex hull over all the objects.
    """
    hull = ConvexHull(points)
    hull_indices = hull.vertices  # indices of points in convex hull
    hull_points = points[hull_indices]
    # Ensure the hull points are in order (ConvexHull generally returns them in counter-clockwise order)
    return hull_points, hull_indices


def are_points_on_same_line(pts, threshold=1e-3):
    """
    Check if a set of points are collinear (stub).
    One way is to check area of polygon they form, or check direction vectors.
    """
    if len(pts) < 2:
        return True

    # Vector from first to second point
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    base_vec = np.array([x2 - x1, y2 - y1])

    for i in range(2, len(pts)):
        xi, yi = pts[i]
        test_vec = np.array([xi - x1, yi - y1])
        cross_val = np.cross(base_vec, test_vec)
        if abs(cross_val) > threshold:
            return False
    return True


def fit_circle_to_points(pts):
    """
    Fit a circle to 3 or more points (stub).
    Returns (cx, cy, r) for center and radius, or None if not possible.
    A full implementation would solve the system of equations for a circle:
       (x - cx)^2 + (y - cy)^2 = r^2
    for each point in pts.
    """
    # For demonstration, we show a naive approach for 3 points:
    if len(pts) < 3:
        return None

    # Just handle exactly 3 points for example (more robust approach needed for many points).
    # Solve by circumcenter formula or direct method:
    (x1, y1), (x2, y2), (x3, y3) = pts[:3]

    d = 2 * (x1 * (y2 - y3) +
             x2 * (y3 - y1) +
             x3 * (y1 - y2))

    if abs(d) < 1e-12:
        return None  # Points are collinear or too close to collinear

    ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) +
          (x2 ** 2 + y2 ** 2) * (y3 - y1) +
          (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d

    uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) +
          (x2 ** 2 + y2 ** 2) * (x1 - x3) +
          (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d

    # Radius is distance from (ux, uy) to any point
    r = math.sqrt((x1 - ux) ** 2 + (y1 - uy) ** 2)
    return (ux, uy, r)


def are_points_co_circular(pts, threshold=1e-3):
    """
    Check if consecutive N objects are co-circular and similarly spaced (stub).
    1) Fit circle to the group (using 3 or more points).
    2) Check if each point is close to that circle radius.
    3) Optionally check if the arc spacing is 'similar'.
    """
    circle_params = fit_circle_to_points(pts)
    if circle_params is None:
        return False
    cx, cy, r = circle_params
    for (x, y) in pts:
        dist = math.hypot(x - cx, y - cy)
        if abs(dist - r) > threshold:
            return False
    # If you need equally spaced arcs, you'd check angles or arc lengths here.
    return True


def detect_lines_and_arcs_on_hull(hull_points, hull_indices_global, min_line_size=2, min_arc_size=4, line_th=1e-2,
                                  circle_th=1e-2):
    """
    Steps 2 & 3:
      - Identify consecutive line segments and arcs on the convex hull.
      - Merge collinear lines or co-circular arcs (stubbed).
    Returns two lists of dicts:
      - group_lines: [ { "indices": [...], "points": np.array(...) }, ... ]
      - group_arcs:  [ { "indices": [...], "points": np.array(...) }, ... ]
    """
    group_lines = []
    group_arcs = []
    num_points = len(hull_points)

    i = 0
    while i < num_points:
        # Try line
        j = i + 1
        while j < num_points:
            segment = hull_points[i:j + 1]
            if not are_points_on_same_line(segment, threshold=line_th):
                break
            j += 1
        if j - i >= min_line_size:
            # Save line as dict
            segment_indices_local = list(range(i, j))
            segment_indices_global = [hull_indices_global[x] for x in segment_indices_local]
            segment_points = hull_points[i:j]

            group_lines.append({
                "indices": segment_indices_global,
                "points": segment_points.copy()
            })
            i = j
            continue

        # Try arc
        j = i + 2
        while j < num_points:
            segment = hull_points[i:j + 1]
            if not are_points_co_circular(segment, threshold=circle_th):
                break
            j += 1
        if j - i >= min_arc_size:
            segment_indices_local = list(range(i, j))
            segment_indices_global = [hull_indices_global[x] for x in segment_indices_local]
            segment_points = hull_points[i:j]

            group_arcs.append({
                "indices": segment_indices_global,
                "points": segment_points.copy()
            })
            i = j
        else:
            i += 1

    return group_lines, group_arcs


def detect_lines_and_arcs_on_hull_allow_multimembership(
        hull_points, hull_indices,
        min_line_size=2, min_arc_size=5
):
    """
    Step 2 & 3: detect lines/arcs on the hull,
    allowing a point to belong to multiple lines/arcs.

    Returns two lists of dicts:
      group_lines = [
         { "indices": [...], "points": np.array([...]) },
         ...
      ]
      group_arcs = [
         { "indices": [...], "points": np.array([...]) },
         ...
      ]

    - We do NOT skip over the points once we detect a line or arc.
      i.e. a point can appear in multiple results if geometry allows.
    """
    num_points = len(hull_points)
    group_lines = []
    group_arcs = []

    i = 0
    while i < num_points:
        # -------------------------------------------------
        # 1) Try to detect a line from i
        # -------------------------------------------------
        found_line = False
        j_line = i + 1
        # Expand j_line as long as points remain collinear
        while j_line < num_points:
            segment = hull_points[i:j_line + 1]
            if not are_points_on_same_line(segment):
                break
            j_line += 1
        length_line = j_line - i
        if length_line >= min_line_size:
            # store the line
            line_indices_local = list(range(i, j_line))
            line_indices_global = [hull_indices[k] for k in line_indices_local]
            line_points = hull_points[i:j_line]

            group_lines.append({
                "indices": line_indices_global,
                "points": line_points.copy()
            })
            found_line = True

        # -------------------------------------------------
        # 2) Try to detect an arc from i
        #    (we reset j to i+2 because arc needs >= 3 points)
        # -------------------------------------------------
        found_arc = False
        if i + 2 < num_points:
            j_arc = i + 2
            while j_arc < num_points:
                segment = hull_points[i:j_arc + 1]
                if not are_points_co_circular(segment):
                    break
                j_arc += 1
            length_arc = j_arc - i
            if length_arc >= min_arc_size:
                # store the arc
                arc_indices_local = list(range(i, j_arc))
                arc_indices_global = [hull_indices[k] for k in arc_indices_local]
                arc_points = hull_points[i:j_arc]

                group_arcs.append({
                    "indices": arc_indices_global,
                    "points": arc_points.copy()
                })
                found_arc = True

        # Move i forward by 1 (NOT by j_line or j_arc),
        # so we can continue detecting overlapping lines/arcs.
        i += 1

    return group_lines, group_arcs


def point_near_line(point, line_pts, threshold):
    """
    Check whether `point` is near the infinite line or line segment
    formed by points in `line_pts` (stub).
    """
    # A simplified check: pick two endpoints (start, end) from line_pts,
    # measure perpendicular distance.
    if len(line_pts) < 2:
        return False
    x1, y1 = line_pts[0]
    x2, y2 = line_pts[-1]  # use first and last as an approximate "main" line
    px, py = point

    # Distance from point to line (x1,y1)-(x2,y2)
    # reference formula for distance from a point to a line
    # d = |(y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denominator = math.hypot(y2 - y1, x2 - x1)
    dist = numerator / (denominator + 1e-12)

    return (dist <= threshold)


def point_near_arc(point, arc_pts, threshold):
    """
    Check whether `point` is near the arc formed by `arc_pts` (stub).
    - Fit circle to arc_pts
    - Check if `point` is near that circle's circumference
    - Optionally also check angle range for the arc
    """
    circle_params = fit_circle_to_points(arc_pts)
    if circle_params is None:
        return False
    cx, cy, r = circle_params
    px, py = point
    dist = math.hypot(px - cx, py - cy)
    return (abs(dist - r) <= threshold)


def check_rest_points(points, hull_indices, group_lines, group_arcs, line_threshold=1e-1, arc_threshold=1e-1):
    """
    Step 4: For all the rest objects that are not on the convex hull,
    check if they are part of any group lines or group arcs (stub).
    """
    rest_indices = list(set(range(len(points))) - set(hull_indices))
    rest_points = points[list(rest_indices)]
    used = torch.zeros(len(rest_points), dtype=torch.bool)
    # For each rest point, test if it lies on or near a known line or arc:
    #   - For line: distance from point to line < threshold
    #   - For arc: distance from point to circle center == radius (within threshold)
    for i, idx in enumerate(rest_indices):
        if used[i]:
            continue
        pt = points[idx]
        # Check against lines
        for l_i, line_pts in enumerate(group_lines):
            if point_near_line(pt, line_pts["points"], line_threshold):
                # Mark or store the membership of this point to that line
                group_lines[l_i]["indices"].append(idx)
                group_lines[l_i]["points"] = torch.cat((line_pts["points"], pt.reshape(-1, 2)))
                used[i] = True

        # Check against arcs
        for a_i, arc_pts in enumerate(group_arcs):
            if point_near_arc(pt, arc_pts["points"], arc_threshold):
                # Mark or store the membership of this point to that arc
                group_arcs[a_i]["indices"].append(idx)
                group_arcs[a_i]["points"] = torch.cat((arc_pts["points"], pt.reshape(-1, 2)))
                used[i] = True

    return group_lines, group_arcs


def add_rest_points_to_groups(points, hull_indices, group_lines, group_arcs,
                              assigned_mask, line_threshold=1e-1, arc_threshold=1e-1):
    """
    Step 4: For points not in `hull_indices`, if they belong to any
    existing line or arc, add them to that line/arc dict.
    Returns: the number of newly assigned points.
    """
    rest_indices = set(range(len(points))) - set(hull_indices)
    newly_assigned = 0

    for idx in rest_indices:
        if assigned_mask[idx]:
            continue  # already assigned

        pt = points[idx]

        # Check lines
        for line_dict in group_lines:

            if point_near_line(pt, line_dict["points"], threshold=line_threshold):
                # Update line_dict: add index and point
                line_dict["indices"].append(idx)
                line_dict["points"] = np.vstack([line_dict["points"], pt])
                assigned_mask[idx] = True
                newly_assigned += 1
                break  # assigned to one line is enough

        # Check arcs only if not already assigned
        if not assigned_mask[idx]:
            for arc_dict in group_arcs:
                if point_near_arc(pt, arc_dict["points"], threshold=arc_threshold):
                    arc_dict["indices"].append(idx)
                    arc_dict["points"] = np.vstack([arc_dict["points"], pt])
                    assigned_mask[idx] = True
                    newly_assigned += 1
                    break  # assigned to one arc is enough
    # if a line is shorter than 2, remove it
    for line in group_lines:
        if len(line["indices"]) == 2:
            assigned_mask[line["indices"]] = False
    return newly_assigned, assigned_mask


def lines_direction_vector(line_dict):
    """
    Compute an approximate 'direction vector' for the line_dict
    using the first and last points in 'points'.
    """
    pts = line_dict["points"]
    if len(pts) < 2:
        return None
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    vec = np.array([x2 - x1, y2 - y1])
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12)


def are_collinear_and_connectable(lineA, lineB, angle_threshold=1e-2, distance_threshold=1e-1):
    """
    Check if lineA and lineB are collinear and 'close enough' to merge.
    - angle_threshold: how close direction vectors must be
    - distance_threshold: how close endpoints must be to consider them one continuous line
    This is a simple example stub.
    """
    vA = lines_direction_vector(lineA)
    vB = lines_direction_vector(lineB)
    if vA is None or vB is None:
        return False
    # Check angle by dot product ~ 1 or -1
    dot = abs(np.dot(vA, vB))  # 1 => same direction, -1 => opposite
    if (1.0 - dot) > angle_threshold:
        return False
    else:
        return True
    # Optional: check if the lines' endpoints are near each other (overlap or continuous)
    # For simplicity, check distance among endpoints
    ptsA = lineA["points"]
    ptsB = lineB["points"]
    # E.g., distance from A's last point to B's first or last point
    # This is simplistic; you might also check bounding boxes for overlap, etc.
    endA1 = ptsA[0]
    endA2 = ptsA[-1]
    endB1 = ptsB[0]
    endB2 = ptsB[-1]

    def dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    # if any pair of endpoints is within distance_threshold, consider them mergeable
    pairs = [dist(endA1, endB1), dist(endA1, endB2),
             dist(endA2, endB1), dist(endA2, endB2)]
    if any(d < distance_threshold for d in pairs):
        return True
    return False


def merge_two_lines(lineA, lineB):
    """
    Merge two lines into a single line dict.
    - Combine indices and points, then reorder them so they are in (roughly) ascending order.
    """
    new_indices = list(set(lineA["indices"] + lineB["indices"]))
    # Combine and deduplicate points
    all_points = np.vstack((lineA["points"], lineB["points"]))
    # For a real approach, you'd reorder the merged line by something like
    # projecting onto the direction vector or sorting by x or y.
    # We do a simple bounding-box-based sort for demonstration:
    # Sort by x then by y:
    all_points_unique = np.unique(all_points, axis=0)
    all_points_sorted = all_points_unique[np.lexsort((all_points_unique[:, 1], all_points_unique[:, 0]))]

    return {
        "indices": new_indices,
        "points": all_points_sorted
    }


def merge_collinear_lines(lines, angle_threshold=1e-2, distance_threshold=1e-1):
    """
    Try to merge lines that are collinear and 'connectable.'
    We'll do repeated passes until no merges happen.
    """
    merged_something = True
    while merged_something and len(lines) > 1:
        merged_something = False
        new_lines = []
        skip_set = set()

        for i in range(len(lines)):
            if i in skip_set:
                continue
            merged_line = lines[i]
            for j in range(i + 1, len(lines)):
                if j in skip_set:
                    continue
                if are_collinear_and_connectable(merged_line, lines[j],
                                                 angle_threshold, distance_threshold):
                    # Merge j into merged_line
                    merged_line = merge_two_lines(merged_line, lines[j])
                    skip_set.add(j)
                    merged_something = True
            new_lines.append(merged_line)
            skip_set.add(i)

        # Because we might have merged multiple lines into one, we can have duplicates in new_lines
        # Let's keep them unique by IDs or by sorting them out. A simple approach:
        # (Here, we do a naive approach: if lines are literally the same object, remove duplicates.)
        # A more robust approach might re-check merges among new_lines in the same pass, etc.
        unique_new = []
        for ln in new_lines:
            if all(not np.array_equal(ln["points"], unq["points"]) for unq in unique_new):
                unique_new.append(ln)
        lines = unique_new

    return lines


def arc_parameters(arc_dict):
    """
    Return the (cx, cy, r) for an arc_dict (stub).
    """
    circle_params = fit_circle_to_points(arc_dict["points"])
    return circle_params


def are_cocircular_and_connectable(arcA, arcB, center_threshold=1e-2, radius_threshold=1e-2):
    """
    Check if arcA and arcB belong to the same circle (same center/radius within threshold).
    In a true merging scenario for arcs, we'd also consider if the arcs overlap in angle, etc.
    """
    cA = arc_parameters(arcA)
    cB = arc_parameters(arcB)
    if cA is None or cB is None:
        return False
    cxA, cyA, rA = cA
    cxB, cyB, rB = cB

    center_dist = math.hypot(cxA - cxB, cyA - cyB)
    if center_dist > center_threshold:
        return False
    if abs(rA - rB) > radius_threshold:
        return False

    # A robust approach would check arcs' angle ranges for overlap or adjacency.
    return True


def merge_two_arcs(arcA, arcB):
    """
    Merge two arcs into a single arc dict.
    Combine indices and points, possibly reorder by angle around center, etc.
    """
    new_indices = list(set(arcA["indices"] + arcB["indices"]))
    all_points = np.vstack((arcA["points"], arcB["points"]))
    # For a real system, you'd order by angle from the fitted circle center.
    # We'll just store them in unique + sorted order for demonstration:
    all_points_unique = np.unique(all_points, axis=0)
    # Naive approach: sort by x then by y
    all_points_sorted = all_points_unique[np.lexsort((all_points_unique[:, 1], all_points_unique[:, 0]))]
    return {
        "indices": new_indices,
        "points": all_points_sorted
    }


def merge_cocircular_arcs(arcs, center_threshold=1e-2, radius_threshold=1e-2):
    """
    Repeatedly merge arcs that share the same circle center+radius within thresholds.
    """
    merged_something = True
    while merged_something and len(arcs) > 1:
        merged_something = False
        new_arcs = []
        skip_set = set()

        for i in range(len(arcs)):
            if i in skip_set:
                continue
            merged_arc = arcs[i]
            for j in range(i + 1, len(arcs)):
                if j in skip_set:
                    continue
                if are_cocircular_and_connectable(merged_arc, arcs[j], center_threshold, radius_threshold):
                    merged_arc = merge_two_arcs(merged_arc, arcs[j])
                    skip_set.add(j)
                    merged_something = True
            new_arcs.append(merged_arc)
            skip_set.add(i)

        # Remove duplicates in new_arcs
        unique_new = []
        for arc in new_arcs:
            if all(not np.array_equal(arc["points"], unq["points"]) for unq in unique_new):
                unique_new.append(arc)
        arcs = unique_new

    return arcs
