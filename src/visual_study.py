# Created by MacBook Pro at 18.06.25

import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os
import colorsys
import json
import random
import torch
from typing import List

import config
from mbg import patch_preprocess
from mbg.object import eval_patch_classifier
from src import bk
from mbg.group import eval_groups
from mbg.scorer import scorer_config


def adjust_color_brightness(color_rgb, factor=1.5):
    """Brighten or saturate a color slightly for better visibility."""
    r, g, b = color_rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1.0, l * factor)
    s = min(1.0, s * factor)
    return colorsys.hls_to_rgb(h, l, s)


def visual_objs(img, objs, meta_data, save_path):
    img_np = TF.to_pil_image(img.cpu()).convert("RGB")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.axis("off")

    for o_i in range(len(objs)):
        x, y = objs[o_i]['s']['x'], objs[o_i]['s']['y']
        w, h = objs[o_i]['s']['w'], objs[o_i]['s']['h']
        color = [c / 255 for c in objs[o_i]['s']['color']]
        bright_color = adjust_color_brightness(color, factor=2.0)

        abs_x = x * img_np.width
        abs_y = y * img_np.height
        abs_w = w * img_np.width
        abs_h = h * img_np.height

        # Predicted shape
        shape_idx = objs[o_i]['s']['shape'].argmax().item()
        shape_name = bk.bk_shapes_2[shape_idx]
        gt_label = meta_data[o_i]["shape"]

        if shape_name == gt_label:
            # Correct detection
            rect = patches.Rectangle(
                (abs_x, abs_y), abs_w, abs_h,
                linewidth=2.5, edgecolor=bright_color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                abs_x, abs_y - 5,
                f"#{objs[o_i]['id']} {shape_name}",
                color=bright_color,
                fontsize=8, weight='bold',
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1)
            )
        else:
            # Incorrect detection — highlight in red dashed box
            rect = patches.Rectangle(
                (abs_x, abs_y), abs_w, abs_h,
                linewidth=2.5, edgecolor='red', linestyle='--', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                abs_x, abs_y - 5,
                f"#{objs[o_i]['id']} {shape_name} ≠ {gt_label}",
                color='red',
                fontsize=8, weight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='red', pad=1)
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved visualization to {save_path}")


def get_meta_data(file_name):
    with open(file_name) as f:
        metadata = json.load(f)
    objects = metadata.get("img_data", [])
    return objects


def visualize_group_predictions(img: torch.Tensor, grp_preds: List[dict], save_path: Path):
    """
    Visualize grouped object predictions as bounding boxes.

    Each group is displayed with a unique colored box that tightly bounds all its member objects.

    Args:
        img: [3, H, W] RGB image tensor
        grp_preds: list of group dicts, each with a 'members' key containing symbolic data
        save_path: where to save the visualization
    """
    img_np = TF.to_pil_image(img.cpu()).convert("RGB")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.axis("off")

    # Pick N distinct colors
    cmap = plt.get_cmap("tab10")
    num_colors = len(grp_preds)

    for i, group in enumerate(grp_preds):
        members = group["members"]
        if len(members) == 0:
            continue

        # Compute absolute bounding box over group members
        xs, ys, ws, hs = [], [], [], []
        for m in members:
            xs.append(m["x"])
            ys.append(m["y"])
            ws.append(m["w"])
            hs.append(m["h"])

        # Convert from relative to absolute
        img_w, img_h = img_np.width, img_np.height
        x_abs = torch.tensor(xs) * img_w
        y_abs = torch.tensor(ys) * img_h
        w_abs = torch.tensor(ws) * img_w
        h_abs = torch.tensor(hs) * img_h

        x0 = torch.min(x_abs)
        y0 = torch.min(y_abs)
        x1 = torch.max(x_abs + w_abs)
        y1 = torch.max(y_abs + h_abs)

        # Draw group bounding box
        group_color = cmap(i % 10)
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             linewidth=2.5, edgecolor=group_color, linestyle='-', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0 + 3, y0 - 5, f"Group {i}", color=group_color, fontsize=8, weight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved group visualization to {save_path}")


def visual_obj_tasks(obj_preds, meta, axs, i, imgs):
    # Plot on corresponding subplot
    ax = axs[i // 2][i % 2]
    img_np = TF.to_pil_image(imgs[0].cpu()).convert("RGB")
    ax.imshow(img_np)
    ax.axis("off")

    for o_i in range(len(obj_preds)):
        x, y = obj_preds[o_i]['s']['x'], obj_preds[o_i]['s']['y']
        w, h = obj_preds[o_i]['s']['w'], obj_preds[o_i]['s']['h']
        color = [c / 255 for c in obj_preds[o_i]['s']['color']]
        bright_color = adjust_color_brightness(color, factor=2.0)

        abs_x = x * img_np.width
        abs_y = y * img_np.height
        abs_w = w * img_np.width
        abs_h = h * img_np.height

        shape_idx = obj_preds[o_i]['s']['shape'].argmax().item()
        shape_name = bk.bk_shapes_2[shape_idx]
        gt_label = meta[o_i]["shape"]

        # Determine whether prediction matches GT (allowing pac_man ≈ circle)
        is_correct = (
                shape_name == gt_label or
                (gt_label == "pac_man" and shape_name == "circle")
        )

        if is_correct:
            rect = plt.Rectangle((abs_x, abs_y), abs_w, abs_h,
                                 linewidth=2.0, edgecolor=bright_color, facecolor='none')
            ax.add_patch(rect)
            ax.text(abs_x, abs_y - 5,
                    f"#{obj_preds[o_i]['id']} {shape_name}",
                    color=bright_color, fontsize=8, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1))
        else:
            rect = plt.Rectangle((abs_x, abs_y), abs_w, abs_h,
                                 linewidth=2.0, edgecolor='red', linestyle='--', facecolor='none')
            ax.add_patch(rect)
            ax.text(abs_x, abs_y - 5,
                    f"#{obj_preds[o_i]['id']} {shape_name} ≠ {gt_label}",
                    color='red', fontsize=8, weight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='red', pad=1))

    return axs


def visual_grp_tasks(imgs, grp_preds, axs, i):
    # Use existing plotting axis instead of saving separately
    ax = axs[i // 2][i % 2]
    img_np = TF.to_pil_image(imgs[0].cpu()).convert("RGB")
    ax.imshow(img_np)
    ax.axis("off")

    cmap = plt.get_cmap("tab10")

    for g_i, group in enumerate(grp_preds):
        members = group["members"]
        if len(members) == 0:
            continue

        xs, ys, ws, hs = zip(*[(m["s"]["x"], m["s"]["y"], m["s"]["w"], m["s"]["h"]) for m in members])
        img_w, img_h = img_np.width, img_np.height
        x0 = min(x * img_w for x in xs)
        y0 = min(y * img_h for y in ys)
        x1 = max((x + w) * img_w for x, w in zip(xs, ws))
        y1 = max((y + h) * img_h for y, h in zip(ys, hs))

        color = cmap(g_i % 10)
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             linewidth=2.5, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x0 + 3, y0 - 5, f"Group {g_i}", color=color, fontsize=8, weight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    return axs


def visualize_multiple_tasks(device="cpu", save_path=None):
    save_path = save_path or config.output / "od_visual.png"
    obj_model = eval_patch_classifier.load_model(device)
    group_model_closure = scorer_config.load_scorer_model("closure", device)

    # Get closure task folders
    closure_root = config.grb_closure / "train"
    closure_folders = sorted(os.listdir(closure_root))
    sampled_folders = random.sample(closure_folders, 4)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig_grp, axs_grp = plt.subplots(2, 2, figsize=(10, 10))
    for i, folder_name in enumerate(sampled_folders):
        img_path = closure_root / folder_name / "positive" / "00000.png"
        meta_path = closure_root / folder_name / "positive" / "00000.json"

        # Load and evaluate
        imgs = patch_preprocess.load_images_fast([img_path], device=device)
        meta = get_meta_data(meta_path)
        obj_preds = eval_patch_classifier.evaluate_image(obj_model, imgs[0], device)
        # Align predictions
        meta, obj_preds, _ = patch_preprocess.align_gt_data_and_pred_data(meta, obj_preds)
        axs = visual_obj_tasks(obj_preds, meta, axs, i, imgs)

        grp_preds = eval_groups.eval_groups(obj_preds, group_model_closure, "closure", device, dim=7)
        axs_grp = visual_grp_tasks(imgs, grp_preds, axs_grp, i)
        # ax.set_title(folder_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved 2x2 visualization grid to {save_path}")


def visualize_multiple_tasks_grp(device="cpu", save_path=None):

    prin = "proximity"

    for prin in ["proximity", "closure", "similarity", "continuity", "symmetry"]:
        obj_model = eval_patch_classifier.load_model(device)
        group_model_closure = scorer_config.load_scorer_model(prin, device)

        # Get closure task folders
        if prin == "closure":
            prin_root = config.grb_closure / "train"
        elif prin == "proximity":
            prin_root = config.grb_proximity / "train"
        elif prin == "continuity":
            prin_root = config.grb_continuity / "train"
        elif prin == "similarity":
            prin_root = config.grb_similarity / "train"
        elif prin == "symmetry":
            prin_root = config.grb_symmetry / "train"
        else:
            raise ValueError
        task_folders = sorted(os.listdir(prin_root))
        sampled_folders = random.sample(task_folders, 4)

        # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig_grp, axs_grp = plt.subplots(2, 2, figsize=(10, 10))
        for i, folder_name in enumerate(sampled_folders):
            img_path = prin_root / folder_name / "positive" / "00000.png"
            meta_path = prin_root / folder_name / "positive" / "00000.json"

            # Load and evaluate
            imgs = patch_preprocess.load_images_fast([img_path], device=device)
            meta = get_meta_data(meta_path)
            obj_preds = eval_patch_classifier.evaluate_image(obj_model, imgs[0], device)
            # Align predictions
            meta, obj_preds, _ = patch_preprocess.align_gt_data_and_pred_data(meta, obj_preds)
            # axs = visual_obj_tasks(obj_preds, meta, axs)

            grp_preds = eval_groups.eval_groups(obj_preds, group_model_closure, "closure", device, dim=7)
            axs_grp = visual_grp_tasks(imgs, grp_preds, axs_grp, i)
            # ax.set_title(folder_name)

        plt.tight_layout()
        plt.savefig(config.output / f"group_vis_grid_{prin}.png", dpi=200)
        plt.close()


def main():
    visualize_multiple_tasks_grp()
    # visualize_multiple_tasks()


if __name__ == "__main__":
    main()
