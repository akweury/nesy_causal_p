# Created by MacBook Pro at 18.06.25

import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os
import colorsys
import json
import random

import config
from mbg import patch_preprocess
from mbg.object import eval_patch_classifier
from src import bk


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



def visualize_multiple_tasks(device="cpu", save_path=None):
    save_path = save_path or config.output / "od_visual.png"
    obj_model = eval_patch_classifier.load_model(device)

    # Get closure task folders
    closure_root = config.grb_closure / "train"
    closure_folders = sorted(os.listdir(closure_root))
    sampled_folders = random.sample(closure_folders, 4)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i, folder_name in enumerate(sampled_folders):
        img_path = closure_root / folder_name / "positive" / "00000.png"
        meta_path = closure_root / folder_name / "positive" / "00000.json"

        # Load and evaluate
        imgs = patch_preprocess.load_images_fast([img_path], device=device)
        meta = get_meta_data(meta_path)
        preds = eval_patch_classifier.evaluate_image(obj_model, imgs[0], device)

        # Align predictions
        meta, preds, _ = patch_preprocess.align_gt_data_and_pred_data(meta, preds)

        # Plot on corresponding subplot
        ax = axs[i // 2][i % 2]
        img_np = TF.to_pil_image(imgs[0].cpu()).convert("RGB")
        ax.imshow(img_np)
        ax.axis("off")

        for o_i in range(len(preds)):
            x, y = preds[o_i]['s']['x'], preds[o_i]['s']['y']
            w, h = preds[o_i]['s']['w'], preds[o_i]['s']['h']
            color = [c / 255 for c in preds[o_i]['s']['color']]
            bright_color = adjust_color_brightness(color, factor=2.0)

            abs_x = x * img_np.width
            abs_y = y * img_np.height
            abs_w = w * img_np.width
            abs_h = h * img_np.height

            shape_idx = preds[o_i]['s']['shape'].argmax().item()
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
                        f"#{preds[o_i]['id']} {shape_name}",
                        color=bright_color, fontsize=8, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1))
            else:
                rect = plt.Rectangle((abs_x, abs_y), abs_w, abs_h,
                                     linewidth=2.0, edgecolor='red', linestyle='--', facecolor='none')
                ax.add_patch(rect)
                ax.text(abs_x, abs_y - 5,
                        f"#{preds[o_i]['id']} {shape_name} ≠ {gt_label}",
                        color='red', fontsize=8, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='red', pad=1))
            #
            # if shape_name == gt_label:
            #     rect = plt.Rectangle((abs_x, abs_y), abs_w, abs_h,
            #                          linewidth=2.0, edgecolor=bright_color, facecolor='none')
            #     ax.add_patch(rect)
            #     ax.text(abs_x, abs_y - 5,
            #             f"#{preds[o_i]['id']} {shape_name}",
            #             color=bright_color, fontsize=8, weight='bold',
            #             bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', pad=1))
            # else:
            #     rect = plt.Rectangle((abs_x, abs_y), abs_w, abs_h,
            #                          linewidth=2.0, edgecolor='red', linestyle='--', facecolor='none')
            #     ax.add_patch(rect)
            #     ax.text(abs_x, abs_y - 5,
            #             f"#{preds[o_i]['id']} {shape_name} ≠ {gt_label}",
            #             color='red', fontsize=8, weight='bold',
            #             bbox=dict(facecolor='white', alpha=0.6, edgecolor='red', pad=1))

        # ax.set_title(folder_name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved 2x2 visualization grid to {save_path}")


def main():
    visualize_multiple_tasks()
    # device = "cpu"
    # save_path = config.output / "od_visual.png"
    # obj_model = eval_patch_classifier.load_model(device)
    # closure_folders = sorted(os.listdir(config.grb_closure / "train"))
    # prox_folders = sorted(os.listdir(config.grb_proximity / "train"))
    #
    #
    # task_id = 100
    # closure_eg = config.grb_closure / "train" / Path(closure_folders[task_id]) / "positive" / "00000.png"
    # closure_meta = get_meta_data(config.grb_closure / "train" / Path(closure_folders[task_id]) / "positive" / "00000.json")
    #
    # imgs = patch_preprocess.load_images_fast([closure_eg], device=device)
    #
    # objs = eval_patch_classifier.evaluate_image(obj_model, imgs[0], device)
    #
    # closure_meta, objs, permutes = patch_preprocess.align_gt_data_and_pred_data(closure_meta, objs)
    #
    # # reorder predicted objects
    # visual_objs(imgs[0], objs, closure_meta, save_path)
    # print("")


if __name__ == "__main__":
    main()
