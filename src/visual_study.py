# Created by MacBook Pro at 18.06.25

import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os

import config
from mbg import patch_preprocess
from mbg.object import eval_patch_classifier


def visual_objs(img, objs, save_path):
    # Convert tensor to numpy image for plotting
    img_np = TF.to_pil_image(img.cpu()).convert("RGB")

    # Setup plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    ax.axis("off")

    # Draw bounding boxes
    for obj in objs:
        x, y = obj['s']['x'], obj['s']['y']
        w, h = obj['s']['w'], obj['s']['h']
        color = [c / 255 for c in obj['s']['color']]

        # Rescale from relative coordinates to image size
        abs_x = x * img_np.width
        abs_y = y * img_np.height
        abs_w = w * img_np.width
        abs_h = h * img_np.height

        rect = patches.Rectangle(
            (abs_x - abs_w / 2, abs_y - abs_h / 2), abs_w, abs_h,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Optional: annotate with shape index
        shape_idx = obj['s']['shape'].argmax().item()
        ax.text(abs_x, abs_y - 5, f"#{obj['id']} S{shape_idx}", color=color, fontsize=8, weight='bold')

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved visualization to {save_path}")


def main():
    device = "cpu"
    save_path = config.output / "od_visual.png"
    obj_model = eval_patch_classifier.load_model(device)
    closure_folders = sorted(os.listdir(config.grb_closure / "train"))
    prox_folders = sorted(os.listdir(config.grb_proximity / "train"))

    closure_eg = config.grb_closure / "train" / Path(closure_folders[0]) / "positive" / "00000.png"
    prox_eg = config.grb_proximity / "train" / Path(prox_folders[1]) / "positive" / "00000.png"

    imgs = patch_preprocess.load_images_fast([closure_eg], device=device)
    # imgs = patch_preprocess.load_images_fast([prox_eg], device=device)
    # Get predicted objects
    objs = eval_patch_classifier.evaluate_image(obj_model, imgs[0], device)
    visual_objs(imgs[0], objs, save_path)
    print("")


if __name__ == "__main__":
    main()
