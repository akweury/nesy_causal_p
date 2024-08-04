# Created by shaji at 04/08/2024


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb
from rtpt import RTPT
import cv2
import os

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

from src.utils import args_utils, log_utils
from src.percept import perception
import config


def grad_cam(model, input_image, target_layer, target_class):
    model.eval()

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    features, gradients = None, None
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    output = model(input_image)
    target_score = output[:, target_class]

    model.zero_grad()
    target_score.backward(retain_graph=True)

    handle_forward.remove()
    handle_backward.remove()

    gradients = gradients.cpu().data.numpy()[0]
    features = features.cpu().data.numpy()[0]

    weights = np.mean(gradients, axis=(1, 2))
    cam = np.sum(weights[:, np.newaxis, np.newaxis] * features, axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

    return cam


def prepare_kp_sy_data(args):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = perception.ShapeDataset(args, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader


def main(exp_name):
    args = args_utils.get_args()
    # data file
    args.data_types = ["data_trianglesquare", "data_trianglecircle", "data_triangle"]
    train_loader, val_loader = prepare_kp_sy_data(args)
    model = perception.SimpleCNN().to(args.device)
    model_dict_path = config.output / exp_name / "detector_model.pth"
    model.load_state_dict(torch.load(model_dict_path))
    mask_path = config.output / exp_name / "mask.pth"
    # Assume val_loader is the DataLoader for the validation set
    input_dim = 128  # This should match the dimensionality of the FC layer input
    target_label = 0  # We want to maximize the logit for label 0
    mask_optimizer = perception.MaskOptimizer(input_dim, target_label)
    mask_optimizer.mask = torch.load(mask_path)

    #
    # target_layer = model.conv3  # The layer you want to visualize
    # target_class = 0  # The class index you are interested in
    # mask = mask_optimizer.get_mask()
    output_folder = config.output / exp_name / "fm_visual"
    os.makedirs(output_folder, exist_ok=True)

    for val_i, (images, labels) in tqdm(enumerate(val_loader)):
        target_class = 0  # Replace with the actual class index
        target_layers = [model.conv3]
        input_tensor = images  # Create an input tensor image for your model..
        # Note: input_tensor can be a batch tensor with several images!

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers)

        # You can also use it within a with statement, to make sure it is freed,
        # In case you need to re-create it inside an outer loop:
        # with GradCAM(model=model, target_layers=target_layers) as cam:
        #   ...

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own custom targets
        # That are, for example, combinations of categories, or specific outputs in a non standard model.

        targets = [ClassifierOutputTarget(0)]

        # In this example grayscale_cam has only one image in the batch:

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor[0:1], targets=targets)
        input_image_np = images[0].squeeze().cpu().unsqueeze(2)
        input_image_np = torch.cat([input_image_np] * 3, dim=2).numpy()
        input_image_np = (input_image_np - np.min(input_image_np)) / (
                np.max(input_image_np) - np.min(input_image_np))
        visualization = show_cam_on_image(input_image_np, grayscale_cam.squeeze(), use_rgb=True)
        resized_image = cv2.resize(visualization, (512, 512), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(str(output_folder / f'cam_{val_i}.png'), resized_image)

        # plt.savefig(output_folder / f'cam_{i * val_i}.png', bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    dataset_name = "kp_sy"
    main(dataset_name)
