# Created by MacBook Pro at 15.04.25
import os
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import config


# Dataset for shape images
class ShapeDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        return img, img_path


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1,
                        help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding extraction")
    args = parser.parse_args()

    # Determine device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load model
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC layer
    model.eval().to(device)

    # image_dir = config.mb_outlines
    output_dir = config.mb_outlines
    os.makedirs(output_dir, exist_ok=True)
    shape_classes = ["triangle", "rectangle", "ellipse"]

    for shape in shape_classes:
        print(f"\nProcessing shape class: {shape}")
        image_dir = os.path.join(config.mb_outlines, shape)
        dataset = ShapeDataset(image_dir, transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        all_embeddings = []
        with torch.no_grad():
            for imgs, _ in tqdm(dataloader, desc=f"Embedding {shape}"):
                imgs = imgs.to(device)
                features = model(imgs).squeeze(-1).squeeze(-1)
                all_embeddings.append(features.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        output_path = os.path.join(output_dir, f"{shape}_embeddings.npy")
        np.save(output_path, embeddings)
        print(f"Saved {shape} embeddings to: {output_path}")


if __name__ == "__main__":
    main()
