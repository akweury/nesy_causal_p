# Created by shaji at 25/07/2024

import torch
import torch.nn.functional as F  # Import F for functional operations
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from einops import rearrange, repeat
from torch import nn, einsum

import config
from src.utils import data_utils, visual_utils, file_utils, chart_utils


class FCN(nn.Module):
    def __init__(self, in_channels):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Final fully connected layer for classification
        self.fc = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x)
        x = torch.sigmoid(self.fc(x))
        return x.view(-1, 2)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 3)  # Assuming 3 classes

    def forward(self, x, mask=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))

        if mask is not None:
            x = x * mask  # Apply the mask to the input of the FC layer

        x = self.fc2(x)
        return x


class ShapeDataset(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform

        self.image_paths = []
        # self.labels = []
        self.device = args.device
        folder = config.kp_dataset / args.exp_name
        imgs = file_utils.get_all_files(folder, "png", False)[:1000]
        # labels = [self.get_label(args.exp_name) for img in imgs]
        self.image_paths += imgs
        # self.labels += labels

    def get_label(self, img_name):
        if 'trianglesquare' in img_name:
            return torch.tensor([0, 0, 1], dtype=torch.float)
        elif 'trianglecircle' in img_name:
            return torch.tensor([0, 1, 0], dtype=torch.float)
        elif 'triangle' in img_name:
            return torch.tensor([1, 0, 0], dtype=torch.float)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_name, file_extension = self.image_paths[idx].split(".")
        data = file_utils.load_json(f"{file_name}.json")
        patch = data_utils.oco2patch(data).unsqueeze(0).to(self.device)
        # label = self.labels[idx].to(self.device)
        return patch


class ContinueShapeDataset(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform

        self.image_paths = []
        # self.labels = []
        self.device = args.device
        folder = config.kp_dataset / args.exp_name
        imgs = file_utils.get_all_files(folder, "png", False)[:1000]
        # labels = [self.get_label(args.exp_name) for img in imgs]
        self.image_paths += imgs
        # self.labels += labels

    def get_label(self, img_name):
        if 'trianglesquare' in img_name:
            return torch.tensor([0, 0, 1], dtype=torch.float)
        elif 'trianglecircle' in img_name:
            return torch.tensor([0, 1, 0], dtype=torch.float)
        elif 'triangle' in img_name:
            return torch.tensor([1, 0, 0], dtype=torch.float)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = data_utils.load_bw_img(self.image_paths[idx], size=64)
        # resize
        # file_name, file_extension = self.image_paths[idx].split(".")
        # data = file_utils.load_json(f"{file_name}.json")
        # patch = data_utils.oco2patch(data).unsqueeze(0).to(self.device)

        return img


class MaskOptimizer:
    def __init__(self, input_dim, target_label, lr=0.01):
        self.target_label = target_label
        self.mask = nn.Parameter(torch.randn(input_dim))  # Initialize the mask
        self.optimizer = optim.Adam([self.mask], lr=lr)

    def get_mask(self):
        return torch.sigmoid(self.mask)  # Use sigmoid to constrain mask values between 0 and 1


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embed


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),  # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=128, num_heads=4, num_layers=6,
                 num_classes=10):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_encoding = PositionalEncoding(self.patch_embed.n_patches, embed_dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = Transformer(16, num_layers, num_heads, 64, embed_dim, 0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)

        x = x.transpose(0, 1)  # (N+1, B, embed_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # (B, N+1, embed_dim)

        cls_token_final = x[:, 0]
        logits = self.mlp_head(cls_token_final)

        return logits


def train_percept(args, model, train_loader, val_loader):
    num_epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        criterion = nn.CrossEntropyLoss()

        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images.to(args.device))
            # label_zero = torch.zeros_like(outputs).to(args.device)
            # for l_i, label in enumerate(labels):
            #     label_zero[l_i, label] = 1.0
            loss = criterion(outputs, labels.to(args.device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # # Validation loop
        # model.eval()
        # val_loss = 0.0
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         outputs = model(inputs.to(args.device))
        #         labels = labels.float().to(args.device)
        #         total += labels.size(0)
        #         pred_labels = outputs.argmax(dim=1)
        #         # gt_labels = labels.argmax(dim=1)
        #         correct += (pred_labels == labels).sum().item()
        #
        # avg_val_loss = val_loss / len(val_loader)
        # accuracy = 100 * correct / total

        wandb.log({'train_loss': running_loss / train_loader.dataset.__len__()})


def optimize_mask(model, train_loader, val_loader, mask_optimizer, target_label, num_steps=10):
    model.eval()

    for step in tqdm(range(num_steps)):
        train_loss = 0.0
        for images, _ in train_loader:
            mask_optimizer.optimizer.zero_grad()
            mask = mask_optimizer.get_mask()
            outputs = model(images, mask=mask)

            # We want to maximize the logit corresponding to the target label
            target_logit = outputs[:, target_label].mean()
            loss = -target_logit  # Negate because we want to maximize

            loss.backward()
            mask_optimizer.optimizer.step()

            train_loss += loss.item()

        # Validation loop
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                mask = mask_optimizer.get_mask()
                outputs = model(inputs, mask=mask)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == target_label).sum().item()

        acc = 100 * correct / total
        avg_val_loss = train_loss / len(train_loader)

        wandb.log({'mask_loss': avg_val_loss,
                   "val_acc": acc})


def kmeans_common_features(args, model, train_loader, val_loader):
    model.eval()

    class_features = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="extracting features"):
            x = F.relu(model.conv1(images.to(args.device)))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(model.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(model.conv3(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 128 * 6 * 6)
            features = F.relu(model.fc1(x))
            for feature, label in zip(features, labels):
                class_features[int(label.argmax().item())].append(feature.numpy())

    # Convert lists to numpy arrays
    for cls in class_features:
        class_features[cls] = np.array(class_features[cls])

    # Find the mean feature vector for each class
    mean_features = {cls: np.mean(class_features[cls], axis=0) for cls in class_features}

    # Find common features by calculating the intersection
    common_features = np.mean([mean_features[cls] for cls in mean_features], axis=0)

    # return the indices of common features
    return common_features


def one_layer_conv(data, kernels):
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
    max_value = kernels.sum(dim=[1, 2, 3])
    max_value = max_value.unsqueeze(1).unsqueeze(2).unsqueeze(0)
    max_value = torch.repeat_interleave(max_value, output.shape[2], dim=-2)
    max_value = torch.repeat_interleave(max_value, output.shape[3], dim=-1)
    mask = (max_value == output).to(torch.float32)
    return mask


def detect_edge(matrices):
    """
    Detect edges in a batch of binary matrices.

    Args:
        matrices (torch.Tensor): A batch of binary matrices of shape (N, 1, H, W), where N is the batch size.

    Returns:
        torch.Tensor: A batch of edge-detected matrices of shape (N, 1, H, W), where edges are marked as 1 and others as 0.
    """

    # Define the edge-detection kernel
    edge_kernel = torch.tensor([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Expand the kernel to apply it separately to each channel
    edge_kernel_repeated = edge_kernel.repeat(matrices.size(1), 1, 1, 1)

    # Apply convolution across the batch
    edges = F.conv2d(matrices, edge_kernel_repeated, groups=matrices.size(1), padding=1)

    # Convert to binary (edge pixels as 1, others as 0)
    edges_binary = (edges > 0).float()
    edges_binary[:, :, 0] = 0
    edges_binary[:, :, -2:] = 0
    edges_binary[:, :, :, :2] = 0
    edges_binary[:, :, :, -2:] = 0
    # Remove the channel dimension to return a batch of (N, H, W)
    return edges_binary


def matrix_similarity(mat1, mat2):
    # Flatten the matrices from (N, W, H) to (N*W*H)
    g1_flat = mat1.view(mat1.size(0), -1)  # Shape: (N1, 192 * 64 * 64)
    g2_flat = mat2.view(mat2.size(0), -1)  # Shape: (N2, 192 * 64 * 64)
    # Compute cosine similarity between all pairs (N1 x N2 matrix)
    similarity_matrix = torch.mm(g1_flat, g2_flat.t()) / (g2_flat.sum(dim=1).unsqueeze(0)+1e-20)  # Shape: (N1, N2)
    return similarity_matrix


def matrix_equality(matrix1, matrix2):
    """
    Calculate the normalized equal item count between two matrices.

    Parameters:
    - matrix1: np.ndarray, first matrix.
    - matrix2: np.ndarray, second matrix.

    Returns:
    - float: Normalized equal item count in the range [0, 1].
    """
    # Ensure input matrices have the same number of columns
    matrix1_flatten = matrix1.sum(dim=1).view(matrix1.size(0), -1)
    matrix2_flatten = matrix2.sum(dim=1).view(matrix2.size(0), -1)
    num_features = matrix2.sum(dim=[1, 2, 3])

    batch_size = 128
    similarity_matrix = torch.zeros((matrix1.shape[0], matrix2.shape[0]))
    for i in tqdm(range(0, matrix1.shape[0], batch_size), desc="Calculating Equality"):
        end_i = min(i + batch_size, matrix1.shape[0])
        batch1 = matrix1_flatten[i:end_i].unsqueeze(1).bool()
        batch2 = matrix2_flatten.unsqueeze(0).bool()

        # Sum over the feature dimension to count matches
        equal_counts = (batch1 * batch2).sum(dim=2)  # Shape: (4096, 197)

        # Normalize by the number of features to get similarity in range [0, 1]
        similarity_matrix[i:end_i] = equal_counts / (num_features + 1e-20)

    return similarity_matrix
