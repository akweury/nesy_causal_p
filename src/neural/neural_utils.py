# Created by jing at 11.12.24

import torch
from torch.utils.data import DataLoader, TensorDataset


def prepare_fm_data(args):

    # Sample data
    fm_file = args.save_path / f'fms_patches_{args.k_size}.pt'
    data = torch.load(fm_file)

    # Normalize the data
    data = (data - data.min()) / (data.max() - data.min())

    # Define a PyTorch Dataset and DataLoader
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    fm_channels = data.shape[1]
    return dataloader,  fm_channels

