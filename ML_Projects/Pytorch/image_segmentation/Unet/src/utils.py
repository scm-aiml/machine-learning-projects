""" Set of utils for UNET model in Carvana challenge

This provides a set of helper functions for use in implementing UNET
for image segmentation in the Carvana challenge (Kaggle)

Functions:
    DoubleConv: A double convolution layer with batch normalization and ReLU
        activation.

author: Shane Moran
(c) 2023 Shane Moran. All rights reserved.
"""

import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import DataLoader


def download_data():
    """ download Carvana data from Kaggle

    """
    DATA_DIR = './data'
    # Need to download
    if len(os.listdir(DATA_DIR)) < 3:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        api.competition_download_file(
            'carvana-image-masking-challenge', file_name='train.zip',
            path=DATA_DIR, quiet=False)
        api.competition_download_file(
            'carvana-image-masking-challenge', file_name='train_masks.zip',
            path=DATA_DIR, quiet=False)

    # Need to unzip
    if len(os.listdir(DATA_DIR)) < 5:
        import zipfile

        # Train images
        with zipfile.ZipFile(os.path.join(DATA_DIR,
                                          'train.zip'), 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.rename("./data/train", "./data/images")

        # Train masks
        with zipfile.ZipFile(os.path.join(DATA_DIR,
                                          'train_masks.zip'), 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.rename("./data/train_masks", "./data/masks")


def display_images(image: torch.Tensor, mask: torch.Tensor) -> None:
    """Display image and mask

    Args:
        image (torch.Tensor): Original image of car
        mask (torch.Tensor): Image of corresponding mask
    """

    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(mask.permute(1, 2, 0))
    plt.show()


def display_comparison(model: torch.nn.Module,
                       dataloader: DataLoader,
                       img_number: int,
                       ) -> None:
    """Plot truth and predicted mask from data loader

    Args:
        model (torch.nn.Module): A pretrained PyTorch model to make prediction.
        dataloader (DataLoader): A dataload instance to get image from.
        img_number (int): Image index.
    """

    for img, mask in dataloader:
        img = img[0, ...]
        mask = mask[0, ...]

        plt.subplot(1, 3, 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.subplot(1, 3, 2)
        plt.imshow(mask.permute(1, 2, 0))

        preds = (torch.sigmoid(model(torch.unsqueeze(img, 0))) > 0.5).float()
        plt.subplot(1, 3, 3)
        preds = preds[0, ...]
        plt.imshow(preds.permute(1, 2, 0))
        plt.show()

        break


if __name__ == "__main__":
    download_data()
