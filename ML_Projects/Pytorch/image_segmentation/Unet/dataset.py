""" Dataset module for UNET experiment

This contains a pytorch dataset to load carvana set for images and masks for
UNET.

Classes:
    DoubleConv: A double convolution layer with batch normalization and ReLU
        activation.

author: Shane Moran
(c) 2023 Shane Moran. All rights reserved.
"""

from utils import display_images
import glob as gb
import os
from typing import Optional
import torchvision.transforms.v2 as T2
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
import torch
from config import (RANDOM_STATE, IMG_HEIGHT, IMG_WIDTH, DATA_FOLDER,
                    IMG_FOLDER, MASKS_FOLDER, TRAIN_FRACTION, BATCH_SIZE)


class CarvanaDataSet(Dataset):
    """Carvana based image segmentation dataset

    Args:
        image_dir (str): Absolute path of image directory.
        masks_dir (str): Absolute path of masks directory.
        image_list (list[str]): List of image names in dataset (no extension).
        transform (torchvision.transforms.v2.Compose, optional): A function
            that transforms an image.
    """

    def __init__(
        self,
        image_dir: str,
        masks_dir: str,
        image_list: list[str],
        transform: Optional[T2.Compose] = None,
    ):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.image_list = image_list
        self.transform = T2.Compose(
            [T2.ToImageTensor(), T2.ConvertDtype()])
        if transform is not None:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return item given an index

        Given an index returns the tuple pair of image and mask.

        Args:
            idx (int): index of item.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Image and masks tensors.
        """

        image = Image.open(os.path.join(self.image_dir, self.image_list[idx]))
        mask = Image.open(
            os.path.join(self.masks_dir, self.image_list[idx]).replace(
                ".jpg", "_mask.gif"
            )
        )

        image, mask = self.transform(image, mask)

        return image, mask


transforms = T2.Compose(
    [T2.Resize((IMG_HEIGHT, IMG_WIDTH)),
     T2.ToImageTensor(),
     T2.ConvertDtype()])


ROOT_DIR = os.path.abspath("")
IMAGE_DIR = os.path.join(ROOT_DIR, DATA_FOLDER, IMG_FOLDER)
MASKS_DIR = os.path.join(ROOT_DIR, DATA_FOLDER, MASKS_FOLDER)
IMAGE_LIST = [
    os.path.basename(x) for x in sorted(gb.glob(
        os.path.join(IMAGE_DIR, "*.jpg")))
]

segmentationDataset = CarvanaDataSet(
    image_dir=IMAGE_DIR,
    masks_dir=MASKS_DIR,
    image_list=IMAGE_LIST,
    transform=transforms,
)

generator = torch.Generator().manual_seed(RANDOM_STATE)
train_dataset, test_dataset = random_split(segmentationDataset,
                                           [TRAIN_FRACTION,
                                            1.0-TRAIN_FRACTION])


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    img, mask = segmentationDataset[11]
    display_images(img, mask)
