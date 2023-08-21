""" Dataset module for UNET experiment

This contains a pytorch dataset to load carvana set for images and masks for UNET.

Classes:
    DoubleConv: A double convolution layer with batch normalization and ReLU activation.

author: Shane Moran
(c) 2023 Shane Moran. All rights reserved.
"""

import glob as gb
import os
from collections.abc import Callable
from typing import Optional
import torchvision.transforms.v2 as T2
from PIL import Image
from torch.utils.data import Dataset


class CarvanaDataSet(Dataset):
    """Carvana based image segmentation dataset

    Args:
        image_dir (str): Absolute path of image directory.
        masks_dir (str): Absolute path of masks directory.
        image_list (list[str]): List of image names in dataset (no extension).
        transform (Callable, optional): A function that transforms an image.
    """

    def __init__(
        self,
        image_dir: str,
        masks_dir: str,
        image_list: list[str],
        transform: Optional[Callable] = None,
    ):
        self.image_dir = image_dir
        self.masks_dir = masks_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx: int):
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

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask


transforms = T2.Compose([T2.Resize((208, 304)), T2.ToImageTensor(), T2.ConvertDtype()])


ROOT_DIR = os.path.abspath("")
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "images")
MASKS_DIR = os.path.join(ROOT_DIR, "data", "masks")
IMAGE_LIST = [
    os.path.basename(x) for x in sorted(gb.glob(os.path.join(IMAGE_DIR, "*.jpg")))
]

segmentationDataset = CarvanaDataSet(
    image_dir=IMAGE_DIR,
    masks_dir=MASKS_DIR,
    image_list=IMAGE_LIST,
    transform=transforms,
)

a, b = segmentationDataset[16]

print(a.shape)
print(b.shape)
