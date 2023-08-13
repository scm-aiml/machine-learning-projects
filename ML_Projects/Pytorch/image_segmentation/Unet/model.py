"""
Pytorch implementation of Unet

This provides a simple implementation of the Unet architecture for semantic (image) segmentation.

Classes:
    DoubleConv: A double convolutional layer with batch normalization and ReLU activation.

author: Shane Moran
(c) 2023 Shane Moran. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as FF

class DoubleConv(nn.Module):
    """ A double convolutional layer in the Unet architecture.
    
    A series of two convolutional layers with batch normalization followed by ReLU activation.

    Args:
        in_chan (int): The number input channels
        out_chan (int): The number of output channels
        kernel_size (int): The size of the convolutional kernel
        stride (int): The stride of the convolutional kernel
        padding (int): The padding of the convolutional kernel

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer
        conv2 (nn.Conv2d): The second convolutional layer
        batchnorm1 (nn.BatchNorm2d): The first batch normalization layer
        batchnorm2 (nn.BatchNorm2d): The second batch normalization layer
        relu1 (nn.ReLU): The first ReLU activation layer
        relu2 (nn.ReLU): The second ReLU activation layer
    """
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        """ Initializes the DoubleConv class

        Initialize the DobuleConv class with the defined input and output channels. The kernel size, stride, and padding are all set to 3, 1, and 1 respectively for a same convolution.

        Args:
            in_chan (int): The number input channels
            out_chan (int): The number of output channels
            kernel_size (int, optional): The size of the convolutional kernel
            stride (int, optional): The stride of the convolutional kernel
            padding (int, optional): The padding of the convolutional kernel
        """
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(
            in_chan,
            out_chan, 
            kernel_size, 
            stride = stride, 
            padding = padding
        )
        self.batchnorm1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(
            out_chan, 
            out_chan, 
            kernel_size, 
            stride = stride, 
            padding = padding
        )
        self.batchnorm2 = nn.BatchNorm2d(out_chan)
        self. relu2 = nn.ReLU(inplace = True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of DoubleConv class

        The forward pass of the DoubleConv class. This passes the input tensor through two rounds of Conv2d, batch normalization, and a ReLU activation.

        Args:
            x (torch.Tensor): An input tensor

        Returns:
            torch.Tensor: The output tensor
        """
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)

        return x





if __name__ == "__main__":
    print("Hello World")
