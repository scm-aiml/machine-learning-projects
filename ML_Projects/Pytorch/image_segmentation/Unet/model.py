"""
Pytorch implementation of UNET

This provides a simple implementation of the UNET architecture for semantic (image) segmentation.

Classes:
    DoubleConv: A double convolution layer with batch normalization and ReLU activation.

author: Shane Moran
(c) 2023 Shane Moran. All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as FF

class DoubleConv(nn.Module):
    """ A double convolution layer in the Unet architecture.
    
    A series of two convolution layers with batch normalization followed by ReLU activation.
    Default kernel, padding, and stride result in a 'same' convolution, with different number
    of channels.

    Args:
        in_chan (int): The number input channels.
        out_chan (int): The number of output channels.
        kernel_size (int, optional): The size of the convolution kernel.
        stride (int, optional): The stride of the convolution kernel.
        padding (int, optional): The padding of the convolution kernel.

    Attributes:
        conv1 (nn.Conv2d): The first convolution layer.
        conv2 (nn.Conv2d): The second convolution layer.
        batchnorm1 (nn.BatchNorm2d): The first batch normalization layer.
        batchnorm2 (nn.BatchNorm2d): The second batch normalization layer.
        relu1 (nn.ReLU): The first ReLU activation layer.
        relu2 (nn.ReLU): The second ReLU activation layer.
    """
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ) -> None:
        
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
            x (torch.Tensor): An input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)

        return x

class DownwardLayers(nn.Module):
    """ The downward portion of UNET architecture

    The downward portion of the UNET architecture. This is a series of DoubleConv layers followed 
    by max-pooling layer with decreasing img size and increasing number of channels.

    Args:
        channels (list[int]): List of channels for each layer in contracting portion.
    
    Attributes:
        downward_layers (nn.ModuleList): A list of DoubleConv blocks followed by max-pooling layers.

    """

    def __init__(self, channels: list[int]) -> None:
        super(DownwardLayers, self).__init__()
        self.downward_layers = nn.ModuleList()

        # Loop over all except last channel value
        for i in range(len(channels)-1):
            self.downward_layers.append(
                DoubleConv(channels[i], channels[i+1])
            )

            # Add MaxPool2d to all but last layer
            if i < (len(channels)-2):
                self.downward_layers.append(
                    nn.MaxPool2d(kernel_size=2)
                )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """ Forward pass of the downward layers

        Forward pass of the downward layers of the UNET architecture. This will also return
        the output list of tensors from each DoubleConv block that is used in concatenation
        with the upward portion of the UNET architecture.

        Args:
            x (torch.Tensor): An input tensor representing image.

        Returns:
            list[torch.Tensor]: A list of the output tensors from each DoubleConv block.
        """ 
        outputs = []
        for step in self.downward_layers:
            x = step(x)

            if isinstance(step, DoubleConv):
                outputs.append(x)
        
        return outputs
    
class UpwardLayers(nn.Module):
    """ The upward portion of the UNET architecture

    The upward portion of the UNET architecture. This is a series of up-convolutions and concatenations 
    with corresponding layer of the downward portion, followed by DoubleConv.

    Args:
        channels (list[int]): List of channels for each layer in the expanding portion.
    
    Attributes:
        upward_layers (nn.ModuleList): A list of up-convolution blocks.

    """
    def __init__(self, channels: list[int]) -> None:
        super(UpwardLayers, self).__init__()
        self.upward_layers = nn.ModuleList()

        for i in range(len(channels)-1):
            self.upward_layers.append(
                nn.ConvTranspose2d(
                channels[i], channels[i+1], 2, 2
                )
            )

            self.upward_layers.append(
                DoubleConv(
                channels[i], channels[i+1]
                )
            )
    
    def forward(self, x: torch.Tensor, downward_features: list[torch.Tensor]) -> torch.Tensor:
        """ forward pass for the upward portion of UNET
        
        Args:
            x (torch.Tensor): An input tensor.
            downward_features (list[torch.Tensor]): List of tensor features from the downward portion.
        
        Returns:
            torch.Tensor: Output feature tensor. 
        """

        for i, step in enumerate(self.upward_layers):
            # concat feature from contract layer of DoubleConv
            if isinstance(step, DoubleConv):
                # tensor size batch,channel,H,W -> concat along chan
                x = torch.cat([downward_features[i//2],x], dim=1)
            
            # Apply the step
            x = step(x)
    
        return x

class UNET(nn.Module):
    """UNET Architecture model
    
    Args:
        out_channels (int): the number of final output channels.
        channels (list[int]): A list of channels for the convolutions at each layer.

    Attributes:
        downward (DownwardLayers): The downward layers of UNET.
        upward (UpwardLayers): The upward layers of UNET.
        output (nn.Conv2d): The final layer.

    Example:
        >>> model = UNET(channels=[3, 64, 128, 256, 512], out_channels = 1)
    """    

    def __init__(
            self,
            channels: list[int],
            out_channels: int
    ) -> None:
        super(UNET,self).__init__()
        self.downward = DownwardLayers(channels=channels)
        # Need to reverse order and omit last
        self.upward = UpwardLayers(channels=channels[::-1][:-1])
        self.output = nn.Conv2d(channels[1],out_channels=out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for UNET architecture
        
        Args:
            x (torch.Tensor): An input tensor.

        """
        # reverse order to pass into expanding layer
        downward_features = self.downward(x)[::-1]
        x = self.upward(downward_features[0], downward_features[1:] )
        x = self.output(x)

        return x
    
 
if __name__ == "__main__":
    # simple model test
    channel_list = [3, 64, 128, 256, 512, 1024]
    out_channels = 1
    model = UNET(channels=channel_list, out_channels=out_channels)

    in_tensor = torch.randn(1,3,256,256)
    print(type(in_tensor))
    out = model(in_tensor)

    print(out.shape)
