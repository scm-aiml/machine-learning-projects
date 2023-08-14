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
        in_chan (int): The number input channels.
        out_chan (int): The number of output channels.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolutional kernel.
        padding (int): The padding of the convolutional kernel.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
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
        """ Initializes the DoubleConv class

        Initialize the DobuleConv class with the defined input and output channels. The kernel size, stride, and padding are all set to 3, 1, and 1 respectively for a same convolution.

        Args:
            in_chan (int): The number input channels.
            out_chan (int): The number of output channels.
            kernel_size (int, optional): The size of the convolutional kernel.
            stride (int, optional): The stride of the convolutional kernel.
            padding (int, optional): The padding of the convolutional kernel.
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

class contracting(nn.Module):
    """ The contracting poertion of UNET architecture

    The contracting portion of the UNET architecture. This is a series of DobubleConv layers followed 
    by maxpooling layer with decreasing img size and increasing number of channels.

    Args:
        channels (list[int]): List of channels for each layer in contracting portion.
    
    Attributes:
        contracting_layers (nn.ModuleList): A list of DoubleConv blocks followed by maxpooling layers.

    """

    def __init__(self, channels: list[int]) -> None:
        super(contracting, self).__init__()
        self.contracting_layers = nn.ModuleList()

        # Loop over all excpet last channel value
        for i in range(len(channels)-1):
            self.contracting_layers.append(
                DoubleConv(channels[i], channels[i+1])
            )

            # Add MaxPool2d to all but last layer
            if i < (len(channels)-2):
                self.contracting_layers.append(
                    nn.MaxPool2d(kernel_size=2)
                )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """ Forward pass of the contracting layers

        Args:
            x (torch.Tensor): An input tensor representing image.

        Returns:
            list[torch.Tensor]: A list of the output tensors from each DoubleConv block
        """ 
        outputs = []
        for step in self.contracting_layers:
            x = step(x)

            if isinstance(step, DoubleConv):
                outputs.append(x)
        
        return outputs
    
class expanding(nn.Module):
    """ The expanding part of the UNET architecture

    The expanding part of the UNET architecture. This is a series of upcovnolutions and concatenations 
    with corresponding layer of contracting, followed by DoubleConv.

    Args:
        channels (list[int]): List of channels for each layer in the expanding portion.
    
    Attributes:
        expanding_layers (nn.ModuleList): A list of up-convolutional blocks.

    """
    def __init__(self, channels: list[int]) -> None:
        super(expanding, self).__init__()
        self.expanding_layers = nn.ModuleList()

        for i in range(len(channels)-1):
            self.expanding_layers.append(
                nn.ConvTranspose2d(
                channels[i], channels[i+1], 2, 2
                )
            )

            self.expanding_layers.append(
                DoubleConv(
                channels[i], channels[i+1]
                )
            )
    
    def forward(self, x: torch.Tensor, contract_features: list[torch.Tensor]) -> torch.Tensor:
        """ forward pass for the expanding portion of UNET
        
        Args:
            x (torch.Tensor): An input tensor.
            contract_features (list[torch.Tensor]): List of tensor features from the contracting portion.
        
        Returns:
            torch.tesnor: Output feature tensor. 
        """

        for i, step in enumerate(self.expanding_layers):
            # concat feature from contract layer of DoubleConv
            if isinstance(step, DoubleConv):
                # tensor size batch,channel,H,W -> concat along chan
                x = torch.cat([contract_features[i//2],x], dim=1)
            
            # Apply the step
            x = step(x)
    
        return x

class UNET(nn.Module):
    """UNET Architecture model
    
    Args:
        out_channels (int): the number of final output channels.
        channels (list[int]): A list of channels for the convolutions at each layer.

    Attributes:
        contractor (contracting): The contracting layers of UNET.
        expander (expanding): The expanding layers of UNET.
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
        self.contractor = contracting(channels=channels)
        # Need to reverse order and ommit last
        self.expander = expanding(channels=channels[::-1][:-1])
        self.output = nn.Conv2d(channels[1],out_channels=out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for UNET architecture
        
        Args:
            x (torch.Tensor): An input tensor.

        """
        # reverse order to pass into expanding layer
        contract_features = self.contractor(x)[::-1]
        x = self.expander(contract_features[0], contract_features[1:] )
        x = self.output(x)

        return x
    
 
if __name__ == "__main__":
    # simple model test
    channel_list = [3, 64, 128, 256, 512, 1024]
    out_channels = 1
    model = UNET(channels=channel_list, out_channels=out_channels)

    in_tensor = torch.randn(10,3,512,512)
    print(type(in_tensor))
    out = model(in_tensor)

    print(out.shape)
