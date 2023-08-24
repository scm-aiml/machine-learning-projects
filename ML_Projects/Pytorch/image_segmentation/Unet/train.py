"""Training loop for UNET image segmentation

Contains code to load the model and perform training on train
and test data from Carvana dataset.

author: Shane Moran
(c) 2023 Shane Moran. All rights reserved.
"""

from model import UNET
from config import LAYER_CHANNELS, OUT_CHANNELS, DEVICE, NUM_EPOCHS
import torch
from dataset import train_dataloader

model = UNET(channels=LAYER_CHANNELS, out_channels=OUT_CHANNELS)
model.to(DEVICE)
# model = torch.compile(model)


if __name__ == "__main__":
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        for imgs, masks in train_dataloader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            output = model(imgs)
            loss = loss_fn(output, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss

            print(epoch_loss)
