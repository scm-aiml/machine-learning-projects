"""Training loop for UNET image segmentation

Contains code to load the model and perform training on train
and test data from Carvana dataset.

author: Shane Moran
(c) 2023 Shane Moran. All rights reserved.
"""

from model import UNET
from config import LAYER_CHANNELS, OUT_CHANNELS, DEVICE, NUM_EPOCHS
import torch
from torch.utils.data import DataLoader
from dataset import train_dataloader, test_dataloader
from typing import Optional
from tqdm.auto import tqdm

model = UNET(channels=LAYER_CHANNELS, out_channels=OUT_CHANNELS)
model.to(DEVICE)
# model = torch.compile(model)


def accuracy_fcn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Simple accuracy function for unet

    NOTE: Accuracy should eventually be replaced by something
        like IOU or dice.
    Args:
        preds (torch.Tensor): Prediction outputs from model of
            shape (batch_size, img_height, img_width).
        targets (torch.Tensor): Target lables of same shape preds.

    Returns:
        torch.Tensor: accurcy of predictions.
    """
    pred_label = (torch.sigmoid(preds) > 0.5).float()
    return torch.mean((pred_label == targets).float())


def train_step(model: torch.nn.Module,
               dataloader: DataLoader,
               loss_fcn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: Optional[str] = str()) -> tuple[float, float]:
    """Train a pytorch model for one epoch.

    Args:
        model (torch.nn.Module): PyTorch model to be trained.
        dataloader (DataLoader): a DataLoader object containing train
            dataset.
        loss_fcn (torch.nn.Module): A loss function to be minimized
        optimizer (torch.optim.Optimizer): A optimizer to minimize
            loss_fcn
        device (str, optional): string defining device to load on
    Returns:
        tuple[float, float]: return performance values in the form
            (train_loss, train_acc).
    """
    model.train()

    train_loss, train_acc = 0.0, 0.0

    for imgs, masks in tqdm(dataloader):
        # optionally, load to device
        if device:
            imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)

        loss = loss_fcn(preds, masks)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += accuracy_fcn(preds, masks).item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: DataLoader,
              losss_fcn: torch.nn.Module,
              device: Optional[str] = str()) -> tuple[float, float]:
    """ Perform testing over a single epoch

    Args:
        model (torch.nn.Module): PyTorch model to be evaluated.
        dataloader (DataLoader): DataLoader object containing test
            dataset.
        losss_fcn (torch.nn.Module): A loss function to calculate
            loss on test dataset.
        device (str, optional): string defining device to load on

    Returns:
        tuple[float, float]: return performance values in the form
            (test_loss, test_acc).
    """

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for imgs, masks in tqdm(dataloader):
            # optionally, load to device
            if device:
                imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)

            test_loss += loss_fn(preds, masks).item()
            test_acc += accuracy_fcn(preds, masks).item()

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


def train_loop(model: torch.nn.Module,
               train_dataloader: DataLoader,
               test_dataloader: DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fcn: torch.nn.Module,
               epochs: int = 10) -> None:

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fcn=loss_fcn,
                                           optimizer=optimizer,
                                           device=DEVICE)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        losss_fcn=loss_fcn,
                                        device=DEVICE)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

    torch.save(model.state_dict(), "mymodel.pth")


if __name__ == "__main__":
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loop(model=model,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               optimizer=optimizer,
               loss_fcn=loss_fn,
               epochs=NUM_EPOCHS)
