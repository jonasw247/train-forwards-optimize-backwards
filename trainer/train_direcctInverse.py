import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath
from torch.utils.checkpoint import checkpoint

from utils.datasets import SyntheticDataset  # Ensure the dataset returns (downsampled_tumor, downsampled_atlas, coeff_list)

from model.directInverseConvNext import ConvNextEncoderForCoeffs  # Ensure this is the correct import for your model

# -------------------------------
# Training loop for coefficient prediction using ConvNeXtEncoderForCoeffs
# -------------------------------
def train_convnext_encoder_coeff_predictor():
    # ---------------------------------------------------
    # 1) Set up configuration and initialize wandb
    # ---------------------------------------------------
    config = {
    "dataset_size": 30000,
    "train_size": 28000,#todo
    "validation_size": 1000,
    "test_size": 1000,
    "lr": 1e-4,
    "batch_size": 2,
    "num_epochs": 100,  # Adjust epochs as needed
    "crop_size": 128,
    "down_sample_size": 128,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "project_name": "direct-inverse-neural-surrogate-in-med",
    "entity": None,
    "model": {
        "in_channels": 2,           # Concatenated tumor and atlas images
        "num_coeffs": 5,
        "n_spatial_dims": 3,
        "spatial_resolution": (128, 128, 128),
        "stages": 4,
        "blocks_per_stage": 1,
        "blocks_at_neck": 1,
        "init_features": 32,
        "gradient_checkpointing": False,
    }
    }
    
    wandb.init(project=config["project_name"], entity=config["entity"], config=config)
    c = wandb.config
    device = c.device

    # ---------------------------------------------------
    # 2) Model initialization using settings from config["model"]
    # ---------------------------------------------------
    model_cfg = c.model
    model = ConvNextEncoderForCoeffs(
        in_channels=model_cfg["in_channels"],
        num_coeffs=model_cfg["num_coeffs"],
        n_spatial_dims=model_cfg["n_spatial_dims"],
        spatial_resolution=model_cfg["spatial_resolution"],
        stages=model_cfg["stages"],
        blocks_per_stage=model_cfg["blocks_per_stage"],
        blocks_at_neck=model_cfg["blocks_at_neck"],
        init_features=model_cfg["init_features"],
        gradient_checkpointing=model_cfg["gradient_checkpointing"],
    )
    model.to(device)
    wandb.watch(model)

    # ---------------------------------------------------
    # 3) Optimizer, loss, and dataset preparation
    # ---------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=c.lr)
    criterion = nn.MSELoss()

    dataset = SyntheticDataset(length=c.dataset_size, crop_size=c.crop_size, down_sample_size=c.down_sample_size)
    train_size = c.train_size
    val_size = c.validation_size
    test_size = c.test_size

    assert len(dataset) >= train_size + val_size + test_size, "Dataset too small for specified splits."

    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - test_size, len(dataset)))

    wandb.log({
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_indices": [idx for idx in train_dataset.indices],
        "val_indices": [idx for idx in val_dataset.indices],
        "test_indices": [idx for idx in test_dataset.indices],
    })

    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=c.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # ---------------------------------------------------
    # 4) Training and Validation Loop
    # ---------------------------------------------------
    best_val_loss = float('inf')
    model_path = f"/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-direct-inverse/checkpoint/model_{wandb.run.name}/"
    os.makedirs(model_path, exist_ok=True)
    wandb.log({"model_path": model_path})

    for epoch in range(c.num_epochs):
        model.train()
        running_loss = 0.0
        for i, (down_tumor, down_atlas, coeff_list) in enumerate(train_loader):
            down_tumor = down_tumor.to(device)
            down_atlas = down_atlas.to(device)
            coeff_list = coeff_list.to(device)
            # Concatenate tumor and atlas along the channel dimension
            inputs = torch.cat([down_tumor, down_atlas], dim=1)  # (B, 2, D, H, W)
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, coeff_list)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * down_tumor.size(0)

                    # Compute individual MSE losses per parameter
            loss_origin_3D = torch.mean((preds[:, 0:3] - coeff_list[:, 0:3])**2)
            loss_muD   = torch.mean(torch.abs(preds[:, 3] - coeff_list[:, 3]))
            loss_muRho = torch.mean(torch.abs(preds[:, 4] - coeff_list[:, 4]))

            parameterDiff = torch.abs(preds - coeff_list)

            # Log the batch losses with wandb
            wandb.log({
                "train_loss_batch": loss.item(),
                "train_loss_origin_3D": loss_origin_3D.item(),
                "train_loss_muD": loss_muD.item(),
                "train_loss_muRho": loss_muRho.item(),
                "train_parameters_difference_mean": parameterDiff.abs().mean().item(),
                "train_epoch": epoch + 1,
                "train_batch": i,
            })

        epoch_train_loss = running_loss / train_size
        wandb.log({"epoch_train_loss": epoch_train_loss, "epoch": epoch + 1})
        print(f"[TRAIN] Epoch [{epoch+1}/{c.num_epochs}] - Loss: {epoch_train_loss:.6f}")

        # Save checkpoint for the epoch
        checkpoint_path = os.path.join(model_path, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for down_tumor, down_atlas, coeff_list in val_loader:
                down_tumor = down_tumor.to(device)
                down_atlas = down_atlas.to(device)
                coeff_list = coeff_list.to(device)
                inputs = torch.cat([down_tumor, down_atlas], dim=1)
                preds = model(inputs)
                loss = criterion(preds, coeff_list)
                val_running_loss += loss.item() * down_tumor.size(0)

                loss_origin_3D = torch.mean((preds[:, 0:3] - coeff_list[:, 0:3])**2)
                loss_muD   = torch.mean(torch.abs(preds[:, 3] - coeff_list[:, 3]))
                loss_muRho = torch.mean(torch.abs(preds[:, 4] - coeff_list[:, 4]))

                parameterDiff = torch.abs(preds - coeff_list)

                wandb.log({
                    "val_loss_batch": loss.item(),
                    "val_loss_origin_3D": loss_origin_3D.item(),
                    "val_loss_muD": loss_muD.item(),
                    "val_loss_muRho": loss_muRho.item(),
                    "val_parameters_difference_mean": parameterDiff.abs().mean().item(),
                    "val_epoch": epoch + 1,
                    "val_batch": i,
                })

        epoch_val_loss = val_running_loss / val_size
        wandb.log({"epoch_val_loss": epoch_val_loss, "epoch": epoch + 1})
        print(f"[VAL] Epoch [{epoch+1}/{c.num_epochs}] - Loss: {epoch_val_loss:.6f}")

        # Save best model
        if epoch == 0 or epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_path = os.path.join(model_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at epoch {epoch+1} with val loss {best_val_loss:.6f}")
            wandb.log({"best_val_loss": best_val_loss})

        #save each epoch model
        epoch_model_path = os.path.join(model_path, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_model_path)
        wandb.log({"epoch_model_path": epoch_model_path})

    # ---------------------------------------------------
    # 5) Test loop
    # ---------------------------------------------------
    model.eval()
    test_running_loss = 0.0
    with torch.no_grad():
        for down_tumor, down_atlas, coeff_list in test_loader:
            down_tumor = down_tumor.to(device)
            down_atlas = down_atlas.to(device)
            coeff_list = coeff_list.to(device)
            inputs = torch.cat([down_tumor, down_atlas], dim=1)
            preds = model(inputs)
            loss = criterion(preds, coeff_list)
            test_running_loss += loss.item() * down_tumor.size(0)
    test_loss = test_running_loss / test_size
    wandb.log({"test_loss": test_loss})
    print(f"[TEST] Loss: {test_loss:.6f}")

    wandb.save(model_path)
    wandb.finish()


if __name__ == "__main__":
    train_convnext_encoder_coeff_predictor()