#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import wandb
import os
import nibabel as nib  
import numpy as np
from scipy import ndimage


from model.fully import FullLearn
from model.convNext import UNetConvNext 
from utils.datasets import  SyntheticDataset #RespondDataset, RhuhDataset
from utils.losses import DiceLoss, GradientLoss, MaskedTVLoss3D
import utils.tools as tools
from monai.losses import SSIMLoss, TverskyLoss
from monai.losses import DiceLoss as MonaiDiceLoss # no need it is the tversky loss with 0.5 and 0.5

import matplotlib.gridspec as gridspec
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import matplotlib.pyplot as plt

def create_wandb_image(array, caption, figsize=(10,10), cmap='gray'):
    """
    Create a matplotlib figure with the given image array, then convert it to a wandb.Image.
    
    Parameters:
        array (np.array): The image data.
        caption (str): Caption for the image.
        figsize (tuple): Figure size in inches (width, height).
        cmap (str): Colormap to use when displaying the image.
    
    Returns:
        wandb.Image: The image wrapped for logging to Weights & Biases.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if cmap == 'bwr':
        array[0,0] = -1
        array[0,1] = 1
        ax.imshow(array, cmap=cmap, vmin=-1, vmax=1)
        #clim
        #plt.clim(-1, 1)
    else:
        ax.imshow(array, cmap=cmap)
    #colorbar 
    ax.figure.colorbar(ax.imshow(array, cmap=cmap)) 
    ax.axis('off')
    ax.set_title(caption)
    # Capture the figure into a wandb.Image
    wandb_img = wandb.Image(fig)
    plt.close(fig)  # Close the figure to free up memory
    return wandb_img

#%%
def train_():
    """
    Runs a single experiment with fixed hyperparameters (no sweeps).
    """
    down_sample_size = 128 #input sizes divisible by 2^stages
    # 1) Define a config dict with your chosen hyperparams:
    config = {
        "dataset_size": 30000,#todo
        "train_size": 28000,
        "test_size": 1000,
        "validation_size":1000,
        "do_dataset_transforms": False,
        "project_name": "neural-surrogate-in-med",
        "entity": None,            # or your W&B username/team
        "lr": 1e-4,                # learning rate
        "batch_size": 6,           # batch size TODO
        "num_epochs": 10000,           # how many epochs to train

        "crop_size": 128, #first cropped and then downsampled
        "down_sample_size": down_sample_size,
        
        "in_channels": 1,           # number of input channels
        "out_channels": 1,
        "device": "cuda:1" if torch.cuda.is_available() else "cpu",
        "brainmask_output" : True,
        "loss":{
            "lambda_MSE": 1,
            "lambda_MSE_in_Brain": 1,
            "lambda_MAE": 0,
        },

        "model":
        {
            "name":"convNext",
            "n_spatial_dims":3,
            "spatial_resolution":(down_sample_size,down_sample_size,down_sample_size) ,
            "stages" : 4,
            "blocks_per_stage": 1,
            "blocks_at_neck":  1,
            "init_features": 32,
            "gradient_checkpointing": False,
            "n_coeffs": 5,
            "clipping_output" : [0,1],
            "final_activation" : "sigmoid",# sigmoid, linear
            "pretrainedWeightsPath": "",#"/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_youthful-brook-57/model_epoch_5.pt"# ""
        } ,
    }

    # 2) Initialize wandb. Pass config dict, so wandb.config is populated.
    wandb.init(
        project=config["project_name"],
        entity=config["entity"],   # or remove if not needed
        config=config,
    )  

    device = config["device"]

    # 3) Access wandb.config to build your model
    c = wandb.config

    # Ensure the model name is valid
    if c.model["name"] == "convNext":
        model = UNetConvNext(
            dim_in=c.in_channels,
            dim_out=c.out_channels,
            n_spatial_dims=c.model["n_spatial_dims"],
            spatial_resolution=c.model["spatial_resolution"],
            stages=c.model["stages"],
            blocks_per_stage=c.model["blocks_per_stage"],
            blocks_at_neck=c.model["blocks_at_neck"],
            init_features=c.model["init_features"],
            gradient_checkpointing=c.model["gradient_checkpointing"],
            n_coeffs=c.model["n_coeffs"],
            clipping_output = c.model["clipping_output"],
            final_activation = c.model["final_activation"]            
        )
        model.to(device)
    else:
        raise ValueError(f"Unsupported model name: {c.model.name}")

    # Print model architecture to wandb
    wandb.watch(model)

    if c.model["pretrainedWeightsPath"] != "":
        pretrained_state = torch.load(c.model["pretrainedWeightsPath"])
        model.load_state_dict(pretrained_state)
        print(f"Pretrained model loaded from {c.model['pretrainedWeightsPath']}")

    # Make sure to set the same seed every time to ensure reproducible splits
    torch.manual_seed(42)

    dataset = SyntheticDataset(length=c.dataset_size, crop_size=c.crop_size, down_sample_size = c.down_sample_size)
    train_size = c.train_size
    val_size = c.validation_size  # Last n for validation
    test_size = c.test_size  # Last n for testing

    # Ensure the dataset is large enough
    assert len(dataset) >= train_size + val_size + test_size, "Dataset size is too small for the specified splits."

    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - test_size, len(dataset)))


    wandb.log({"train_size": train_size, "val_size": val_size, "test_size": test_size})
    wandb.log({"train_files": [idx for idx in train_dataset.indices]})
    wandb.log({"val_files": [idx for idx in val_dataset.indices]})
    wandb.log({"test_files": [idx for idx in test_dataset.indices]})

    train_dataset.dataset.do_dataset_transforms = c.do_dataset_transforms
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=c.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    """external_test_data = RhuhDataset(length=40, path="/mnt/8tb_slot8/jonas/datasets/rhuh/rhuh-gbm_n40", registeredTissuePath="/mnt/8tb_slot8/jonas/workingDirDatasets/rhuh/registeredAtlasToFullRhuh")
    external_test_loader = DataLoader(external_test_data, batch_size=1, shuffle=False)"""

    # 4) Set up optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=c.lr)
    mseLoss_criterion = nn.MSELoss()  

    def getLoss( prediction, target  ,setString):
        mse_loss = c.loss["lambda_MSE"] * mseLoss_criterion(prediction, target) 
        #maskedPrediction =  prediction # TODO this is not done
        #maskedTarget = maskedTarget
        #mse_in_brain = c.loss["lambda_MSE_in_brain"] * mseLoss_criterion(prediction, target)

        loss = mse_loss# + mse_in_brain

        lossDict = {
            setString + "_mse_loss": mse_loss,
            setString + "_total_loss": loss,
        }
        wandb.log(lossDict)
        return loss

    best_val_loss = float('inf')

    # 5) Training loop
    model_path =  f"/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_{wandb.run.name}/"
    os.makedirs(model_path, exist_ok=True)
    wandb.log({"model_path": model_path})
    best_val_loss = float("inf")
    for epoch in range(c.num_epochs):
        model.train()
        running_loss = 0.
        for j, ( downsampled_tumor, downsampled_atlas,  coeff_list) in enumerate(train_loader):


            downsampled_tumor = downsampled_tumor.to(device)
            downsampled_atlas = downsampled_atlas.to(device)


            coeff_list = coeff_list.to(device)

            prediction = model(downsampled_atlas, coeff_list)

            if c.brainmask_output:
                brainMask = downsampled_atlas.clone() * 0
                brainMask[downsampled_atlas > 1.5] = 1
                prediction[brainMask == 0] = 0

            optimizer.zero_grad()  
            loss = getLoss(prediction, downsampled_tumor,  "train")

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * downsampled_tumor.size(0)

            dictImg = {}
            if j % (len(train_loader) // 4) == 0:
                z = int(ndimage.center_of_mass(downsampled_tumor[0, 0].detach().cpu().numpy())[2])
                            
                difference = (prediction - downsampled_tumor)

                dictImg["train_reconstructed" + str(j)] = [
                    create_wandb_image(downsampled_atlas[0, 0].detach().cpu().numpy()[:, :, z], "tissue"),
                    create_wandb_image(downsampled_tumor[0, 0].detach().cpu().numpy()[:, :, z], "ground truth"),

                    create_wandb_image(prediction[0, 0].detach().cpu().numpy()[:, :, z], "prediction"),
                    create_wandb_image(difference[0, 0].detach().cpu().numpy()[:, :, z], "difference" ,  cmap='bwr'),
                ]

                # save nii
                #nib.save(nib.Nifti1Image(reconstructed[0, 0].detach().cpu().numpy(), np.eye(4)), f"/mnt/8tb_slot8/jonas/checkpoints/learnable-brain-tumor-concentration-estimation/checkpoint/model_{wandb.run.name}/reconstructed_t1c{j}.nii.gz")
                #nib.save(nib.Nifti1Image(reconstructed[0, 1].detach().cpu().numpy(), np.eye(4)), f"/mnt/8tb_slot8/jonas/checkpoints/learnable-brain-tumor-concentration-estimation/checkpoint/model_{wandb.run.name}/reconstructed_flair{j}.nii.gz")
            wandb.log(dictImg)

        epoch_train_loss = running_loss / train_size
        wandb.log({"epoch": epoch + 1, "epoch_train_loss": epoch_train_loss})
        print(f"[TRAIN] Epoch [{epoch+1}/{c.num_epochs}] - Loss: {epoch_train_loss:.4f}")

        checkpoint_path = os.path.join(model_path, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)

        # === VALIDATION ===
        model.eval()
        val_running_loss = 0.
        for j, (downsampled_tumor, downsampled_atlas, coeff_list) in enumerate(val_loader):
            downsampled_tumor = downsampled_tumor.to(device)
            downsampled_atlas = downsampled_atlas.to(device)
            coeff_list = coeff_list.to(device)

            with torch.no_grad():
                prediction = model(downsampled_atlas, coeff_list)

                if c.brainmask_output:
                    brainMask = downsampled_atlas.clone() * 0
                    brainMask[downsampled_atlas > 1.5] = 1
                    prediction[brainMask == 0] = 0
                
                loss = getLoss(prediction, downsampled_tumor, "val")
                val_running_loss += loss.item() * downsampled_tumor.size(0)

                dictImg = {}
                # Optionally log images for validation
                if j % (len(val_loader) // 4) == 0:
                    z = int(ndimage.center_of_mass(downsampled_tumor[0, 0].detach().cpu().numpy())[2])
                    difference = (prediction - downsampled_tumor)
                    dictImg[f"val_reconstructed_{j}"] = [
                        create_wandb_image(downsampled_atlas[0, 0].detach().cpu().numpy()[:, :, z], "tissue"),
                        create_wandb_image(downsampled_tumor[0, 0].detach().cpu().numpy()[:, :, z], "ground truth"),
                        create_wandb_image(prediction[0, 0].detach().cpu().numpy()[:, :, z], "prediction"),
                        create_wandb_image(difference[0, 0].detach().cpu().numpy()[:, :, z], "difference", cmap='bwr'),
                    ]
                    wandb.log(dictImg)

        epoch_val_loss = val_running_loss / val_size
        wandb.log({"epoch_val_loss": epoch_val_loss})
        print(f"[VAL] Epoch [{epoch+1}/{c.num_epochs}] - Loss: {epoch_val_loss:.4f}")



        model.eval()
        test_running_loss = 0.
        for j, (downsampled_tumor, downsampled_atlas, coeff_list) in enumerate(test_loader):
            downsampled_tumor = downsampled_tumor.to(device)
            downsampled_atlas = downsampled_atlas.to(device)
            coeff_list = coeff_list.to(device)

            with torch.no_grad():
                prediction = model(downsampled_atlas, coeff_list)
                loss = getLoss(prediction, downsampled_tumor, "test")
                test_running_loss += loss.item() * downsampled_tumor.size(0)

                

        test_loss = test_running_loss / test_size
        wandb.log({"test_loss": test_loss})
        print(f"[TEST] Loss: {test_loss:.4f}")
        
        
        # Save best model based on validation loss
        if epoch == 0 or epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_path = os.path.join(model_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")
            wandb.log({"best_val_test_loss" : test_loss})
       
    wandb.save(model_path)
    wandb.finish()

if __name__ == "__main__":
    train_()

# %%
