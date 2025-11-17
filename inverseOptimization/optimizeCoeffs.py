#%%
"""
input: batch of patients 
    tissue, tumor, coeffs

output: optimization series of:
    tumor, loss_tumor, coeffs, loss_coeffs



"""

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import wandb
import os
import nibabel as nib  
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt


from model.fully import FullLearn
from model.convNext import UNetConvNext 
from utils.datasets import  SyntheticDataset, brats_lucas_run_with_sbtc #RespondDataset, RhuhDataset
from utils.losses import DiceLoss, GradientLoss, MaskedTVLoss3D
import utils.tools as tools
from monai.losses import SSIMLoss, TverskyLoss
from monai.losses import DiceLoss as MonaiDiceLoss # no need it is the tversky loss with 0.5 and 0.5

import matplotlib.gridspec as gridspec

import torch.nn.functional as F
import torch.autograd as autograd
import datetime

import time
#%%


down_sample_size = 128 #input sizes divisible by 2^stages
# 1) Define a config dict with your chosen hyperparams:
config = {
    "dataset_size": 25,#30000,#TODO
    "train_size": 0, ##28000,
    "test_size": 25, #1000,
    "datasete_type": "brats_lucas_run_with_sbtc", # "brats", "synthetic", brats_lucas_run_with_sbtc
    "validation_size":1000,
    "do_dataset_transforms": False,
    "project_name": "inverse-neural-surrogate-in-med",
    "entity": None,            # or your W&B username/team
    "lr": 1e-4,                # learning rate
    "batch_size": 6,           # batch size 
    "num_epochs": 10000,           # how many epochs to train

    "crop_size": 128, #first cropped and then downsampled
    "down_sample_size": down_sample_size,
    
    "in_channels": 1,           # number of input channels
    "out_channels": 1,
    "device":"cuda:1" if torch.cuda.is_available() else "cpu",
    "brainmask_output" : True,

    "loss":{
        "lambda_MSE": 1,
    },

    "model":
    {
        "name":"convNext",
        "n_spatial_dims":3,
        "spatial_resolution":(down_sample_size,down_sample_size,down_sample_size),
        "stages" : 4,
        "blocks_per_stage": 1,
        "blocks_at_neck":  1,
        "init_features": 32,
        "gradient_checkpointing": False,
        "n_coeffs": 5,
        "clipping_output" : [0,1],
        "final_activation" : "sigmoid",
        "pretrainedWeightsPath":"/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_rural-blaze-54/best_model.pt"#"/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_rural-blaze-54/best_model.pt"#"/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_nagus-farpoint-61/best_model.pt"# "/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_youthful-brook-57/model_epoch_16.pt" #"/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_nagus-farpoint-61/best_model.pt"# TODO not best best_model.pt"# "" TODO limit ranges!

    } ,
    "inversion":{
        "optimizer":"LBFGS", #  Adam, LBFGS
        "optimization_steps":1000, #TODO adapt for LBFGS
        "learning_rate":0.1, # 0.1, # TODO adapt for LBFGS
        "fix_origin": False, 
        "plot_every_n_steps": 100,
        "lb" : [-0.2, -0.2, -0.2, 0.1, 0.1],
        "ub" : [ 0.2,  0.2,  0.2, 40.0, 15.0],
        "lambda_MSE": 1,
        "lambda_MAE": 0,
        "use_init_from_direct_inversion": True,
    }
}
def runForPatient(thePatientIDX, experimentName = "test"):

    # 2) Initialize wandb. Pass config dict, so wandb.config is populated.
    wandb.init(
        project=config["project_name"],
        entity=config["entity"],   # or remove if not needed
        config=config,
        group = experimentName, #f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        reinit=True,
        name=f"{thePatientIDX:03d}",
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

    if c.model["pretrainedWeightsPath"] != "":
        pretrained_state = torch.load(c.model["pretrainedWeightsPath"])
        model.load_state_dict(pretrained_state)
        print(f"Pretrained model loaded from {c.model['pretrainedWeightsPath']}")

    
    
    if c.datasete_type == "brats_lucas_run_with_sbtc":
        dataset = brats_lucas_run_with_sbtc(length=c.dataset_size, crop_size=c.crop_size, down_sample_size = c.down_sample_size)
    elif c.datasete_type == "synthetic":
        dataset = SyntheticDataset(length=c.dataset_size, crop_size=c.crop_size, down_sample_size = c.down_sample_size)

    test_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - c.test_size, len(dataset)))
    runtimeStart = time.time()

    downsampled_tumor, downsampled_atlas, coeff_list = test_dataset[thePatientIDX]

    original_index = test_dataset.indices[thePatientIDX]
    filename = dataset.getFileName(original_index)
    wandb.log({"filename": filename})

    downsampled_tumor = downsampled_tumor.to(device).unsqueeze(0)
    downsampled_atlas = downsampled_atlas.to(device).unsqueeze(0)
    coeff_list = coeff_list.to(device).unsqueeze(0)


    myOptim = torch.optim.Adam

    model.to(device)
    model.eval() 

    def lossfunction(pred, target): # TODO mask
        pred   = pred.float()
        target = target.float()
        loss = c.inversion["lambda_MSE"] * F.mse_loss(pred, target) + c.inversion["lambda_MAE"] * torch.mean(torch.abs(pred- target))
        return loss

    #original prediction with perfect parameters
    with torch.no_grad():
        gtParametersPrediction = model(downsampled_atlas, coeff_list)
        masked_gtParametersPrediction = gtParametersPrediction.to(device) * downsampled_atlas[:, 0].unsqueeze(1)  > 1.5
        gt_parameters_loss = lossfunction(masked_gtParametersPrediction, downsampled_tumor.to(device))
    
    parameters = coeff_list.clone()

    if c.inversion["use_init_from_direct_inversion"]:

        direct_allParams_File =  np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_rural-wood-5/test_results.npy", allow_pickle=True).item() # this is a bit better
        direct_allParams = np.array(direct_allParams_File["preds"]).T[:, thePatientIDX]
        parameters[:,0] = float(direct_allParams[0])
        parameters[:,1] = float(direct_allParams[1])
        parameters[:,2] = float(direct_allParams[2])
        parameters[:,3] = float(direct_allParams[3])
        parameters[:,4] = float(direct_allParams[4])
    else:
        with torch.no_grad():
            """
            parameters[:,0] = -0.0236732 
            parameters[:,1] = -0.05205802
            parameters[:,2] =  0.0365184
            parameters[:,3] = 16.022573 # derived from training mean
            parameters[:,4] = 4.5779634# derived from training mean
            """
            parameters[:,0] = 0
            parameters[:,1] = 0
            parameters[:,2] = 0
            parameters[:,3] = 12.8 # derived from training mean
            parameters[:,4] = 3.8# derived from training mean

    downsampled_atlas = downsampled_atlas.to(device)
    parameters = parameters.to(device)
    parameters.requires_grad = True

    if c.inversion["optimizer"] == "Adam":
        optimizer = myOptim([parameters], lr=c.inversion["learning_rate"])

    elif c.inversion["optimizer"] == "LBFGS":
        optimizer = torch.optim.LBFGS(
            [parameters],
            lr=c.inversion["learning_rate"],              # initial step size
            max_iter=20,         # inner iterations per call
            #history_size=10,     # memory for approximating Hessian
            line_search_fn='strong_wolfe'
            )

    # define your lower‐ and upper‐bounds (shape must broadcast to parameters.shape)


    mask = downsampled_atlas[:, 0].unsqueeze(1)  > 1.5
    #to device
    mask = mask.to(device)

    lb = torch.tensor(c.inversion["lb"], device=device).unsqueeze(0)
    ub = torch.tensor(c.inversion["ub"], device=device).unsqueeze(0)
    def closure():
        optimizer.zero_grad()

        parameters.data.clamp_(min=lb, max=ub)

        pred = model(downsampled_atlas, parameters)
        loss = lossfunction(pred * mask, downsampled_tumor)
        loss.backward()
        return loss
    
    prev_loss = None
    for optimStep in range(c.inversion["optimization_steps"]):
        if optimStep % 10 == 0:
            print("step: ", optimStep)
        optimizer.zero_grad()

        if c.inversion["optimizer"] == "LBFGS":
            # LBFGS calls closure internally (possibly multiple times)
            loss = optimizer.step(closure)
        else:
            # Adam: manual forward/backward/step
            optimizer.zero_grad()
            #parameters.data.clamp_(min=lb, max=ub)

            pred = model(downsampled_atlas, parameters)
            loss = F.mse_loss(pred * mask, downsampled_tumor)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            parameters.clamp_(min=lb, max=ub)


        if c.inversion["fix_origin"]:
            with torch.no_grad():
                parameters.grad[:,0] = 0
                parameters.grad[:,1] = 0
                parameters.grad[:,2] = 0
        #optimizer.step()


        logDict = {}
        # plot the difference image with wandb
        if True: #optimStep % c.inversion["plot_every_n_steps"] == 0  or c.inversion["optimizer"] == "LBFGS":
            pred = model(downsampled_atlas, parameters)
            prediction_masked = pred * mask

            z = int(ndimage.center_of_mass(downsampled_tumor[0, 0].detach().cpu().numpy())[2])
            if optimStep == 0:
                logDict["ground_truth_image"] = wandb.Image(downsampled_tumor[0,0,:,:,z].detach().cpu().numpy())
            logDict["prediction_image"] = wandb.Image(prediction_masked[0,0,:,:,z].detach().cpu().numpy())
            logDict["difference_image"] = wandb.Image(prediction_masked[0,0,:,:,z].detach().cpu().numpy() - downsampled_tumor[0,0,:,:,z].detach().cpu().numpy())

        if True:# optimStep % 10 == 0 or optimStep % c.inversion["plot_every_n_steps"] == 0:
            if c.inversion["optimizer"] == "LBFGS":
                st = optimizer.state[parameters]
                n_iter     = st.get('n_iter',     0)   # inner LBFGS iters this step
                func_evals = st.get('func_evals',  0)   # total closure calls so far
                logDict["n_iterations"] = n_iter 
                logDict["n_eval"] = func_evals
                #logDict["n_iterations_total"] = optimizer.get('n_iter', 0)
            # gt_parameters_loss
            logDict["gt_parameters_loss"] = gt_parameters_loss.item()
            logDict["_loss"] = loss.item()
            logDict["_grad_parameters_mean"] = parameters.grad.mean().item()
            #log each grad_parameter
            parameterDiff = parameters - coeff_list.to(device)
            labels = ["x", "y", "z", "muD", "muRho"]
            for j in range(5):
                logDict["grad_parameters_" + labels[j]] = parameters.grad[0,j].item()
                logDict["parameters_" + labels[j]] = parameters[0,j].item()
                logDict["parameters_difference_" + labels[j]] = parameterDiff[0,j].item()

            origin_difference_3D = (parameters[0,0:3] - coeff_list[0,0:3]).abs().mean().item()
            logDict["origin_difference_3D"] = origin_difference_3D
            #log mean parameter difference
            logDict["_parameters_difference_mean"] = parameterDiff.abs().mean().item()
                
            stopTime = time.time()
            logDict["_runtime"] = stopTime - runtimeStart

        
            wandb.log(logDict)

            # stop if it only did e.g. 1–2 inner steps and loss
        # changed by less than 1e-5 compared to last outer step
        if prev_loss is not None:
            Δ = abs(prev_loss - loss.item())
            if Δ < 1e-8:
                print("Loss change is small, stopping optimization")
                break

        prev_loss = loss.item()

experiment_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

for i in range(0,1000):#TODO

    runForPatient(i, experiment_name)

# %%
