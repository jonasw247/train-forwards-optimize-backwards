#!/usr/bin/env python3
"""
Standalone inversion script using CMA-ES
"""
import os
import sys
import time
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import cma
from evaluate.run_forward_sim import runForwardSimulation
from multiprocessing import Pool, cpu_count
from functools import partial
import nibabel as nib
#from pathos.multiprocessing import ProcessingPool as Pool


# make sure your repo root is on PYTHONPATH
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from model.convNext import UNetConvNext
from utils.datasets import SyntheticDataset
from scipy import ndimage

# 1) Configuration
config = {
    # data
    "dataset_size": 30000,
    "test_size": 1000,
    "crop_size": 128,
    "down_sample_size": 128,
    # device
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    # model
    "in_channels": 1,
    "out_channels": 1,
    "model": {
        "name": "convNext",
        "n_spatial_dims": 3,
        "spatial_resolution": (128, 128, 128),
        "stages": 4,
        "blocks_per_stage": 1,
        "blocks_at_neck": 1,
        "init_features": 32,
        "gradient_checkpointing": False,
        "n_coeffs": 5,
        "clipping_output": [0, 1],
        "final_activation": "sigmoid",
        "pretrainedWeightsPath": "/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_nagus-farpoint-61/best_model.pt"# "/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_youthful-brook-57/model_epoch_16.pt" #"/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-in-med/checkpoint/model_nagus-farpoint-61/best_model.pt"# TODO not best best_model.pt"# "" TODO limit ranges!  # fill in if you have a checkpoint
    },
    # inversion (CMA-ES)
    "inversion": {
        "optimizer": "CMA-ES-classical_solver", # "CMA-ES-classical_solver", "CMA-ES"
        "optimization_steps": 1000,
        "sigma0": 0.3,
        "plot_every_n_steps": 2,
        "lb": [-0.2, -0.2, -0.2, 0.1, 0.1],
        "ub": [ 0.2,  0.2,  0.2, 40.0, 15.0],
        "lambda_MSE": 1.0,
        "lambda_MAE": 0.0,
    },
    # wandb
    "project_name": "cma-es-inverse-neural-surrogate-in-med",
    "entity": None,
}

def run_for_patient(idx, group_name="cma_test"):
    # Initialize W&B
    wandb.init(
        project=config["project_name"] + config["inversion"]["optimizer"],
        entity=config["entity"],
        config=config,
        group=group_name,
        name=f"patient_{idx:03d}",
        reinit=True
    )
    paramlabels = ["x", "y", "z", "muD", "muRho"]

    c = wandb.config
    if c.inversion["optimizer"] == "CMA-ES-classical_solver":
        device = "cpu"
    else:
        device = torch.device(config["device"])

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
        clipping_output=c.model["clipping_output"],
        final_activation=c.model["final_activation"],
    ).to(device)

    # Optionally load pretrained weights
    pw = c.model["pretrainedWeightsPath"]
    if pw:
        state = torch.load(pw, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded pretrained weights from {pw}")

    # Prepare data
    dataset = SyntheticDataset(
        length=c.dataset_size,
        crop_size=c.crop_size,
        down_sample_size=c.down_sample_size
    )
    test_ds = torch.utils.data.Subset(dataset,
        range(len(dataset) - c.test_size, len(dataset))
    )
    # pick one
    tumor, atlas, true_coeffs = test_ds[idx]
    atlas = atlas.unsqueeze(0).to(device)
    tumor = tumor.unsqueeze(0).to(device)
    true_coeffs = true_coeffs.unsqueeze(0).to(device)

    # mask to restrict evaluation to tissue region
    mask = (atlas[:, 0] > 1.5).unsqueeze(1).to(device)

    # initial guess: zero for x,y,z; training means for muD, muRho
    params = true_coeffs.clone()
    params[:, :3] = 0
    params[:, 3]  = 12.8
    params[:, 4]  = 3.8

    # unpack inversion config
    inv = c.inversion
    lb = np.array(inv["lb"], dtype=np.float64)
    ub = np.array(inv["ub"], dtype=np.float64)
    bounds = [lb, ub]
    sigma0 = inv["sigma0"]
    steps = inv["optimization_steps"]

    # define loss (pure Python → no torch.autograd inside CMA-ES)
    def eval_loss(x_flat):
        x = torch.tensor(
            x_flat.reshape(params.shape),
          dtype=torch.float32,
           device=device )
        with torch.no_grad():
            pred = model(atlas, x)
            mse = F.mse_loss(pred* mask, tumor)
            mae = F.l1_loss(pred * mask, tumor)
            return (inv["lambda_MSE"] * mse + inv["lambda_MAE"] * mae).item()
        
    
    def run_classical_solver_for_single_list(predictionParameter):
            predictionParameterList = predictionParameter.tolist()
            """predictionParameterList[0] = 0
            predictionParameterList[1] = 0
            predictionParameterList[2] = 0
            predictionParameterList[3] = 0.1
            predictionParameterList[4] = 0.1"""


            #predictionParameterList= true_coeffs.cpu().numpy().flatten().tolist() #TODO remove this # this is for testing

            original_idx = test_ds.indices[idx]
    
            paramsPred = dataset.get_parameters_from_pedicted_parameters_for_patient(original_idx, predictionParameterList)
            print("paramsPred", paramsPred)
            try: 
                predictedTumor = runForwardSimulation(params=paramsPred)
                affine = nib.load("/home/jonas/workspace/programs/neural-surrogate-in-med/data/sub-mni152_tissues_space-sri.nii.gz").affine
                #, savePath="temp/"+str(idx)+".nii.gz")
                targetTumor = dataset.get_original_image(original_idx).numpy()[0]

                #predictedTumor = nib.load
                # predicted tumor into float
                predictedTumor = predictedTumor.astype(np.float32)
                #flip predicted tumor
                #predictedTumor = np.flip(predictedTumor, axis=1)
                
                
                # log images
                if True:

                    z = int(ndimage.center_of_mass(targetTumor)[2])
                    logDict = {}
                    logDict["ground_truth_image"] = wandb.Image(targetTumor[:, :, z], caption="Ground Truth")
                    logDict["prediction_image"] = wandb.Image(predictedTumor[:, :, z], caption="Prediction")
                    #logDict["difference_image"] = wandb.Image(prediction_masked[0,0,:,:,z].detach().cpu().numpy() - downsampled_tumor[0,0,:,:,z].detach().cpu().numpy())
                    print("predicted_tumor_shape", predictedTumor.shape)
                    print("target_tumor_shape", targetTumor.shape)
                    wandb.log(logDict)


                loss = np.mean((predictedTumor - targetTumor) ** 2)
            
            except Exception as e:
                print(f"Error in forward simulation: {e}")
                loss = np.inf
            
            print("loss", loss)
            return loss

    def eval_loss_classical_solver(candidates):
        losses = []
        for i in range(len(candidates)): 
            losses.append(run_classical_solver_for_single_list(candidates[i]))
        return losses

    def eval_loss_classical_solver_parallel(candidates):
        pool_size = min(len(candidates), 8)

        payload = list(enumerate(candidates))

        with Pool(processes=pool_size) as pool:
            func = partial(run_classical_solver_for_single_list)      # fixes the first positional arg (idx)

            losses = pool.map(run_classical_solver_for_single_list, payload)

        return losses

            
    def eval_loss_batch_batch(candidates):
        batch_size = 8
        λ = len(candidates)

        loss_list = []                       # will collect one scalar per candidate
        with torch.no_grad():
            for start in range(0, λ, batch_size):
                end   = min(start + batch_size, λ)
                # 1) stack coefficients for this mini‑batch → (b, n_coeffs)
                coeff_batch = torch.as_tensor(candidates[start:end],
                                            dtype=torch.float32,
                                            device=device)

                b = coeff_batch.shape[0]

                # 2) replicate inputs along batch dim
                atlas_b  = atlas.repeat(b, 1, 1, 1, 1)
                tumor_b  = tumor.repeat(b, 1, 1, 1, 1)
                mask_b   = mask.repeat(b, 1, 1, 1, 1)

                # 3) forward pass
                pred_b = model(atlas_b, coeff_batch)

                # 4) compute losses per sample
                diff   = (pred_b * mask_b) - tumor_b
                mse_b  = diff.pow(2).view(b, -1).mean(dim=1)
                mae_b  = diff.abs().view(b, -1).mean(dim=1)

                loss_b = inv["lambda_MSE"] * mse_b + inv["lambda_MAE"] * mae_b

                loss_list.extend(loss_b.cpu().tolist())   # append python floats
        return loss_list


    # CMA-ES setup
    x0 = params.detach().cpu().numpy().ravel()
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'bounds': bounds})

    start = time.time()
    prev_loss = []

    for gen in range(steps):
        candidates = es.ask()
        if c.inversion["optimizer"] == "CMA-ES-classical_solver":
            losses = eval_loss_classical_solver(candidates)
        elif c.inversion["optimizer"] == "CMA-ES":
            losses = eval_loss_batch_batch(candidates)#[eval_loss(c) for c in candidates]
        es.tell(candidates, losses)

        best_flat = es.result.xbest

        #compare to true coeffs
        diff = best_flat - true_coeffs.cpu().numpy().flatten()

        best_loss  = es.result.fbest
        wandb.log({ 
            "cma/gen": gen,
            "cma/best_loss": es.result.fbest,
            "cma/mean_loss": np.mean(losses),
            "cma/sigma": es.sigma,
            **{f"param_{paramlabels[i]}": best_flat[i] for i in range(len(best_flat))},
            **{f"diff_param_{paramlabels[i]}": diff[i] for i in range(len(diff))},
            "cma/num_canidates": len(candidates),
        })

        #if gen % inv["plot_every_n_steps"] == 0 or gen == steps - 1:
        #    best_flat = es.result.xbest
        #    best_loss  = es.result.fbest
        #    wandb.log({
        #        "cma/gen": gen,
        #        "cma/best_loss": best_loss,
        #        **{f"param_{i}": best_flat[i] for i in range(len(best_flat))}
        #    })
        prev_loss.append(best_loss)
        if len(prev_loss) > 20:
            Δ = abs(prev_loss[-10] - best_loss)
            if Δ < 1e-8:
                print("Loss change is small, stopping optimization")
                break

    duration = time.time() - start

    # inject best back into tensor for any downstream use
    best = es.result.xbest.reshape(params.shape)
    params.data = torch.tensor(
        best,
        dtype=torch.float32,
        device=device
    )

    # final eval
    with torch.no_grad():
        final_pred = model(atlas, params)
        final_mse = F.mse_loss(final_pred * mask, tumor).item()
    wandb.log({
        "cma/final_mse": final_mse,
        "cma/runtime_s": duration,
        "cma/final_best_coeffs": params.cpu().numpy().flatten(),
    })
    wandb.finish()

experiment = f"cma_{datetime.datetime.now():%Y%m%d_%H%M%S}"
"""
if __name__ == "__main__":
    # Example: invert first 100 patients
    for i in [10]:#range(1): #TODO
        try:
            run_for_patient(i)#, group_name=experiment)
        except Exception as e:
            print(f"Error processing patient {i}: {e}")
            
a """

"""
if __name__ == "__main__":
    run_for_patient(0, group_name="cma_test")


"""
"""
def _safe_run(i_experiment):      # top‑level so it can be pickled
    i, experiment = i_experiment  # unpack the tuple
    try:
        run_for_patient(i, group_name=experiment)
    except Exception as e:
        print(f"Error processing patient {i}: {e}")

if __name__ == "__main__":
    experiment = f"cma_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    n_patients   = 300             # set how many you really need
    n_workers    = min(n_patients, 5)  # or any fixed number

    #with Pool(processes=n_workers) as pool:
    # feed (i, experiment) tuples with i = 1 … 5
    #    pool.map()(_safe_run, [(i, experiment) for i in range(1, n_patients + 1)])

    with Pool(processes=n_workers) as pool:
        pool.starmap(
            _safe_run,
            ((i, experiment) for i in range(1, n_patients + 1))
        )
"""
""" I think this was the one I used for the classical??"""
def _safe_run(i_experiment):          # <‑‑ unchanged
    i, experiment = i_experiment
    try:
        run_for_patient(i, group_name=experiment)
    except Exception as e:
        print(f"Error processing patient {i}: {e}")

if __name__ == "__main__":
    experiment  = f"cma_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    n_patients  = 50
    n_workers   = min(n_patients, 25)

    with Pool(processes=n_workers) as pool:
        # feed (i, experiment) tuples; map passes each tuple as ONE argument
        pool.map(_safe_run,
                 [(i, experiment) for i in range(1, n_patients + 1)])
