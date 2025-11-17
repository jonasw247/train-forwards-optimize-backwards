#%% 
import sys
sys.path.append('/home/jonas/workspace/programs/neural-surrogate-in-med/')
from utils.datasets import SyntheticDataset, invert_data_preprocess, compute_center_and_displacement, brats_lucas_run_with_sbtc
import pandas as pd
from evaluate.run_forward_sim import runForwardSimulation

import torch
import multiprocessing
import numpy as np
import os
from multiprocessing import Pool
from functools import partial


def run_for_pat(patientNum, group, paramsType = ""):
    print("paramsType", paramsType)
    directPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_rural-wood-5/"


    # Save to CSV
    pathDirGradBasedOptim = "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/gradient-based-optim/"+ group 

    df = pd.read_csv(f"{pathDirGradBasedOptim}/inverse_optimization_results.csv")

    

    #set path to the execution path to the parant directory

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

    device = config["device"]

    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)

    c = Config(config)

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



    if paramsType == "optimizationPrediction":
        predictionParameterList = [
                            df["parameters_x (last)"][patientNum], 
                                df["parameters_y (last)"][patientNum],
                                df["parameters_z (last)"][patientNum],#TODO fix this...
                                df["parameters_muD (last)"][patientNum], 
                            df["parameters_muRho (last)"][patientNum]]

        saveDir = pathDirGradBasedOptim + "/inverse_optimization_results"

    elif paramsType == "optimizationPrediction_brats":
        #brats fixed
        predictionParameterList = [
                            df["parameters_x (last)"][patientNum], 
                                df["parameters_y (last)"][patientNum],
                                df["parameters_z (last)"][patientNum],
                                df["parameters_muD (last)"][patientNum], 
                            df["parameters_muRho (last)"][patientNum]]

        saveDir = pathDirGradBasedOptim + "/inverse_optimization_results_brats"
    
    elif paramsType == "optimizationPrediction_com_origin":

        #origin fixed
        predictionParameterList = [
                                0, 
                                    0,
                                    0,
                                    df["parameters_muD (last)"][patientNum], 
                                df["parameters_muRho (last)"][patientNum]]

        saveDir = pathDirGradBasedOptim + "/inverse_optimization_results_origin_at_com"

    elif paramsType == "directPrediction":

        direcPredParams = np.load(directPath + "test_results.npy", allow_pickle=True).item()

        predictionParameterList = direcPredParams["preds"][patientNum].tolist()

        saveDir = directPath + "direct_prediction_results"

    elif paramsType == "directPrediction_com_origin":
        direcPredParams = np.load(directPath + "test_results.npy", allow_pickle=True).item()

        predictionParameterList = [
                                0, 
                                    0,
                                    0,
                                    direcPredParams["preds"][patientNum][3], 
                                direcPredParams["preds"][patientNum][4]]

        saveDir = directPath + "direct_prediction_results_origin_at_com"

    elif "cma-es" in paramsType:
        #cma-es
        if paramsType == "cma-es-network":
            #cma-es network
            groupCMAES = "cma_20250503_170946"
        
        
            pathDirCMAESNetwork = "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-med/" + groupCMAES

        elif paramsType == "cma-es-classic":

            group = "cma_20250513_234624"
        
            pathDirCMAESNetwork = "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-medCMA-ES-classical_solver/" +group

        df_cma_es = pd.read_csv(f"{pathDirCMAESNetwork}/inverse_optimization_results.csv")
        theRow = None
        for i in range(0, len(df_cma_es["name"])):
            if patientNum == int(df_cma_es["name"][i].split("_")[1]):
                theRow = i
                break
        if theRow is None:
            print("Error: patientNum not found in df_cma_es")
            return
        predictionParameterList = [
                                df_cma_es["parameters_x (last)"][theRow], 
                                    df_cma_es["parameters_y (last)"][theRow],
                                    df_cma_es["parameters_z (last)"][theRow],
                                    df_cma_es["parameters_muD (last)"][theRow], 
                                df_cma_es["parameters_muRho (last)"][theRow]]

        saveDir = pathDirCMAESNetwork  + "/cma-es-network_fwd_results"

    os.makedirs(saveDir, exist_ok=True)

    paramsPred = test_dataset.dataset.get_parameters_from_pedicted_parameters_for_patient(test_dataset.indices[patientNum], predictionParameterList)
    paramsOrg = test_dataset.dataset.get_parameters_from_pedicted_parameters_for_patient(test_dataset.indices[patientNum])
    print(paramsPred)
    
    if paramsType == "optimizationPrediction_brats":
        test_dataset_brats = brats_lucas_run_with_sbtc(length=c.dataset_size, crop_size=c.crop_size, down_sample_size = c.down_sample_size)
        paramsPred = test_dataset_brats.get_parameters_from_pedicted_parameters_for_patient(patientNum, predictionParameterList)

        runForwardSimulation(params=paramsPred, savePath=f"{saveDir}/testset_{patientNum:04d}_dataset_{patientNum}.nii.gz", getAtlas=True)
    else:
        runForwardSimulation(params=paramsPred, savePath=f"{saveDir}/testset_{patientNum:04d}_dataset_{test_dataset.indices[patientNum]}.nii.gz")

    

#%%
def safe_run(i, group, paramsType):
    try:
        print(f"Starting patient {i}")
        run_for_pat(i, group, paramsType)
        print(f"Finished patient {i}")
    except Exception as e:
        print(f"Error processing patient {i}: {e}")

def pool_run( group, paramsType, pool_size, stop):

    with Pool(processes=pool_size) as pool:
        # partial fixes paramsType, so pool.map only varies i
        func = partial(safe_run, group =group, paramsType=paramsType )
        # imap_unordered yields results as they complete
        for _ in pool.imap_unordered(func, range(0, stop)):
            pass

if __name__ == "__main__":
    #run_for_pat(9,"run_20250515_133058", "optimizationPrediction_brats")
    #run_for_pat(0,"run_20250503_191918", "cma-es-classic")
    #for i in range(10):
    #    run_for_pat(i,"run_20250515_133058", "optimizationPrediction_brats")
    pass
#%%
if __name__ == "__main__":
    paramsType ="cma-es-classic"# "cma-es-network" # "directPrediction" #"optimizationPrediction" #"directPrediction" #"optimizationPrediction_com_origin"
    print("lalalla")

    stop =40
    # Your WANDB credentials    
    entity = "tumcompimg"
    project = "inverse-neural-surrogate-in-med"
    #group = "run_20250406_153026"
    group = "run_20250430_185035"
    group = "run_20250501_201936"
    group = "run_20250503_191918"

    #group = "run_20250507_181640"
    #group = "cma_20250513_234624" # classic CMA-ES this is set in the function


    #group = "run_20250515_171328" # real images
    #paramsType = "optimizationPrediction_brats"


    pool_size = 10 # number of worker processes
    pool_run(group, paramsType, pool_size, stop)
    #run_for_pat(2, group, paramsType)



    """        print(f"Error processing patient {i}: {e}")
   

    paramsType = "directPrediction"# "directPrediction" #"optimizationPrediction" #"directPrediction" #"optimizationPrediction_com_origin"
    for i in range(0,500):
        try:
            print("run_for_pat", i)
            run_for_pat(i, paramsType)
        except Exception as e:
     """

# %%
df_cma_es = pd.read_csv(f"/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-medCMA-ES-classical_solver/cma_20250504_183704/inverse_optimization_results.csv")
# %%
row = df_cma_es["name"]#.to_numpy()#str.split("_")
numpyRow = np.array(row.to_numpy())
id = 10
theRow = None
for i in range(0, len(numpyRow)):
    if id == int(numpyRow[i].split("_")[1]):
        theRow = numpyRow[i]

        break
i
df_cma_es["name"][i]
#row_with_id = row[row.int[1] == id]
#row_with_id
# %%
numpyRow.T
# %%
