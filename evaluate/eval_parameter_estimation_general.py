#%%
import wandb
import pandas as pd
import numpy as np
from tabulate import tabulate

# Your WANDB credentials
entity = "tumcompimg"
project = "inverse-neural-surrogate-in-med"
group = "run_20250507_181640"


#%%
# Save to CSV
pathDir = "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/gradient-based-optim/"+ group 

inverseOptimizationDF = pd.read_csv(f"{pathDir}/inverse_optimization_results.csv")
df = inverseOptimizationDF.copy()
#direct_allParams_File =  np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_xindi-maquis-4/test_results.npy", allow_pickle=True).item() # this is the first one which is not so good...
direct_allParams_File =  np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_rural-wood-5/test_results.npy", allow_pickle=True).item() # this is a bit better

inverseOptimization_CMA_ES_DF = pd.read_csv(f"/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-med/cma_20250503_170946/inverse_optimization_results.csv")

inverseOptimization_CMA_ES_DF_classic =  pd.read_csv(f"/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-medCMA-ES-classical_solver/cma_20250513_234624/inverse_optimization_results.csv")

inverseOptimization_CMA_ES_DF_classic = inverseOptimization_CMA_ES_DF_classic.sort_values(by="name")
inverseOptimization_CMA_ES_DF = inverseOptimization_CMA_ES_DF.sort_values(by="name")

# %%
#plot the coeff_loss
NumPat = 500

import matplotlib.pyplot as plt

labels = ["x", "y", "z", "muD", "muRho"]

optim_allParams = [
    inverseOptimizationDF["parameters_x (last)"],
    inverseOptimizationDF["parameters_y (last)"],
    inverseOptimizationDF["parameters_z (last)"],
    inverseOptimizationDF["parameters_muD (last)"],
    inverseOptimizationDF["parameters_muRho (last)"],
    inverseOptimizationDF["_runtime (last)"]
]
optim_allParams_CMA_ES_network = [
    inverseOptimization_CMA_ES_DF["parameters_x (last)"],
    inverseOptimization_CMA_ES_DF["parameters_y (last)"],
    inverseOptimization_CMA_ES_DF["parameters_z (last)"],
    inverseOptimization_CMA_ES_DF["parameters_muD (last)"],
    inverseOptimization_CMA_ES_DF["parameters_muRho (last)"],
    inverseOptimization_CMA_ES_DF["runtime"]
]

optim_allParams_CMA_ES_classic = [
    inverseOptimization_CMA_ES_DF_classic["parameters_x (last)"],
    inverseOptimization_CMA_ES_DF_classic["parameters_y (last)"],
    inverseOptimization_CMA_ES_DF_classic["parameters_z (last)"],
    inverseOptimization_CMA_ES_DF_classic["parameters_muD (last)"],
    inverseOptimization_CMA_ES_DF_classic["parameters_muRho (last)"],
    inverseOptimization_CMA_ES_DF_classic["runtime"]
]

optim_allParams = np.array(optim_allParams)[:, :NumPat]
optim_allParams_CMA_ES_network = np.array(optim_allParams_CMA_ES_network)[:, :NumPat]
optim_allParams_CMA_ES_classic = np.array(optim_allParams_CMA_ES_classic)[:, :25]
direct_allParams = np.array(direct_allParams_File["preds"]).T[:, :NumPat]
gt_allParams = np.array(direct_allParams_File["trues"]).T[:, :NumPat]

print("mean runtime optim: ", np.mean(optim_allParams[5]),"+-", np.std(optim_allParams[5])/np.sqrt(len(optim_allParams[5])))

print("mean runtime CMA-ES: ", np.mean(optim_allParams_CMA_ES_network[5]),"+-", np.std(optim_allParams_CMA_ES_network[5])/np.sqrt(len(optim_allParams_CMA_ES_network[5])))

print("mean runtime CMA-ES classic: ", np.mean(optim_allParams_CMA_ES_classic[5]),"+-", np.std(optim_allParams_CMA_ES_classic[5])/np.sqrt(len(optim_allParams_CMA_ES_classic[5])))

#%%
def calcMSE(preds, gt):
    meanMSE = np.mean((preds - gt) ** 2, axis=0)
    stdErrMSE = np.std((preds - gt) ** 2, axis=0) / np.sqrt(len(preds))
    return meanMSE, stdErrMSE

def calcMAE(preds, gt):
    meanMAE = np.mean(np.abs(preds - gt), axis=0)
    stdErrMAE = np.std(np.abs(preds - gt), axis=0) / np.sqrt(len(preds))
    return meanMAE, stdErrMAE

def calcMAE_Relative(preds, gt):
    meanMAE = np.mean(np.abs(preds - gt) / np.abs(gt), axis=0)
    stdErrMAE = np.std(np.abs(preds - gt) / np.abs(gt), axis=0) / np.sqrt(len(preds))
    return meanMAE, stdErrMAE

def calcMSE_Relative(preds, gt):
    meanMSE = np.mean((preds - gt) ** 2 / np.abs(gt) ** 2, axis=0)
    stdErrMSE = np.std((preds - gt) ** 2 / np.abs(gt) ** 2, axis=0) / np.sqrt(len(preds))
    return meanMSE, stdErrMSE

def calcOrigin_mse(preds, gt):
    meanMSE = np.mean(np.linalg.norm(preds - gt, axis=0) ** 2)
    stdErrMSE = np.std(np.linalg.norm(preds - gt, axis=0) ** 2) / np.sqrt(len(preds))
    return meanMSE, stdErrMSE

def calcOrigin_mae(preds, gt):
    meanMAE = np.mean(np.linalg.norm(preds - gt, axis=0))
    stdErrMAE = np.std(np.linalg.norm(preds - gt, axis=0)) / np.sqrt(len(preds))
    return meanMAE, stdErrMAE

def calcOrigin_mae_relative(preds, gt):
    meanMAE = np.mean(np.linalg.norm(preds - gt, axis=0) / np.linalg.norm(gt, axis=0))
    stdErrMAE = np.std(np.linalg.norm(preds - gt, axis=0) / np.linalg.norm(gt, axis=0)) / np.sqrt(len(preds))
    return meanMAE, stdErrMAE

def calcOrigin_mse_relative(preds, gt):
    meanMSE = np.mean(np.linalg.norm(preds - gt, axis=0) ** 2 / np.linalg.norm(gt, axis=0) ** 2)
    stdErrMSE = np.std(np.linalg.norm(preds - gt, axis=0) ** 2 / np.linalg.norm(gt, axis=0) ** 2) / np.sqrt(len(preds))
    return meanMSE, stdErrMSE


#%% print results table #papertable
results = []
for i in range(3, 5):
    # Direct Prediction
    meanMSE_direct, stdErrMSE_direct = calcMSE(direct_allParams[i], gt_allParams[i])
    meanMAE_direct, stdErrMAE_direct = calcMAE(direct_allParams[i], gt_allParams[i])
    meanMSE_rel_direct, stdErrMSE_rel_direct = calcMSE_Relative(direct_allParams[i], gt_allParams[i])
    meanMAE_rel_direct, stdErrMAE_rel_direct = calcMAE_Relative(direct_allParams[i], gt_allParams[i])
    results.append([f"Direct {labels[i]}", meanMSE_direct, stdErrMSE_direct, meanMAE_direct, stdErrMAE_direct, meanMSE_rel_direct, stdErrMSE_rel_direct, meanMAE_rel_direct, stdErrMAE_rel_direct])
    # Optimization
    meanMSED_optim, stdErrMSE_optim = calcMSE(optim_allParams[i], gt_allParams[i])
    meanMAE_optim, stdErrMAE_optim = calcMAE(optim_allParams[i], gt_allParams[i])
    meanMSE_rel_optim, stdErrMSE_rel_optim = calcMSE_Relative(optim_allParams[i], gt_allParams[i])
    meanMAE_rel_optim, stdErrMAE_rel_optim = calcMAE_Relative(optim_allParams[i], gt_allParams[i])
    results.append([f"Optim {labels[i]}", meanMSED_optim, stdErrMSE_optim, meanMAE_optim, stdErrMAE_optim, meanMSE_rel_optim, stdErrMSE_rel_optim, meanMAE_rel_optim, stdErrMAE_rel_optim])
    # CMA-ES Network
    meanMSE_CMAES_network, stdErrMSE_CMAES_network = calcMSE(optim_allParams_CMA_ES_network[i], gt_allParams[i])
    meanMAE_CMAES_network, stdErrMAE_CMAES_network = calcMAE(optim_allParams_CMA_ES_network[i], gt_allParams[i])
    meanMSE_rel_CMAES_network, stdErrMSE_rel_CMAES_network = calcMSE_Relative(optim_allParams_CMA_ES_network[i], gt_allParams[i])
    meanMAE_rel_CMAES_network, stdErrMAE_rel_CMAES_network = calcMAE_Relative(optim_allParams_CMA_ES_network[i], gt_allParams[i])
    results.append([f"CMA-ES Network {labels[i]}", meanMSE_CMAES_network, stdErrMSE_CMAES_network, meanMAE_CMAES_network, stdErrMAE_CMAES_network, meanMSE_rel_CMAES_network, stdErrMSE_rel_CMAES_network, meanMAE_rel_CMAES_network, stdErrMAE_rel_CMAES_network])
    # CMA-ES classic
    gt_allParams_classic = gt_allParams[:,1:26]
    meanMSE_CMAES_classic, stdErrMSE_CMAES_classic = calcMSE(optim_allParams_CMA_ES_classic[i], gt_allParams_classic[i])
    meanMAE_CMAES_classic, stdErrMAE_CMAES_classic = calcMAE(optim_allParams_CMA_ES_classic[i], gt_allParams_classic[i])
    meanMSE_rel_CMAES_classic, stdErrMSE_rel_CMAES_classic = calcMSE_Relative(optim_allParams_CMA_ES_classic[i], gt_allParams_classic[i])
    meanMAE_rel_CMAES_classic, stdErrMAE_rel_CMAES_classic = calcMAE_Relative(optim_allParams_CMA_ES_classic[i], gt_allParams_classic[i])
    results.append([f"CMA-ES classic {labels[i]}", meanMSE_CMAES_classic, stdErrMSE_CMAES_classic, meanMAE_CMAES_classic, stdErrMAE_CMAES_classic, meanMSE_rel_CMAES_classic, stdErrMSE_rel_CMAES_classic, meanMAE_rel_CMAES_classic, stdErrMAE_rel_CMAES_classic])



# Calculate origin difference for i = 0-3
meanMSE_origin_direct, stdErrMSE_origin_direct = calcOrigin_mse(direct_allParams[:3], gt_allParams[:3])
meanMAE_origin_direct, stdErrMAE_origin_direct = calcOrigin_mae(direct_allParams[:3], gt_allParams[:3])
meanMSE_rel_origin_direct, stdErrMSE_rel_origin_direct = calcOrigin_mse_relative(direct_allParams[:3], gt_allParams[:3])
meanMAE_rel_origin_direct, stdErrMAE_rel_origin_direct = calcOrigin_mae_relative(direct_allParams[:3], gt_allParams[:3])

meanMSE_origin_optim, stdErrMSE_origin_optim = calcOrigin_mse(optim_allParams[:3], gt_allParams[:3])
meanMAE_origin_optim, stdErrMAE_origin_optim = calcOrigin_mae(optim_allParams[:3], gt_allParams[:3])
meanMSE_rel_origin_optim, stdErrMSE_rel_origin_optim = calcOrigin_mse_relative(optim_allParams[:3], gt_allParams[:3])
meanMAE_rel_origin_optim, stdErrMAE_rel_origin_optim = calcOrigin_mae_relative(optim_allParams[:3], gt_allParams[:3])

meanMSE_origin_CMAES_network, stdErrMSE_origin_CMAES_network = calcOrigin_mse(optim_allParams_CMA_ES_network[:3], gt_allParams[:3])
meanMAE_origin_CMAES_network, stdErrMAE_origin_CMAES_network = calcOrigin_mae(optim_allParams_CMA_ES_network[:3], gt_allParams[:3])
meanMSE_rel_origin_CMAES_network, stdErrMSE_rel_origin_CMAES_network = calcOrigin_mse_relative(optim_allParams_CMA_ES_network[:3], gt_allParams[:3])
meanMAE_rel_origin_CMAES_network, stdErrMAE_rel_origin_CMAES_network = calcOrigin_mae_relative(optim_allParams_CMA_ES_network[:3], gt_allParams[:3])

meanMSE_origin_CMAES_classic, stdErrMSE_origin_CMAES_classic = calcOrigin_mse(optim_allParams_CMA_ES_classic[:3], gt_allParams_classic[:3])
meanMAE_origin_CMAES_classic, stdErrMAE_origin_CMAES_classic = calcOrigin_mae(optim_allParams_CMA_ES_classic[:3], gt_allParams_classic[:3])
meanMSE_rel_origin_CMAES_classic, stdErrMSE_rel_origin_CMAES_classic = calcOrigin_mse_relative(optim_allParams_CMA_ES_classic[:3], gt_allParams_classic[:3])
meanMAE_rel_origin_CMAES_classic, stdErrMAE_rel_origin_CMAES_classic = calcOrigin_mae_relative(optim_allParams_CMA_ES_classic[:3], gt_allParams_classic[:3])


results.append(["Direct Origin", meanMSE_origin_direct, stdErrMSE_origin_direct, meanMAE_origin_direct, stdErrMAE_origin_direct, meanMSE_rel_origin_direct, stdErrMSE_rel_origin_direct, meanMAE_rel_origin_direct, stdErrMAE_rel_origin_direct])
results.append(["Optim Origin", meanMSE_origin_optim, stdErrMSE_origin_optim, meanMAE_origin_optim, stdErrMAE_origin_optim, meanMSE_rel_origin_optim, stdErrMSE_rel_origin_optim, meanMAE_rel_origin_optim, stdErrMAE_rel_origin_optim])
results.append(["CMA-ES Network Origin", meanMSE_origin_CMAES_network, stdErrMSE_origin_CMAES_network, meanMAE_origin_CMAES_network, stdErrMAE_origin_CMAES_network, meanMSE_rel_origin_CMAES_network, stdErrMSE_rel_origin_CMAES_network, meanMAE_rel_origin_CMAES_network, stdErrMAE_rel_origin_CMAES_network])
results.append(["CMA-ES classic Origin", meanMSE_origin_CMAES_classic, stdErrMSE_origin_CMAES_classic, meanMAE_origin_CMAES_classic, stdErrMAE_origin_CMAES_classic, meanMSE_rel_origin_CMAES_classic, stdErrMSE_rel_origin_CMAES_classic, meanMAE_rel_origin_CMAES_classic, stdErrMAE_rel_origin_CMAES_classic])


# Print results as a table
headers = ["Method", "Mean MSE", "StdErr MSE", "Mean MAE", "StdErr MAE", "Mean MSE Rel", "StdErr MSE Rel", "Mean MAE Rel", "StdErr MAE Rel"]
print(tabulate(results, headers=headers, tablefmt="grid"))

#%%
plt.scatter(optim_allParams_CMA_ES_classic[3], gt_allParams_classic[3])
plt.scatter(optim_allParams_CMA_ES_classic[4], gt_allParams_classic[4])
plt.xlabel("CMA-ES classic")
plt.ylabel("GT")
plt.figure()
plt.plot(optim_allParams_CMA_ES_classic[3] - gt_allParams_classic[3], marker='o', label="muD CMA-ES classic")

#rho
plt.plot(optim_allParams_CMA_ES_classic[4], marker='o', label="muRho CMA-ES classic")
plt.plot(gt_allParams_classic[4], marker='o', label="muRho GT")
#D
plt.figure()
plt.plot(optim_allParams_CMA_ES_classic[3], marker='o', label="muD GT")
plt.plot(gt_allParams_classic[3], marker='o', label="muD GT")


plt.title("Difference between direct prediction and optimization for muD")
plt.legend()
#%% table #papertable

def calcMSE(preds, gt):
    medianMSE = np.median((preds - gt) ** 2, axis=0)
    stdErrMSE = np.std((preds - gt) ** 2, axis=0) / np.sqrt(len(preds))
    return medianMSE, stdErrMSE

def calcMAE(preds, gt):
    medianMAE = np.median(np.abs(preds - gt), axis=0)
    stdErrMAE = np.std(np.abs(preds - gt), axis=0) / np.sqrt(len(preds))
    return medianMAE, stdErrMAE

def calcMAE_Relative(preds, gt):
    medianMAE = np.median(np.abs(preds - gt) / np.abs(gt), axis=0)
    stdErrMAE = np.std(np.abs(preds - gt) / np.abs(gt), axis=0) / np.sqrt(len(preds))
    return medianMAE, stdErrMAE

def calcMSE_Relative(preds, gt):
    medianMSE = np.median((preds - gt) ** 2 / np.abs(gt) ** 2, axis=0)
    stdErrMSE = np.std((preds - gt) ** 2 / np.abs(gt) ** 2, axis=0) / np.sqrt(len(preds))
    return medianMSE, stdErrMSE

def calcOrigin_mse(preds, gt):
    medianMSE = np.median(np.linalg.norm(preds - gt, axis=0) ** 2)
    stdErrMSE = np.std(np.linalg.norm(preds - gt, axis=0) ** 2) / np.sqrt(len(preds))
    return medianMSE, stdErrMSE

def calcOrigin_mae(preds, gt):
    medianMAE = np.median(np.linalg.norm(preds - gt, axis=0))
    stdErrMAE = np.std(np.linalg.norm(preds - gt, axis=0)) / np.sqrt(len(preds))
    return medianMAE, stdErrMAE

def calcOrigin_mae_relative(preds, gt):
    medianMAE = np.median(np.linalg.norm(preds - gt, axis=0) / np.linalg.norm(gt, axis=0))
    stdErrMAE = np.std(np.linalg.norm(preds - gt, axis=0) / np.linalg.norm(gt, axis=0)) / np.sqrt(len(preds))
    return medianMAE, stdErrMAE

def calcOrigin_mse_relative(preds, gt):
    medianMSE = np.median(np.linalg.norm(preds - gt, axis=0) ** 2 / np.linalg.norm(gt, axis=0) ** 2)
    stdErrMSE = np.std(np.linalg.norm(preds - gt, axis=0) ** 2 / np.linalg.norm(gt, axis=0) ** 2) / np.sqrt(len(preds))
    return medianMSE, stdErrMSE



results = []
for i in range(3, 5):
    # Direct Prediction
    medianMSE_direct, stdErrMSE_direct = calcMSE(direct_allParams[i], gt_allParams[i])
    medianMAE_direct, stdErrMAE_direct = calcMAE(direct_allParams[i], gt_allParams[i])
    medianMSE_rel_direct, stdErrMSE_rel_direct = calcMSE_Relative(direct_allParams[i], gt_allParams[i])
    medianMAE_rel_direct, stdErrMAE_rel_direct = calcMAE_Relative(direct_allParams[i], gt_allParams[i])
    results.append([f"Direct {labels[i]}", medianMSE_direct, stdErrMSE_direct, medianMAE_direct, stdErrMAE_direct, medianMSE_rel_direct, stdErrMSE_rel_direct, medianMAE_rel_direct, stdErrMAE_rel_direct])
    # Optimization
    medianMSED_optim, stdErrMSE_optim = calcMSE(optim_allParams[i], gt_allParams[i])
    medianMAE_optim, stdErrMAE_optim = calcMAE(optim_allParams[i], gt_allParams[i])
    medianMSE_rel_optim, stdErrMSE_rel_optim = calcMSE_Relative(optim_allParams[i], gt_allParams[i])
    medianMAE_rel_optim, stdErrMAE_rel_optim = calcMAE_Relative(optim_allParams[i], gt_allParams[i])
    results.append([f"Optim {labels[i]}", medianMSED_optim, stdErrMSE_optim, medianMAE_optim, stdErrMAE_optim, medianMSE_rel_optim, stdErrMSE_rel_optim, medianMAE_rel_optim, stdErrMAE_rel_optim])
    # CMA-ES Network
    medianMSE_CMAES_network, stdErrMSE_CMAES_network = calcMSE(optim_allParams_CMA_ES_network[i], gt_allParams[i])
    medianMAE_CMAES_network, stdErrMAE_CMAES_network = calcMAE(optim_allParams_CMA_ES_network[i], gt_allParams[i])
    medianMSE_rel_CMAES_network, stdErrMSE_rel_CMAES_network = calcMSE_Relative(optim_allParams_CMA_ES_network[i], gt_allParams[i])
    medianMAE_rel_CMAES_network, stdErrMAE_rel_CMAES_network = calcMAE_Relative(optim_allParams_CMA_ES_network[i], gt_allParams[i])
    results.append([f"CMA-ES Network {labels[i]}", medianMSE_CMAES_network, stdErrMSE_CMAES_network, medianMAE_CMAES_network, stdErrMAE_CMAES_network, medianMSE_rel_CMAES_network, stdErrMSE_rel_CMAES_network, medianMAE_rel_CMAES_network, stdErrMAE_rel_CMAES_network])
    # CMA-ES classic
    medianMSE_CMAES_classic, stdErrMSE_CMAES_classic = calcMSE(optim_allParams_CMA_ES_classic[i], gt_allParams_classic[i])
    medianMAE_CMAES_classic, stdErrMAE_CMAES_classic = calcMAE(optim_allParams_CMA_ES_classic[i], gt_allParams_classic[i])
    medianMSE_rel_CMAES_classic, stdErrMSE_rel_CMAES_classic = calcMSE_Relative(optim_allParams_CMA_ES_classic[i], gt_allParams_classic[i])
    medianMAE_rel_CMAES_classic, stdErrMAE_rel_CMAES_classic = calcMAE_Relative(optim_allParams_CMA_ES_classic[i], gt_allParams_classic[i])
    results.append([f"CMA-ES classic {labels[i]}", medianMSE_CMAES_classic, stdErrMSE_CMAES_classic, medianMAE_CMAES_classic, stdErrMAE_CMAES_classic, medianMSE_rel_CMAES_classic, stdErrMSE_rel_CMAES_classic, medianMAE_rel_CMAES_classic, stdErrMAE_rel_CMAES_classic])




# Calculate origin difference for i = 0-3
medianMSE_origin_direct, stdErrMSE_origin_direct = calcOrigin_mse(direct_allParams[:3], gt_allParams[:3])
medianMAE_origin_direct, stdErrMAE_origin_direct = calcOrigin_mae(direct_allParams[:3], gt_allParams[:3])
medianMSE_rel_origin_direct, stdErrMSE_rel_origin_direct = calcOrigin_mse_relative(direct_allParams[:3], gt_allParams[:3])
medianMAE_rel_origin_direct, stdErrMAE_rel_origin_direct = calcOrigin_mae_relative(direct_allParams[:3], gt_allParams[:3])

medianMSE_origin_optim, stdErrMSE_origin_optim = calcOrigin_mse(optim_allParams[:3], gt_allParams[:3])
medianMAE_origin_optim, stdErrMAE_origin_optim = calcOrigin_mae(optim_allParams[:3], gt_allParams[:3])
medianMSE_rel_origin_optim, stdErrMSE_rel_origin_optim = calcOrigin_mse_relative(optim_allParams[:3], gt_allParams[:3])
medianMAE_rel_origin_optim, stdErrMAE_rel_origin_optim = calcOrigin_mae_relative(optim_allParams[:3], gt_allParams[:3])

medianMSE_origin_CMAES_network, stdErrMSE_origin_CMAES_network = calcOrigin_mse(optim_allParams_CMA_ES_network[:3], gt_allParams[:3])
medianMAE_origin_CMAES_network, stdErrMAE_origin_CMAES_network = calcOrigin_mae(optim_allParams_CMA_ES_network[:3], gt_allParams[:3])
medianMSE_rel_origin_CMAES_network, stdErrMSE_rel_origin_CMAES_network = calcOrigin_mse_relative(optim_allParams_CMA_ES_network[:3], gt_allParams[:3])
medianMAE_rel_origin_CMAES_network, stdErrMAE_rel_origin_CMAES_network = calcOrigin_mae_relative(optim_allParams_CMA_ES_network[:3], gt_allParams[:3])

medianMSE_origin_CMAES_classic, stdErrMSE_origin_CMAES_classic = calcOrigin_mse(optim_allParams_CMA_ES_classic[:3], gt_allParams_classic[:3])
medianMAE_origin_CMAES_classic, stdErrMAE_origin_CMAES_classic = calcOrigin_mae(optim_allParams_CMA_ES_classic[:3], gt_allParams_classic[:3])
medianMSE_rel_origin_CMAES_classic, stdErrMSE_rel_origin_CMAES_classic = calcOrigin_mse_relative(optim_allParams_CMA_ES_classic[:3], gt_allParams_classic[:3])
medianMAE_rel_origin_CMAES_classic, stdErrMAE_rel_origin_CMAES_classic = calcOrigin_mae_relative(optim_allParams_CMA_ES_classic[:3], gt_allParams_classic[:3])


results.append(["Direct Origin", medianMSE_origin_direct, stdErrMSE_origin_direct, medianMAE_origin_direct, stdErrMAE_origin_direct, medianMSE_rel_origin_direct, stdErrMSE_rel_origin_direct, medianMAE_rel_origin_direct, stdErrMAE_rel_origin_direct])
results.append(["Optim Origin", medianMSE_origin_optim, stdErrMSE_origin_optim, medianMAE_origin_optim, stdErrMAE_origin_optim, medianMSE_rel_origin_optim, stdErrMSE_rel_origin_optim, medianMAE_rel_origin_optim, stdErrMAE_rel_origin_optim])
results.append(["CMA-ES Network Origin", medianMSE_origin_CMAES_network, stdErrMSE_origin_CMAES_network, medianMAE_origin_CMAES_network, stdErrMAE_origin_CMAES_network, medianMSE_rel_origin_CMAES_network, stdErrMSE_rel_origin_CMAES_network, medianMAE_rel_origin_CMAES_network, stdErrMAE_rel_origin_CMAES_network])
results.append(["CMA-ES classic Origin", medianMSE_origin_CMAES_classic, stdErrMSE_origin_CMAES_classic, medianMAE_origin_CMAES_classic, stdErrMAE_origin_CMAES_classic, medianMSE_rel_origin_CMAES_classic, stdErrMSE_rel_origin_CMAES_classic, medianMAE_rel_origin_CMAES_classic, stdErrMAE_rel_origin_CMAES_classic])


# Print results as a table
headers = ["Method", "Median MSE", "StdErr MSE", "Median MAE", "StdErr MAE", "Median MSE Rel", "StdErr MSE Rel", "Median MAE Rel", "StdErr MAE Rel"]
print(tabulate(results, headers=headers, tablefmt="grid"))

#%% plot the difference between the direct prediction and the optimization
# Plot for muD
plt.figure(figsize=(10, 6))
plt.plot(direct_allParams[3] - gt_allParams[3], marker='o', label="muD direct prediction")
plt.plot(optim_allParams[3] - gt_allParams[3], marker='o', label="muD optimization")
plt.plot(optim_allParams_CMA_ES_network[3] - gt_allParams[3], marker='o', label="muD CMA-ES Network")
plt.title("Difference between direct prediction and optimization for muD")
plt.legend()

# Plot for muRho
plt.figure(figsize=(10, 6))
plt.plot(direct_allParams[4] - gt_allParams[4], marker='o', label="muRho direct prediction")
plt.plot(optim_allParams[4] - gt_allParams[4], marker='o', label="muRho optimization")
plt.plot(optim_allParams_CMA_ES_network[4] - gt_allParams[4], marker='o', label="muRho CMA-ES Network")
plt.title("Difference between direct prediction and optimization for muRho")
plt.legend()


#%% plot the difference between the direct prediction and the optimization
# Plot for muD (relative difference)
plt.figure(figsize=(10, 6))
plt.plot((direct_allParams[3] - gt_allParams[3]) / gt_allParams[3], marker='o', label="muD direct prediction (relative)")
plt.plot((optim_allParams[3] - gt_allParams[3]) / gt_allParams[3], marker='o', label="muD optimization (relative)")
plt.plot((optim_allParams_CMA_ES_network[3] - gt_allParams[3]) / gt_allParams[3], marker='o', label="muD CMA-ES Network (relative)")
plt.title("Relative Difference between direct prediction and optimization for muD")
plt.legend()

# Plot for muRho (relative difference)
plt.figure(figsize=(10, 6))
plt.plot((direct_allParams[4] - gt_allParams[4]) / gt_allParams[4], marker='o', label="muRho direct prediction (relative)")
plt.plot((optim_allParams[4] - gt_allParams[4]) / gt_allParams[4], marker='o', label="muRho optimization (relative)")
plt.plot((optim_allParams_CMA_ES_network[4] - gt_allParams[4]) / gt_allParams[4], marker='o', label="muRho CMA-ES Network (relative)")
plt.title("Relative Difference between direct prediction and optimization for muRho")
plt.legend()

#%% plot the difference between the direct prediction and the optimization


#%%

plt.figure(figsize=(10, 6))
plt.plot(df["run_id"], df["parameters_difference_muRho (last)"], marker='o')
plt.plot(df["run_id"], df["parameters_difference_muD (last)"], marker='o')
# %%
calcMean_muRho = df["parameters_difference_muRho (last)"].abs().mean()
calcStdErr_muRho = df["parameters_difference_muRho (last)"].abs().sem()
print(f"Mean of parameters_difference_muRho (last): {calcMean_muRho}, Standard Error: {calcStdErr_muRho}")

calcMean_muD = df["parameters_difference_muD (last)"].abs().mean()
calcStdErr_muD = df["parameters_difference_muD (last)"].abs().sem()
print(f"Mean of parameters_difference_muD (last): {calcMean_muD}, Standard Error: {calcStdErr_muD}")
# %%  plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df["parameters_difference_muRho (last)"].abs(), bins=20, alpha=0.5, label='muRho')
plt.hist(df["parameters_difference_muD (last)"].abs(), bins=20, alpha=0.5, label='muD')


# %%
calcMean_muRho = df["parameters_difference_muRho (last)"].abs().mean()
calcStdErr_muRho = df["parameters_difference_muRho (last)"].abs().sem()
print(f"Mean of parameters_difference_muRho (last): {calcMean_muRho}, Standard Error: {calcStdErr_muRho}")

calcMean_muD = df["parameters_difference_muD (last)"].abs().mean()
calcStdErr_muD = df["parameters_difference_muD (last)"].abs().sem()
print(f"Mean of parameters_difference_muD (last): {calcMean_muD}, Standard Error: {calcStdErr_muD}")

x_diff = np.array(df["parameters_difference_x (last)"].abs())
y_diff = np.array(df["parameters_difference_y (last)"].abs())
z_diff = np.array(df["parameters_difference_z (last)"].abs())

diff_direct_origin = x_diff**2 + y_diff**2 + z_diff**2
diff_direct_origin = np.sqrt(diff_direct_origin)
calcMean_origin = np.mean(diff_direct_origin)
calcStdErr_origin = np.std(diff_direct_origin) / np.sqrt(len(diff_direct_origin))
print(f"Mean of parameters_difference_origin (last): {calcMean_origin}, Standard Error: {calcStdErr_origin}")

# %%
import numpy as np
from tabulate import tabulate
direcPredParams = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_xindi-maquis-4/test_results.npy", allow_pickle=True).item()

params = ["x", "y", "z", "muD", "muRho"]

directMu = direcPredParams["preds"].T[3]
gtMu = direcPredParams["trues"].T[3]
# %%
labels = direcPredParams["labels"]
diff_direct_muD = np.abs(direcPredParams["preds"].T[3] - direcPredParams["trues"].T[3])

diff_direct_muD_mean = np.mean(diff_direct_muD)
diff_direct_muD_std = np.std(diff_direct_muD)
diff_direct_muD_std_err = np.std(diff_direct_muD) / np.sqrt(len(diff_direct_muD))

print(f"Mean of direct prediction {params}: {diff_direct_muD_mean}, Standard Deviation: {diff_direct_muD_std}")
# %%
diff_direct_muRho = np.abs(direcPredParams["preds"].T[4] - direcPredParams["trues"].T[4])
diff_direct_muRho_mean = np.mean(diff_direct_muRho)
diff_direct_muRho_std = np.std(diff_direct_muRho)
diff_direct_muRho_std_err = np.std(diff_direct_muRho) / np.sqrt(len(diff_direct_muRho))

print(f"Mean of direct prediction {params}: {diff_direct_muRho_mean}, Standard Deviation: {diff_direct_muRho_std}")
# %%
diff_direct_origin = np.linalg.norm(direcPredParams["preds"].T[:3] - direcPredParams["trues"].T[:3], axis=0)
diff_direct_origin_mean = np.mean(diff_direct_origin)
diff_direct_origin_std = np.std(diff_direct_origin)
diff_direct_origin_std_err = np.std(diff_direct_origin) / np.sqrt(len(diff_direct_origin))

print(f"Mean of direct prediction origin distance: {diff_direct_origin_mean}, Standard Deviation: {diff_direct_origin_std}")

# %% make the bar chart for diff_direct_muD_mean, calcMean_muD, and origin with error bars

# Data for the bar chart
means = [
    [diff_direct_muD_mean, calcMean_muD],
    [diff_direct_muRho_mean, calcMean_muRho],
    [diff_direct_origin_mean, calcMean_origin]
]
errors = [
    [diff_direct_muD_std_err, calcStdErr_muD],
    [diff_direct_muRho_std_err, calcStdErr_muRho],
    [diff_direct_origin_std_err, calcStdErr_origin]
]
labels = ["muD", "muRho", "Origin"]
categories = ["Direct Prediction", "Optimization"]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects2 = ax.bar(x + width / 2, [m[1] for m in means], width, yerr=[e[1] for e in errors], label=categories[1], alpha=0.7)
rects1 = ax.bar(x - width / 2, [m[0] for m in means], width, yerr=[e[0] for e in errors], label=categories[0], alpha=0.7)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel("Mean and Standard Deviation")
ax.set_xlabel("Parameters")
ax.set_title("Comparison of Direct Prediction and Optimization")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()

# %% plot mu over patients
plotNumber = 100
plt.figure(figsize=(10, 6))
plt.plot( df["parameters_difference_muRho (last)"][:plotNumber], marker='o', label="muRho optimiz")
diff_direct_rho = direcPredParams["preds"].T[4] - direcPredParams["trues"].T[4]

plt.plot(diff_direct_rho[:plotNumber] ,marker='o', label="muRho direct prediction")
plt.legend()
#%%
argwhere = np.where(np.abs(df["parameters_difference_muRho (last)"]) > 5)[0]
print(df.iloc[argwhere])

# %% plot mu over patients
#plotNumber = 50 
plt.figure(figsize=(10, 6))
plt.plot(df["parameters_muRho (last)"][:plotNumber], marker='o', label="muRho optimiz")
diff_direct_rho = direcPredParams["preds"].T[4] - direcPredParams["trues"].T[4]
gt_rho = direcPredParams["trues"].T[4]

plt.plot(direcPredParams["preds"].T[4][:plotNumber], marker='o', label="muRho direct prediction")



plt.plot(gt_rho[:plotNumber], marker='o', label="muRho GT")
plt.title("muRho")

plt.legend()
#plt.plot(df["run_id"], df["parameters_muD (last)"], marker='o', label="muD")


# %%
#plotNumber = 50 
plt.figure(figsize=(10, 6))
plt.plot( df["parameters_muD (last)"][:plotNumber], marker='o', label="muD optimiz")
diff_direct_D = direcPredParams["preds"].T[3] - direcPredParams["trues"].T[3]
gt = direcPredParams["trues"].T[3]

plt.plot(direcPredParams["preds"].T[3][:plotNumber] ,marker='o', label="muD direct prediction")
plt.plot(gt[:plotNumber] ,marker='o', label="muD GT")
plt.title("muD")
plt.legend()




# %% origin 
#plotNumber = 50
plt.figure(figsize=(10, 6))
plt.plot(df["parameters_x (last)"][:plotNumber], marker='o', label="x optimiz")
diff_direct_x = direcPredParams["preds"].T[0] - direcPredParams["trues"].T[0]
plt.plot(diff_direct_x[:plotNumber] ,marker='o', label="x direct prediction")
plt.plot(direcPredParams["trues"].T[0][:plotNumber] ,marker='o', label="x GT")
plt.legend()


# %% y
#plotNumber = 50
plt.figure(figsize=(10, 6)) 
plt.plot(df["parameters_y (last)"][:plotNumber], marker='o', label="y optimiz")
diff_direct_y = direcPredParams["preds"].T[1] - direcPredParams["trues"].T[1]
plt.plot(diff_direct_y[:plotNumber] ,marker='o', label="y direct prediction")
plt.plot(direcPredParams["trues"].T[1][:plotNumber] ,marker='o', label="y GT")
plt.legend()


# %%
plotNumber = 500
plt.figure(figsize=(10, 6))
plt.plot(df["parameters_z (last)"][:plotNumber], marker='o', label="z optimiz")
diff_direct_z = direcPredParams["preds"].T[2] - direcPredParams["trues"].T[2]
plt.plot(diff_direct_z[:plotNumber] ,marker='o', label="z direct prediction")
plt.plot(direcPredParams["trues"].T[2][:plotNumber] ,marker='o', label="z GT")
plt.legend()

# %%
