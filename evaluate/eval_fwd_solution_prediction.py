#%%
import numpy as np
import nibabel as nib
import pandas as pd
from skimage.metrics import structural_similarity as calcSSIM
from sklearn.metrics import mutual_info_score
import os
import matplotlib.pyplot as plt

# %%
def normalized_cross_correlation(x, y):
    x = x.ravel() - np.mean(x)
    y = y.ravel() - np.mean(y)
    return np.sum(x * y) / (np.linalg.norm(x) * np.linalg.norm(y))

def eval_for_one_pat(gtPath, tumorFilePath, atlasPath):
    atlasTissue = nib.load(atlasPath).get_fdata()

    tumordata = nib.load(tumorFilePath).get_fdata()
    patIDx = tumorFilePath.split("_")[-1].split(".")[0]
    datasetNum =  "patient_00" + patIDx+ "/tumor_concentration.nii.gz"
    groundTruth = nib.load(os.path.join(gtPath, datasetNum, )).get_fdata()

    masked_pred = tumordata.copy()
    masked_gt = groundTruth.copy()

    masked_pred[atlasTissue < 1.5] = 0
    masked_gt[atlasTissue < 1.5] = 0

    #mse
    diff = masked_pred - masked_gt
    
    mse = np.mean((diff) ** 2)
    mae = np.mean(np.abs(diff))

    gtVolume = np.sum(masked_gt)
    volume = np.sum(masked_pred)
    mseNormed = np.mean((diff) ** 2) / gtVolume


    ssim = calcSSIM(masked_pred, masked_gt, data_range=masked_gt.max() - masked_gt.min(), multichannel=False)

    mi_val = mutual_info_score(None, None, contingency=np.histogram2d(
    masked_pred.ravel(), masked_gt.ravel(), bins=64
)[0])

    ncc= normalized_cross_correlation(masked_pred, masked_gt)


    resltDict = {
        "mse": mse,
        "mae": mae,
        "mseNormed": mseNormed,
        "ssim": ssim,
        "mi_val": mi_val,
        "normalized_cross_correlation": ncc,
        "gtVolume": gtVolume,
        "volume": volume,
        "patIDx": patIDx,
        "tumorFilePath": tumorFilePath,
    }
    return resltDict


def eval_one_run(resultPath, max_files = 1000000):
    atlasPath = "/home/jonas/workspace/programs/neural-surrogate-in-med/data/sub-mni152_tissues_space-sri.nii.gz"
    gtPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/synthetic_FK_Michals_solver_smaller"

    fileListAll = sorted(os.listdir(resultPath))
    
    # only files with .nii.gz
    fileList = [f for f in fileListAll if f.endswith(".nii.gz")]


    resultDict = {}

    for i, file in enumerate(fileList):
        if i > max_files:
            break

        if i % 1 == 0:
            print(f"Processing {i} of {len(fileList)} files")

        filePath = os.path.join(resultPath, file)
        #print(f"Processing {filePath}")
        resDict = eval_for_one_pat(gtPath, filePath, atlasPath)
        for key, value in resDict.items():
            if key in resultDict:
                resultDict[key].append(value)
            else:
                resultDict[key] = [value]

    return resultDict

#

resultPaths =  ["/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-med/cma_20250503_170946/cma-es-network_fwd_results",
               "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_rural-wood-5/direct_prediction_results",
               
               #"/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/gradient-based-optim/run_20250503_191918/inverse_optimization_results",# good long run with init of deep learning result
               "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/gradient-based-optim/run_20250507_181640/inverse_optimization_results",# rerun

               #"/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_xindi-maquis-4/direct_prediction_results_origin_at_com",
              #"/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/gradient-based-optim/run_20250406_153026/inverse_optimization_results_origin_at_com",
              "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-medCMA-ES-classical_solver/cma_20250513_234624/cma-es-network_fwd_results", # the first classic cma-es run
]
#resultPaths = ["/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-medCMA-ES-classical_solver/cma_20250504_183704/cma-es-network_fwd_results",]
#resultPaths = ["/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-medCMA-ES-classical_solver/cma_20250513_234624/cma-es-network_fwd_results"]

#resultPaths =  ["/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-med/cma_20250503_170946/cma-es-network_fwd_results"]
#%%
len(os.listdir("/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/gradient-based-optim/run_20250503_191918/inverse_optimization_results",))

#%% extract the values
"""   
for  i, path in enumerate(resultPaths):
    allResultsForRun = eval_one_run(path, max_files = 500)
    print(f"Results for {path}:")
    #save as npy
    np.save(os.path.join(path, "summary_results.npy"), allResultsForRun)
 """
   

#%% load the results
allResultsForRun = []
for i, path in enumerate(resultPaths):
    res = np.load(os.path.join(path, "summary_results.npy"), allow_pickle=True).item()

    
    #res["mse_normed"] = np.array(res["mse"]) / np.array(res["gtVolume"])
    res["maeNormed"] = np.array(res["mae"]) / np.array(res["gtVolume"])


    #add stuff 
    allResultsForRun
    allResultsForRun.append(res)

lala = allResultsForRun.copy()
direcctPredictions = lala[1].copy()
inversePredictions = lala[2].copy()
criterionMetric = "mse"

bestOfBoth ={}
for i, key in enumerate(direcctPredictions):

    bestOfBoth[key] = []
    for j in range(500): #len(direcctPredictions[criterionMetric])): TODO
        #print(f"Processing {j} ")

        valList = []
        if direcctPredictions[criterionMetric][j] < inversePredictions[criterionMetric][j]:
            bestOfBoth[key].append(direcctPredictions[key][j])
        else:
            bestOfBoth[key].append(inversePredictions[key][j])
        
    #bestOfBoth[key] = valList
    print(bestOfBoth[key])


allResultsForRun.append(bestOfBoth)
experimentNames = []

for i, path in enumerate(resultPaths):
    experimentNames.append(path.split("/")[-2] + "/" + path.split("/")[-1])
experimentNames.append("0/Best of both")


#print lengt of each result
for i, path in enumerate(resultPaths):
    print(f"Results for {path}:")
    print("len: ", len(allResultsForRun[i]["mse"]))
    

print("mse 0" , np.mean(allResultsForRun[0]["mse"][:150]))
print("mse 1" , np.mean(allResultsForRun[1]["mse"][:150]))
print("mse 2" , np.mean(allResultsForRun[2]["mse"][:150]))
print("mse 3" , np.mean(allResultsForRun[3]["mse"][:150]))

# median
print("median mse 0" , np.median(allResultsForRun[0]["mse"][:150]))
print("median mse 1" , np.median(allResultsForRun[1]["mse"][:150]))
print("median mse 2" , np.median(allResultsForRun[2]["mse"][:150]))


#%%
# Group values by shared exponent and move exponent to column header
def create_summary_table(simulated_results, expNames, stop = 1000000):

    simulated_results = simulated_results

    print("length of simulated results: ", len(simulated_results))
    #newS#imResults = []
    for i, result in enumerate(simulated_results):
        for key in result:
            if len(result[key]) < stop:
                #throw error if the length is not equal to stop
                print(f"Error: {key} has length {len(result[key])} but stop is {stop}")
            else:
                result[key] = result[key][:stop]
        #newSimResults.append(result)


    # Function to extract shared exponent and normalized values
    def extract_base_and_exponent(values):
        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(len(values))
        if mean == 0:
            return 0, "0.0000 ± 0.0000"
        exponent = int(np.floor(np.log10(abs(mean))))
        base_mean = mean / (10 ** exponent)
        base_stderr = std_err / (10 ** exponent)
        return exponent, f"{base_mean:.4f} ± {base_stderr:.4f}"

    # Build new DataFrame with exponents in header
    restructured_data = {}
    exponent_map = {}

    for metric in simulated_results[0].keys():
        if isinstance(simulated_results[0][metric][0], (float, int)):
            exps = []
            for i in range(len(simulated_results)):
                exp, _ = extract_base_and_exponent(simulated_results[i][metric])
                exps.append(exp)
            shared_exp = min(exps)  # Choose common smaller exponent for consistency
            exponent_map[metric] = shared_exp

    # Create a DataFrame using these adjusted values
    for i, result in enumerate(simulated_results):
        row = {}
        for metric, exponent in exponent_map.items():
            values = result[metric]
            mean = np.mean(values) / (10 ** exponent)
            std_err = (np.std(values, ddof=1) / np.sqrt(len(values))) / (10 ** exponent)
            row[metric] = f"{mean:.2f} ± {std_err:.2f}"
        restructured_data[f"{expNames[i]}"] = row

    # Build final DataFrame with exponent in column headers
    new_columns = [f"{metric} (×10^{exponent_map[metric]})" for metric in exponent_map]
    df_final = pd.DataFrame(restructured_data).T
    df_final.columns = new_columns
    #print(df_final)
    return df_final

create_summary_table(allResultsForRun, experimentNames, stop = 500)
#%% plot the volume over the mse for each method
plt.figure()
plt.title("Volume over MSE for each method")
plt.xlabel("Volume")
plt.ylabel("MSE")
for i in range(len(allResultsForRun)):
    if not i==3:
        continue
    plt.scatter(allResultsForRun[i]["volume"], allResultsForRun[i]["mse"], label=experimentNames[i], marker=".")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()
#%%
start = 0
stop = 50
plotMetrics = "mse" # ["mse", "mae", "mseNormed", "ssim", "mi_val", "normalized_cross_correlation"]
# Create a DataFrame for the metrics

for i in range(len(allResultsForRun)):
    if i==2:
        pass#continue
    plt.plot(allResultsForRun[i][plotMetrics][start:stop], label=experimentNames[i])
plt.legend()

plt.title("MSE and MAE for each method")
plt.xlabel("Patient Number")
plt.ylabel("MSE")

#%%  plot volume
plt.figure()
plt.plot(allResultsForRun[0]["gtVolume"][start:stop], label="GT Volume")
for i in range(0, len(allResultsForRun)):
    i = 3
    plt.plot(np.array(allResultsForRun[i]["volume"][start:stop]) , label=experimentNames[i])
plt.legend()

# open "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-medCMA-ES-classical_solver/cma_20250504_183704/cma-es-network_fwd_results/summary_results.npy"

dictData = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/cma-es-inverse-neural-surrogate-in-medCMA-ES-classical_solver/cma_20250504_183704/cma-es-network_fwd_results/summary_results.npy", allow_pickle=True).item()
#%% scatter
plt.figure()
#for i in range(1, len(allResultsForRun)):
i = 2
plt.scatter(allResultsForRun[0][plotMetrics][start:stop], allResultsForRun[i][plotMetrics][start:stop], label=experimentNames[i], marker=".")

#diagonal
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("MSE" + experimentNames[0])
plt.ylabel("MSE" + experimentNames[i])
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.title("MSE and MAE for each method")
plt.show()


#%% numpy histogram # paperfigure 
plt.figure(figsize=(10, 3))
plotMetrics = "mse"  # ["mse", "mae", "mseNormed", "ssim", "mi_val", "normalized_cross_correlation"]

# Prepare data for boxplot
data = [allResultsForRun[i][plotMetrics] for i in range(len(allResultsForRun))]

plt.boxplot(data, labels=experimentNames, showmeans=True)
plt.xlabel("Methods")
plt.ylabel("MSE")
plt.title("MSE distribution for each method")
plt.yscale("log")
plt.xticks(rotation=45)
plt.tight_layout()

#%%
#%% numpy histogram # paperfigure 
plt.figure(figsize=(8, 3))
plotMetrics = "mse" # ["mse", "mae", "mseNormed", "ssim", "mi_val", "normalized_cross_correlation"]
maxAll = 0
minAll = 1000000000
for i in range(len(allResultsForRun)):

    max = np.max(allResultsForRun[i][plotMetrics])
    min = np.min(allResultsForRun[i][plotMetrics])
    if max > maxAll:
        maxAll = max
    if min < minAll:
        minAll = min
    print(f"Max {experimentNames[i]}: {max}")

labelsFromExp = ["CMA-ES Nural Surrogate", 
                 "Direct Inverse Prediction", 
                 "GB Neural Surrogate",  "CMA-ES classical solver","(Ours) Inverse Prediction and GB Optimization"]                           
for i in range(len(allResultsForRun)):
    if i ==3:
        continue
    vals, x = np.histogram(np.log(allResultsForRun[i][plotMetrics]), bins=50, density=True, range=(np.log(min), np.log(maxAll)))
    x = np.exp(x)
    plt.plot(x[:-1], vals, label=labelsFromExp[i])
plt.xlabel("MSE")
plt.ylabel("Density")
plt.title("Error Distribution")
plt.xscale("log")
plt.legend()
plt.savefig("error_distribution.pdf", bbox_inches='tight')
#%%
len(allResultsForRun[0][plotMetrics])
#%% do paired t test
from scipy.stats import ttest_rel
a,b = 1,3
p_values = []
pvalue = ttest_rel(allResultsForRun[a]["mse"], allResultsForRun[b]["mse"])
print(f"Paired t-test between {experimentNames[a]} and {experimentNames[b]}: p-value = {pvalue.pvalue:.4f}")
#%% plot mse
import matplotlib.pyplot as plt
plt.plot(allResultsForRun[0]["mse"], label="MSE")
plt.plot(allResultsForRun[1]["mse"], label="MAE")

#%%
allResultsForRun = []
for path in resultPaths:
    res = np.load(os.path.join(path, "s ummary_results.npy"), allow_pickle=True).item()
    allResultsForRun.append(res)

# Create summary table: average per metric for each experiment
summary_table = {}
for i, result in enumerate(allResultsForRun):
    summary = {metric: np.mean(values) for metric, values in result.items() if isinstance(values[0], (float, int))}
    summary_table[f"Experiment {i+1}"] = summary

# Convert to DataFrame
df_summary = pd.DataFrame(summary_table).T

df_summary
#%%
# Display the table
#import ace_tools as tools; tools.display_dataframe_to_user(name="Experiment Metrics Summary", dataframe=df_summary)
"""
# Re-run the final display step after re-importing the necessary tools
tools.display_dataframe_to_user(name="Simulated Experiment Metrics Summary", dataframe=df_summary_simulated)

#%% generate latex table for all

#%%


#%% saveTheForwardResults
saveDir = {
    "mse": all_results,
    "mae": all_resultsMAE,
    "patIDXs": patIDXs,
    "pathLabels": pathLabels,
}
np.save("./eval_fwd_solution_prediction.npy", saveDir)
# %% 
import matplotlib.pyplot as plt
stop = 50
start =0
for i, path in enumerate(resultPaths[:2]):
    plt.plot(patIDXs[start:stop], all_results.T[i, start:stop], label=pathLabels[i], marker=".")

plt.xlabel("Patient Number")
plt.ylabel("MSE")
plt.legend()
plt.show()
#same for mae
plt.figure()
for i, path in enumerate(resultPaths):
    plt.scatter(patIDXs[start:stop], all_resultsMAE.T[i, start:stop], label=pathLabels[i], marker=".")
plt.xlabel("Patient Number")
plt.ylabel("MAE")
plt.legend()
plt.show()
#%%
#plot normed
plt.figure()
for i, path in enumerate(resultPaths):
    plt.plot(patIDXs[start:stop], all_resultsMSENormed.T[i, start:stop], label=pathLabels[i], marker=".")
plt.xlabel("Patient Number")''
plt.ylabel("MSE normed")
plt.legend()
plt.show()

#%%  get all over 0.01
bad = np.array(patIDXs)[np.where(all_results.T[1] > 0.04)[0]]
#values = np.array(all_results.T[1][np.where(all_results.T[1] > 0.01)[0]])
print("Bad examples:")
for i in bad:
    print(i)#, values[np.where(all_results.T[1] > 0.01)[0]])

#%%
print("Bad examples:")
print(all_results.T[1][np.where(all_results.T[1] > 0.01)[0]])
#%% print the bad examples
bad = patIDXs[np.where(all_results.T[1] > 0.1)[0]]
print("Bad examples:")
for i in bad:
    print(i)

#%%  print table 
print("MSE Table")
print("mean MSE for each method")
print("-----------------------------------------------------")
print("Method\t\t\tMSE")
stds = np.std(all_results, axis=1)
for i, path in enumerate(resultPaths):
    print(f"{pathLabels[i]}: {np.mean(all_results.T[i]):.4f} +- {stds[i]:.4f}")

defaultFallback = np.min(all_results.T, axis=0)
print(f"Default fallback: {np.mean(defaultFallback):.4f} +- {np.std(defaultFallback):.4f}")

print("-----------------------------------------------------")
print("")


#MAE

# median now
print("median MSE for each method")
print("-----------------------------------------------------")
print("Method\t\t\tMSE")
stds = np.std(all_results, axis=1)
for i, path in enumerate(resultPaths):
    print(f"{pathLabels[i]}: {np.median(all_results.T[i]):.4f} +- {stds[i]:.4f}")

defaultFallback = np.min(all_results.T, axis=0)
print(f"Default fallback: {np.median(defaultFallback):.4f} +- {np.std(defaultFallback):.4f}")

print("-----------------------------------------------------")
#%%MAE


print("MAE Table")
print("mean MAE for each method")
print("-----------------------------------------------------")
print("Method\t\t\tMAE")
stds = np.std(all_resultsMAE, axis=1)
for i, path in enumerate(resultPaths):
    print(f"{pathLabels[i]}: {np.mean(all_resultsMAE.T[i]):.4f} +- {stds[i]:.4f}")

defaultFallback = np.min(all_resultsMAE.T, axis=0)
print(f"Default fallback: {np.mean(defaultFallback):.4f} +- {np.std(defaultFallback):.4f}")
print("-----------------------------------------------------")

print("median MAE for each method")
print("-----------------------------------------------------")
print("Method\t\t\tMAE")
stds = np.std(all_resultsMAE, axis=1)
for i, path in enumerate(resultPaths):
    print(f"{pathLabels[i]}: {np.median(all_resultsMAE.T[i]):.4f} +- {stds[i]:.4f}")


defaultFallback = np.min(all_resultsMAE.T, axis=0)
print(f"Default fallback: {np.median(defaultFallback):.4f} +- {np.std(defaultFallback):.4f}")
print("-----------------------------------------------------")


#%%
def pltscatter(resultsPlot):
    x, y1, y2 = 1,0,2
    plt.plot(resultsPlot.T[x], resultsPlot.T[y1], ".", label = pathLabels[y1])

    plt.xlabel(pathLabels[x])
    plt.ylabel(pathLabels[y1])
    #plot diag
    plt.plot([0, 1], [0, 1], "k--")
    # log scale
    plt.xscale("log")
    plt.yscale("log")
    plt.title("MSE")
    plt.legend()

    plt.show()
    
    x, y1, y2 = 1,2,0
    plt.plot(resultsPlot.T[x], resultsPlot.T[y2], ".", label = pathLabels[y2])

    plt.xlabel(pathLabels[x])
    plt.ylabel(pathLabels[y2])
    #plot diag
    plt.plot([0, 1], [0, 1], "k--")
    # log scale
    plt.xscale("log")
    plt.yscale("log")
    plt.title("MSE")
    plt.legend()
    plt.show()

    x, y1, y2 = 1,0,2
    plt.plot(resultsPlot.T[x], resultsPlot.T[y2], ".", label = pathLabels[y2])
    plt.xlabel(pathLabels[x])
    plt.ylabel(pathLabels[y2])
    #plot diag
    plt.plot([0, 1], [0, 1], "k--")
    # log scale
    plt.xscale("log")
    plt.yscale("log")
    plt.title("MSE")
    plt.legend()

pltscatter(all_results)
#pltscatter(all_resultsMAE)
#pltscatter(all_resultsMSENormed)



#%%

pathDir = "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/gradient-based-optim/"+ group 

inverseOptimizationDF = pd.read_csv(f"{pathDir}/inverse_optimization_results.csv")
df = inverseOptimizationDF.copy()

D = df["parameters_muD (last)"]
rho = df["parameters_muRho (last)"]

difD = df['parameters_difference_muD (last)']
difRho = df['parameters_difference_muRho (last)']

#keys 
df.keys()

#plot scatter
plt.scatter(D, rho)
plt.xlabel("Diffusion coefficient")
plt.ylabel("Growth rate")
plt.title("Diffusion coefficient vs Growth rate")
plt.show()
#plot scatter
plt.scatter(difRho, difD)
plt.xlabel("Diffusion coefficient")
plt.ylabel("Growth rate")
plt.title("Diffusion coefficient vs Growth rate")
plt.show()
#%% error
stop = 250
#scatter Diff D and results
plt.scatter(difRho[:stop], all_results.T[1][:stop], marker=".")
plt.xlabel("growth rate err")
plt.ylabel("MSE")
plt.title(" MSE " + str(pathLabels[1]))

plt.show()
#scatter Diff rho and results
plt.scatter(difD[:stop], all_results.T[1][:stop], marker=".")

plt.xlabel("Diffusion coefficient err")
plt.ylabel("MSE")
plt.title(" MSE " + str(pathLabels[1]))
plt.show()

#scatter Diff D and results
color = np.array(all_results.T[1])
plt.scatter(rho[:stop], all_results.T[1][:stop], marker=".",label="GT growth rate", c=color[:stop], cmap='bwr')
#plt.scatter(rho[:stop], all_results.T[1][:stop], marker=".")
plt.xlabel("growth rate")
plt.ylabel("MSE")
plt.title(" MSE " + str(pathLabels[1]))

plt.show()
#scatter Diff rho and results
plt.scatter(D[:stop], all_results.T[1][:stop], marker=".")

plt.xlabel("Diffusion coefficient ")
plt.ylabel("MSE")
plt.title(" MSE " + str(pathLabels[1]))
plt.show()
#%%

direcPredParams = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_xindi-maquis-4/test_results.npy", allow_pickle=True).item()

params = ["x", "y", "z", "muD", "muRho"]

directMuD = direcPredParams["preds"].T[3]
gtMuD = direcPredParams["trues"].T[3]

directMurho = direcPredParams["preds"].T[4]
gtMurho = direcPredParams["trues"].T[4]

originPred = direcPredParams["preds"][:,:3]
gtOrigin = direcPredParams["trues"][:, :3]

diffOrigin = np.sum((originPred - gtOrigin)**2, axis=1)

#%%

#%%
#scatter gt and pred
stop = 250
lab = 1
color = np.array(all_results.T[lab])
import matplotlib.colors as mcolors

cmap = mcolors.LinearSegmentedColormap.from_list("RedGreen", ["blue", "red", "orange"])



plt.scatter(  gtMuD[:stop], gtMurho[:stop], marker=".",label="GT diffusion coefficient", c=color[:stop], cmap=cmap, alpha=1)

plt.xlabel("Diffusion coefficient ")
plt.ylabel("Growth rate")
plt.title(" MSE " + str(pathLabels[lab]))
plt.colorbar(label='MSE')

#%% plot mse over origin diff
plt.scatter(  diffOrigin[:stop], all_results.T[1][:stop], marker=".",label="GT diffusion coefficient", c=color[:stop], cmap=cmap, alpha=1)
plt.xlabel("origin diff ")
plt.ylabel("MSE")
plt.title(" MSE " + str(pathLabels[1]))
plt.colorbar(label='MSE')
plt.show()

#%% plot mse over volume
plt.scatter(  all_volumes[:stop], all_results.T[1][:stop], marker=".",label="GT diffusion coefficient", c=color[:stop], cmap=cmap, alpha=1)
plt.xlabel("volume ")
plt.ylabel("MSE")
plt.title(" MSE " + str(pathLabels[1]))
plt.colorbar(label='MSE')
plt.show()




#%%

#scatter Diff D and results
plt.scatter(  (rho[:stop] - difRho[:stop]) , all_results.T[1][:stop], marker=".",label="GT growth rate")
plt.scatter(  rho[:stop]  , all_results.T[1][:stop], marker=".", label="predicted growth rate")
plt.scatter(  (gtMurho[:stop]) , all_results.T[1][:stop], marker=".", label="tures")
plt.legend()

plt.xlabel("growth rate")
plt.ylabel("MSE")
plt.title(" MSE " + str(pathLabels[1]))

plt.show()
#scatter Diff rho and results
plt.scatter( (D - difD)[:stop], all_results.T[1][:stop], marker=".", label="GT diffusion coefficient")
plt.scatter( D[:stop], all_results.T[1][:stop], marker=".", label="predicted diffusion coefficient")
plt.legend()

plt.xlabel("Diffusion coefficient ")
plt.ylabel("MSE")
plt.title(" MSE " + str(pathLabels[1]))
plt.show()

#%% scatter muD gt and pred and color the mse
plt.scatter(  gtMuD[:stop], D[:stop], marker=".",label="GT diffusion coefficient")
#%%
#scatter muD and rho
plt.scatter(  gtMuD[:stop], gtMurho[:stop], marker=".",label="GT growth rate")

#%%
np.mean(all_results.T[0])
#%%
np.mean(all_results.T[1])
# %%
list(common_files)[16]
# %%
common_files

# %% histogram
plt.hist(all_results.T[1] - all_results.T[0], bins=100)
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.title("MSE distribution for " + pathLabels[1])
plt.show()
# %%
originGT = direcPredParams["trues"][:,:3]
# %%
originGT

# %%
"""