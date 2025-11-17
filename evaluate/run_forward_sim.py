
#%%
from TumorGrowthToolkit.FK import Solver as FKSolver
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import time
import os


def shift_origin_around(par,radius):
    pull_to = np.array([121/240, 115/240, 71/155]) # this is in the center and has not csf
    inputPArs = np.array([par["NxT1_pct"], par["NyT1_pct"], par["NzT1_pct"]])

    directionVe = pull_to - inputPArs

   
    directionVeScale = radius * directionVe / np.linalg.norm(directionVe)

    newVec = inputPArs + directionVeScale

    #if np.linalg.norm(directionVe) <= radius:
    #    newVec = 

    parOut = par.copy()
    parOut["NxT1_pct"] = newVec[0]
    parOut["NyT1_pct"] = newVec[1]
    parOut["NzT1_pct"] = newVec[2]
    return parOut


def runForwardSimulation(params, savePath=None, getAtlas=False):

    atlasPath = "/home/jonas/workspace/programs/neural-surrogate-in-med/data/sub-mni152_tissues_space-sri.nii.gz"

    if getAtlas:
        print(savePath)

        bratsPat = savePath.split("_")[-3]
        datasetPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats-lucas-results"
        stringPat = np.sort(os.listdir(datasetPath)).tolist()[int(bratsPat)]
        savePath = savePath.replace(".nii.gz", f"_{stringPat}.nii.gz")
        atlasPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats-lucas-results/" + stringPat+ "/processed/tissue_segmentation/tissue_seg.nii.gz"

    atlasTissue = nib.load(atlasPath).get_fdata()
    affine = nib.load(atlasPath).affine

    wm_data = atlasTissue == 3
    gm_data = atlasTissue == 2

    # Set up parameters
    parameters = {
        'Dw': np.maximum(params["Dw"], 0.00001) ,          # Diffusion coefficient for white matter
        'rho':  np.maximum(params["rho"], 0.00001),        # Growth rate
        'RatioDw_Dg': 10,  # Ratio of diffusion coefficients in white and grey matter
        'gm': gm_data,      # Grey matter data
        'wm': wm_data,      # White matter data
        'NxT1_pct': np.clip(params["NxT1_pct"], 0.001, 0.999),    # tumor position [%]
        'NyT1_pct': np.clip(params["NyT1_pct"], 0.001, 0.999),    # tumor position [%]
        'NzT1_pct': np.clip(params["NzT1_pct"], 0.001, 0.999),    # tumor position [%]
        'init_scale': 1., #scale of the initial gaussian
        'resolution_factor': 1, #resultion scaling for calculations
        'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
        'verbose': True, #printing timesteps 
        'time_series_solution_Nt': None,#64, #64, # number of timesteps in the output
        #'stopping_volume' : stopping_volume, #stop when the volume of the tumor is less than this value
        'stopping_time' : 100, #stop when the time is greater than this value
    }        

    shiftBy = 0.001

    #Run the FK_solver and plot the results
    start_time = time.time()
    
    totalShift = 0 
    for i in range(1000):
        fk_solver = FKSolver(parameters)
        result = fk_solver.solve()


        if result["success"] == False and result["error"] == "Initial tumor position is outside the brain matter":
            #print("The FK solver did not run successfully. Please check the parameters.")
            #print(result["error"])

            
            parameters = shift_origin_around(parameters, shiftBy)
            print("shift_to", parameters["NxT1_pct"], parameters["NyT1_pct"], parameters["NzT1_pct"])

        elif result["success"] == True:
            print("The FK solver ran successfully.")
            totalShift += shiftBy
            print(f"Shifting the tumor position by {shiftBy}.")
            break
        else:   
            print("The FK solver did not run successfully. Please check the parameters.")
            print(result["error"])
            break

    
    end_time = time.time()  # Store the end time
    execution_time = int(end_time - start_time)  # Calculate the difference
    print(f"Execution time: {execution_time} seconds")
    # close all figures
    plt.close("all")

    

    if savePath is not None:
        # Save the result as a NIfTI file
        result_img = nib.Nifti1Image(result["final_state"], affine)
        nib.save(result_img, savePath)

    # Save the result as a NIfTI file
    return result["final_state"]
#%%
if __name__ == "__main__":
    # Example parameters
    params = {
        "Dw": 0.2371937279705981,
        "rho":0.025256267990744315,
        "NxT1_pct": 0.8091666696468989,
        "NyT1_pct": 0.6246752003828683,
        "NzT1_pct": 0.6322580674963613
    }
    params_at_perfect_location = {
        "Dw": 1.6371937279705981,
        "rho":0.025256267990744315,
        "NxT1_pct": 121/240,  
        "NyT1_pct": 115/240,
        "NzT1_pct": 71/155,
    }

    # Run the forward simulation
    result = runForwardSimulation(params, savePath="forward_simulation_result.nii.gz")
    a = {'Dw': 1.6371937279705981, 'rho': 0.025256267990744315, 'NxT1_pct': 0.9291666696468989, 'NyT1_pct': 0.6246752003828683, 'NzT1_pct': 0.6322580674963613}
# %% thest the shifting...
if __name__ == "__main__":

    
    # Example parameters
    params = {
        "Dw": 1.6371937279705981,
        "rho":0.025256267990744315,
        "NxT1_pct": 0.9291666696468989,
        "NyT1_pct": 0.6246752003828683,
        "NzT1_pct": 0.6322580674963613
    }
    
    newParams = shift_origin_around(params, 0.003)
    newParams2  = shift_origin_around(newParams, 0.003)

    plt.plot([params["NxT1_pct"], newParams["NxT1_pct"], newParams2["NxT1_pct"]], [params["NyT1_pct"], newParams["NyT1_pct"], newParams2["NyT1_pct"]], "o-")

# %%