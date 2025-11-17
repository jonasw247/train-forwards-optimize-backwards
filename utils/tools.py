#%%
import torch
import numpy as np
import torch.nn as nn
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy
from scipy.ndimage import distance_transform_edt



# %%
def create_standard_plan(core_segmentation, distance):
    
    # Calculate the Euclidean distance for areas not in the core
    distance_transform = distance_transform_edt(~ (core_segmentation >0))
    
    # Mark regions within a specific distance from the core
    dilated_core = distance_transform <= distance

    return dilated_core


def find_threshold(volume, target_volume, tolerance=0.01, initial_threshold=0.5,maxIter = 10000):

    if np.sum(volume > 0) < target_volume:
        print("Volume is too small")
        return 0.00000000001 #above the model

    # Define the initial threshold, step, and previous direction
    threshold = initial_threshold
    step = 0.1
    previous_direction = None

    # Calculate the current volume
    current_volume = np.sum(volume > threshold)

    # Iterate until the current volume is within the tolerance of the target volume

    while abs(current_volume - target_volume) / target_volume > tolerance:
        # Determine the current direction
        if current_volume > target_volume:
            direction = 'increase'
        else:
            direction = 'decrease'

        # Adjust the threshold
        if direction == 'increase':
            threshold += step
        else:
            threshold -= step

        # Check if the threshold is out of bounds
        if threshold < 0 or threshold > 1:
            return 1.01 #above the model

        # Update the current volume
        current_volume = np.sum(volume > threshold)

        # Reduce the step size if the direction has alternated
        if previous_direction and previous_direction != direction:
            step *= 0.5

        # Update the previous direction
        previous_direction = direction

        maxIter -= 1
        if maxIter < 0:
            print("Max Iter reached, no threshold found")
            return 1.1 #above the model

    return threshold

def getRecurrenceCoverage(tumorRecurrence, treatmentPlan):

    if np.sum(tumorRecurrence) <=  0.00001:
        return 1

    # Calculate the intersection between the recurrence and the plan
    intersection = np.logical_and(tumorRecurrence, treatmentPlan)

    # Calculate the coverage as the ratio of the intersection to the recurrence
    coverage = np.sum(intersection) / np.sum(tumorRecurrence)
    return coverage

# relative part of the prediction that is inside recurrence
def getPredictionInRecurrence(tumorRecurrence, treatmentPlan):

    if np.sum(tumorRecurrence) <=  0.00001:
        return 0
    
    if np.sum(treatmentPlan) <= 0.00001:
        return 0
    
    # normalize sum of treatment plan to 1
    normalizedTreatmentPlan = treatmentPlan / np.sum(treatmentPlan)

    coverage = np.sum((tumorRecurrence > 0) * normalizedTreatmentPlan)

    return coverage

