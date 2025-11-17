#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import os
import scipy.ndimage
from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
)
import json
import torch.nn.functional as F
#%%
def get_train_transforms(main_input, alpha_input, allTumorSeg, tumorCoreSeg, tumorNecrosisSeg, brainMask):

    train_transforms = Compose([
        # Apply random flip to both "image" and "label"
        RandFlipd(
            keys=["main_input", "alpha_input", "allTumorSeg", "tumorCoreSeg", "tumorNecrosisSeg", "brainMask"],
            prob=0.5,
            spatial_axis=1
        ),
        # Apply random affine transform to both
        RandAffined(
            keys=["main_input", "alpha_input", "allTumorSeg", "tumorCoreSeg", "tumorNecrosisSeg", "brainMask"],
            prob=0.7,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1)
        ),
    ])

    dataDict = {
        "main_input": main_input,
        "alpha_input": alpha_input,
        "allTumorSeg": allTumorSeg,
        "tumorCoreSeg": tumorCoreSeg,
        "tumorNecrosisSeg": tumorNecrosisSeg,
        "brainMask": brainMask
    }
    transformedData = train_transforms(dataDict)

    return transformedData["main_input"].as_tensor(), transformedData["alpha_input"].as_tensor(), transformedData["allTumorSeg"].as_tensor(), transformedData["tumorCoreSeg"].as_tensor(), transformedData["tumorNecrosisSeg"].as_tensor(), transformedData["brainMask"].as_tensor()
    

class BratsDataset(torch.utils.data.Dataset):
    def __init__(self, length,  path , registeredTissuePath, useRegisteredTissueSeg = True, is_half_resolution = False):
        super().__init__()
        self.do_dataset_transforms = False
        self.length = length

        self.useRegisteredTissueSeg = useRegisteredTissueSeg
        self.is_half_resolution = is_half_resolution

        self.path =path# "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_t1_and_t1c"

        self.registeredTissuePath = registeredTissuePath# "/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_good_registerd_atlas"

        allFiles= np.sort(os.listdir(self.path)).tolist()
        if len(allFiles) < length:
            self.length = len(allFiles)

        self.files = allFiles[:self.length]

    def __len__(self):
        return self.length

    def getPatientName(self, idx):
        return self.files[idx]

    def __getitem__(self, idx):
        a = idx
        tumorSegPath = self.path + "/" + self.files[idx] + "/preop/sub-" + self.files[idx] + "_ses-preop_space-sri_seg.nii.gz"

        if self.useRegisteredTissueSeg:
            #/mnt/8tb_slot8/jonas/workingDirDatasets/brats/brats_output_andre/base_line_SyN_s2/BraTS2021_00014/
            tissueSegPath = self.registeredTissuePath + "/" + self.files[idx] + "/tissue_segmentation.nii.gz"

        else:
            tissueSegPath = self.path + "/" + self.files[idx] + "/preop/" + self.files[idx] + "_ses-preop_space-sri_tissueSegBinary.nii.gz" # TODO_data 

        t1cPath = self.path + "/" + self.files[idx] + "/preop/sub-" + self.files[idx] + "_ses-preop_space-sri_t1c.nii.gz"
        flairPath = self.path + "/" + self.files[idx] + "/preop/sub-" + self.files[idx] + "_ses-preop_space-sri_flair.nii.gz"


        tumorSegImg = np.round(nib.load(tumorSegPath).get_fdata())
        tissueSegImg = np.round(nib.load(tissueSegPath).get_fdata())
        tissueSegImg[np.logical_and(tumorSegImg > 0, tissueSegImg <=1)] = 2


        t1cImg = nib.load(t1cPath).get_fdata()
        flairImg = nib.load(flairPath).get_fdata()

        t1cImg = t1cImg / np.mean(t1cImg[t1cImg>0])/5
        flairImg = flairImg /np.mean(flairImg[t1cImg>0])/5

        # stack t1c and flair
        main_input = torch.clamp(torch.stack([torch.tensor(t1cImg), torch.tensor(flairImg)], dim=0), 0, 1)


        alpha_input = torch.stack([torch.tensor(tissueSegImg)], dim=0)

        roundTumorSeg = np.round(tumorSegImg)

        # 1 = necrosis, 2 = edema, 4 = enhancing tumor
        allTumorSeg = torch.tensor(roundTumorSeg > 0).float().unsqueeze(0)
        tumorCoreSeg = torch.tensor(np.logical_or(
            roundTumorSeg == 1,roundTumorSeg == 4)).float().unsqueeze(0)
        tumorNecrosisSeg = torch.tensor(roundTumorSeg == 1).float().unsqueeze(0)

        brainMask = torch.tensor(tissueSegImg > 0).float().unsqueeze(0)

        #convert to float
        main_input = main_input.float()
        alpha_input = alpha_input.float()

        #padd all to 256x256x160
        padding = (2,3,8,8,8,8)
        main_input = torch.nn.functional.pad(main_input, padding)
        alpha_input = torch.nn.functional.pad(alpha_input, padding)
        allTumorSeg = torch.nn.functional.pad(allTumorSeg, padding)
        tumorCoreSeg = torch.nn.functional.pad(tumorCoreSeg, padding)
        tumorNecrosisSeg = torch.nn.functional.pad(tumorNecrosisSeg, padding)
        brainMask = torch.nn.functional.pad(brainMask, padding)

        if self.is_half_resolution:
            main_input = torch.nn.functional.interpolate(main_input.unsqueeze(0), scale_factor=0.5, mode="trilinear", align_corners=False).squeeze(0)
            alpha_input = torch.nn.functional.interpolate(alpha_input.unsqueeze(0), scale_factor=0.5, mode="nearest").squeeze(0)
            allTumorSeg = torch.nn.functional.interpolate(allTumorSeg.unsqueeze(0), scale_factor=0.5, mode="nearest").squeeze(0)
            tumorCoreSeg = torch.nn.functional.interpolate(tumorCoreSeg.unsqueeze(0), scale_factor=0.5, mode="nearest").squeeze(0)
            tumorNecrosisSeg = torch.nn.functional.interpolate(tumorNecrosisSeg.unsqueeze(0), scale_factor=0.5, mode="nearest").squeeze(0)
            brainMask = torch.nn.functional.interpolate(brainMask.unsqueeze(0), scale_factor=0.5, mode="nearest").squeeze(0)

        if self.do_dataset_transforms:
            main_input, alpha_input, allTumorSeg, tumorCoreSeg, tumorNecrosisSeg, brainMask = get_train_transforms(main_input, alpha_input, allTumorSeg, tumorCoreSeg, tumorNecrosisSeg, brainMask)

        return main_input, alpha_input, allTumorSeg, tumorCoreSeg, tumorNecrosisSeg, brainMask


#%%
def load_params(params_file_path):
        """
        Load tumor simulation parameters from a JSON file.

        Args:
            params_file (str): Path to the parameters JSON file.
            is_synthetic (bool): Flag indicating whether the data is synthetic.

        Returns:
            tuple: (params dict, diffusion coefficient (Dw), growth rate (rho), final time (T)).
        """
        with open(params_file_path, 'r') as f:
            jsonFile = json.load(f)

        params = jsonFile['parameters']
        D = float(params['Dw'])
        rho = float(params['rho'])
        T = float(jsonFile['results']['final_time'])
        return params, D, rho, T

def crop_and_downsample(shifted_img, img_center, crop_size, downsample_size, mode):
    """
    Crop and optionally downsample the image.

    Args:
        shifted_img (Tensor): Shifted image tensor.
        img_center (list): Center of the image.
        crop_size (int): Desired crop size (e.g., 120x120x120).
        downsample_size (int): Target downsample size (e.g., 64x64x64).

    Returns:
        Tensor: Cropped and downsampled image.
    """
    cropped_img = shifted_img[
        img_center[0] - crop_size // 2:img_center[0] + crop_size // 2,
        img_center[1] - crop_size // 2:img_center[1] + crop_size // 2,
        img_center[2] - crop_size // 2:img_center[2] + crop_size // 2
    ]

    if downsample_size is not None and not downsample_size == crop_size :
        print("downsampled to: ", downsample_size)
        output_size = (downsample_size, downsample_size, downsample_size)
        downsampled_img = F.interpolate(cropped_img.unsqueeze(0).unsqueeze(1), size=output_size, mode=mode)
        return downsampled_img.squeeze(0).squeeze(0)

    return cropped_img

def invert_crop_and_downsample(processed_img, original_shape, img_center, crop_size, downsample_size, mode): 
    """
    Inverts the crop_and_downsample operation.

    This function first upsamples the processed image (if it was downsampled)
    back to the original crop size, then inserts it into a blank image of shape
    `original_shape` at the same location where the crop was originally taken.

    Args:
        processed_img (Tensor): Cropped and possibly downsampled image.
        original_shape (tuple): Shape of the original image (e.g., (D, H, W)).
        img_center (list or tuple): Center coordinates used for cropping.
        crop_size (int): Crop size originally used (assumed even for simplicity).
        downsample_size (int): Downsample size used in the forward operation; if None,
                               then no upsampling is performed.
        mode (str): Interpolation mode used (and reused here for upsampling).

    Returns:
        Tensor: An image of shape `original_shape` with the inverted crop inserted.
    """
    # If the image was downsampled, upsample it back to the original crop size.
    if downsample_size is not None:
        upsampled_img = F.interpolate(
            processed_img.unsqueeze(0).unsqueeze(1),
            size=(crop_size, crop_size, crop_size),
            mode=mode
        )
        upsampled_img = upsampled_img.squeeze(0).squeeze(1)
    else:
        upsampled_img = processed_img

    # Create a blank image of the original shape.
    inverted_img = torch.zeros(original_shape, dtype=upsampled_img.dtype, device=upsampled_img.device)

    # Determine the start and end indices for the crop along each axis.
    start_x = img_center[0] - crop_size // 2
    end_x   = img_center[0] + crop_size // 2
    start_y = img_center[1] - crop_size // 2
    end_y   = img_center[1] + crop_size // 2
    start_z = img_center[2] - crop_size // 2
    end_z   = img_center[2] + crop_size // 2

    # Insert the upsampled crop back into the blank image.
    inverted_img[start_x:end_x, start_y:end_y, start_z:end_z] = upsampled_img

    return inverted_img

def compute_derived_params(D, rho, T):
    """
    Compute derived parameters for tumor growth dynamics.

    Args:
        D (float): Diffusion coefficient.
        rho (float): Growth rate.
        T (float): Simulation time.

    Returns:
        tuple: Square root of scaled diffusion (muD) and growth rate (muRho).
    """
    muD = np.sqrt(D * T).astype(np.float32)
    muRho = np.sqrt(rho * T).astype(np.float32)
    return muD, muRho

def compute_center_and_displacement(tumorImg, params = {}):
    """
    Compute the center of mass and displacement of the tumor.

    Args:
        tumorImg (ndarray): Tumor simulation image.
        params (dict): Tumor growth parameters.

    Returns:
        tuple: (Image center, center of mass, displacement values in x, y, z).
    """
    image_center = img_center = [tumorImg.shape[0] // 2, tumorImg.shape[1] // 2, tumorImg.shape[2] // 2]
    com = scipy.ndimage.center_of_mass(tumorImg)
    com = [int(i) for i in com]
    if params == {}:

        return image_center, com


    NxT1_pct, NyT1_pct, NzT1_pct = params['NxT1_pct'], params['NyT1_pct'], params['NzT1_pct']
    [icx, icy, icz] = [tumorImg.shape[1] * NxT1_pct, tumorImg.shape[0] * NyT1_pct, tumorImg.shape[-1] * NzT1_pct]
    
    x = (icx - com[0]) / tumorImg.shape[0]
    y = (icy - com[1]) / tumorImg.shape[1]
    z = (icz - com[2]) / tumorImg.shape[2]
    
    return img_center, com, x, y, z

def shift_images(atlasImg, tumorImg, img_center, com):
    """
    Shift images to align the tumor center with the image center.

    Args:
        atlasImg (ndarray): Tissue atlas image.
        tumorImg (ndarray): Tumor simulation image.
        img_center (list): Center of the image.
        com (list): Center of mass of the tumor.

    Returns:
        tuple: (Shifted atlas image, shifted tumor image).
    """
    rollX = img_center[0] - com[0]
    rollY = img_center[1] - com[1]
    rollZ = img_center[2] - com[2]
    shifted_atlasImg = torch.tensor(atlasImg).roll(shifts=(rollX, rollY, rollZ), dims=(0, 1, 2))
    shifted_tumorImg = torch.tensor(tumorImg).roll(shifts=(rollX, rollY, rollZ), dims=(0, 1, 2))
    return shifted_atlasImg, shifted_tumorImg

def inverse_shift_images(shifted_img, com):
    """
    shift the image back to its original position. Currently it was shifted to the center of the image.
    """
    rollX =  com[0] - shifted_img.shape[0] // 2
    rollY =  com[1] - shifted_img.shape[1] // 2
    rollZ =  com[2] - shifted_img.shape[2] // 2 
    return shifted_img.roll(shifts=(rollX, rollY, rollZ), dims=(0, 1, 2))

def data_preprocess(tumorImg, atlasImg, params_file, crop_sz, downsample_sz):
    """
    Preprocess tumor and atlas images by computing displacements, 
    shifting, cropping, and downsampling.

    Args:
        tumorImg (ndarray): Tumor simulation image.
        atlasImg (ndarray): Tissue atlas image.
        params_file (str): Path to the JSON file containing simulation parameters.
        crop_sz (int): Size for cropping the images.
        downsample_sz (int): Size for downsampling the images.

    Returns:
        tuple: 
            - Downsampled tumor image (Tensor).
            - Downsampled atlas image (Tensor).
            - List of derived parameters for the simulation.
    """
    params, D, rho, T = load_params(params_file)
    muD, muRho = compute_derived_params(D, rho, T)
    img_center, com, x, y, z = compute_center_and_displacement(tumorImg, params)
    shifted_atlasImg, shifted_tumorImg = shift_images(atlasImg, tumorImg, img_center, com)
    downsampled_tumor = crop_and_downsample(shifted_tumorImg, img_center, crop_size=crop_sz, downsample_size=downsample_sz, mode="trilinear")
    downsampled_atlas = crop_and_downsample(shifted_atlasImg, img_center, crop_size=crop_sz, downsample_size=downsample_sz, mode="nearest")
    param_list = [x, y, z, muD, muRho]
    return downsampled_tumor, downsampled_atlas, param_list

def invert_data_preprocess(processed_tumor, processed_atlas, param_list, original_shape, img_center, crop_sz, downsample_sz, com):
    """
    Invert the preprocessing operations to restore images to their original space.
    Operations are inverted in reverse order: upsampling -> uncropping -> unshifting.
    Also inverts the parameter preprocessing.

    Args:
        processed_tumor (Tensor): Processed tumor image.
        processed_atlas (Tensor): Processed atlas image.
        param_list (list): List of parameters [x, y, z, muD, muRho].
        original_shape (tuple): Original shape of the images before preprocessing.
        img_center (list): Center coordinates of the original image.
        crop_sz (int): Size used for cropping.
        downsample_sz (int): Size used for downsampling.
        mode (str): Interpolation mode for upsampling.

    Returns:
        tuple: (Restored tumor image, restored atlas image, original parameters dict)
    """
    # 1. First invert the cropping and downsampling

    uncropped_tumor = invert_crop_and_downsample(
        processed_tumor, 
        original_shape,
        img_center,
        crop_sz,
        downsample_sz,
        "trilinear"
    )
    
    uncropped_atlas = invert_crop_and_downsample(
        processed_atlas,
        original_shape,
        img_center,
        crop_sz,
        downsample_sz,
        "nearest"  # Use nearest neighbor for atlas to preserve segmentation
    )
    
    # 3. Invert the shift
    restored_tumor = inverse_shift_images(uncropped_tumor, com)
    restored_atlas = inverse_shift_images(uncropped_atlas, com)

    # 2. Calculate the original center of mass from displacement parameters
    x, y, z, muD, muRho = param_list


    # 4. Invert parameter preprocessing
    # For muD and muRho, we have: mu = sqrt(param * T)
    # So to get original param: param = (mu^2) / T
    T = 100  # If T is variable, it should be passed as a parameter
    D = (muD * muD) / T
    rho = (muRho * muRho) / T

    # To get back to NxT1_pct format:
    # Original: x = (icx - com[0]) / shape[0] where icx = shape[1] * NxT1_pct
    # So: NxT1_pct = (com[0] + x * shape[0]) / shape[1]
    newX = float((com[0] + x * restored_tumor.shape[0]) / restored_tumor.shape[1])
    newY = float((com[1] + y * restored_tumor.shape[1]) / restored_tumor.shape[0])
    newZ = float((com[2] + z * restored_tumor.shape[2]) / restored_tumor.shape[2])

    # Original parameters dictionary
    original_params_recovered = {
        "Dw": float(D),
        "rho": float(rho),
        "NxT1_pct": newX,  # Convert displacement back to original percentage
        "NyT1_pct": newY,
        "NzT1_pct": newZ,
        
    }

    return restored_tumor, restored_atlas, original_params_recovered

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, length, crop_size, down_sample_size):
        super().__init__()
        
        self.datasetPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/synthetic_FK_Michals_solver_smaller"
        self.atlas_image_path = "/home/jonas/workspace/programs/neural-surrogate-in-med/data/sub-mni152_tissues_space-sri.nii.gz"

        self.patientList = np.sort(os.listdir(self.datasetPath)).tolist()

        if len(self.patientList) < length:
            self.length = len(self.patientList)
        else:
            self.length = length

        self.crop_size = crop_size
        self.down_sample_size = down_sample_size
    

    def __len__(self):
        return self.length
    
    def getFileName(self, idx):
        return self.patientList[idx]

    def get_original_image(self, idx):

        tumor_image_path = os.path.join(self.datasetPath, self.patientList[idx], "tumor_concentration.nii.gz")
        tumorImg = nib.load(tumor_image_path).get_fdata()
        return torch.tensor(tumorImg).float().unsqueeze(0)

    def get_parameters_from_pedicted_parameters_for_patient(self, idx, predicted_parameters = []):

        downsampled_tumor_original, downsampled_atlas, param_list_dataset = self.__getitem__(idx)

        if predicted_parameters == []:
            predicted_parameters = param_list_dataset
            
        orginalTumor = self.get_original_image(idx)
        image_center, com= compute_center_and_displacement(orginalTumor[0])

        inversely_shifted, restored_atlas, params_recovered  = invert_data_preprocess(
            downsampled_tumor_original[0], 
            downsampled_atlas[0], 
            predicted_parameters, 
            orginalTumor[0].numpy().shape,
            image_center,
            self.crop_size,
            self.down_sample_size,
            com
        )
        return params_recovered
    
    def __getitem__(self, idx):
        
        params_file_path = self.datasetPath + "/" + self.patientList[idx] + "/saveDict.json"
        tumor_image_path = self.datasetPath + "/" + self.patientList[idx] + "/tumor_concentration.nii.gz" 

        tumorImg = nib.load(tumor_image_path).get_fdata()
        atlasImg = nib.load(self.atlas_image_path).get_fdata()

        tumorImg[atlasImg < 1.5] = 0

        params, D, rho, T = load_params(params_file_path)
        muD, muRho = compute_derived_params(D, rho, T)
        img_center, com, x, y, z = compute_center_and_displacement(tumorImg, params)
        shifted_atlasImg, shifted_tumorImg = shift_images(atlasImg, tumorImg, img_center, com)
        downsampled_tumor = crop_and_downsample(shifted_tumorImg, img_center, crop_size=self.crop_size,         downsample_size=self.down_sample_size, mode = "trilinear")
        downsampled_atlas = crop_and_downsample(shifted_atlasImg, img_center, crop_size=self.crop_size, downsample_size=self.down_sample_size, mode = "nearest")
        coeff_list = [x, y, z, muD, muRho]

        # this is done to correct for numerical differences from downsampling
        # so no tumor is in CSF or outside the brain
        downsampled_tumor[downsampled_atlas < 1.5] = 0

        return torch.tensor(downsampled_tumor).float().unsqueeze(0),  torch.tensor(downsampled_atlas).float().unsqueeze(0),  torch.tensor(coeff_list).float()


#%%
class brats_lucas_run_with_sbtc(torch.utils.data.Dataset):
    def __init__(self, length, crop_size, down_sample_size):
        super().__init__()
        # all..........................
        self.datasetPath = "/mnt/8tb_slot8/jonas/workingDirDatasets/brats-lucas-results"

        self.patientList = np.sort(os.listdir(self.datasetPath)).tolist()

        if len(self.patientList) < length:
            self.length = len(self.patientList)
        else:
            self.length = length

        self.crop_size = crop_size
        self.down_sample_size = down_sample_size
    

    def __len__(self):
        return self.length
    
    def getFileName(self, idx):
        return self.patientList[idx]

    def get_original_image(self, idx):
        #tumor_image_path = os.path.join(self.datasetPath, self.patientList[idx], "tumor_concentration.nii.gz")
        tumor_image_path = os.path.join(self.datasetPath, self.patientList[idx], "processed/growth_models/sbtc/sbtc_pred.nii.gz")
        tumorImg = nib.load(tumor_image_path).get_fdata()
        return torch.tensor(tumorImg).float().unsqueeze(0)

    def get_parameters_from_pedicted_parameters_for_patient(self, idx, predicted_parameters = []):

        downsampled_tumor_original, downsampled_atlas, param_list_dataset = self.__getitem__(idx)

        if predicted_parameters == []:
            predicted_parameters = param_list_dataset
            
        orginalTumor = self.get_original_image(idx)
        image_center, com= compute_center_and_displacement(orginalTumor[0])

        print("com of patient: ", com, idx)

        inversely_shifted, restored_atlas, params_recovered  = invert_data_preprocess(
            downsampled_tumor_original[0], 
            downsampled_atlas[0], 
            predicted_parameters, 
            orginalTumor[0].numpy().shape,
            image_center,
            self.crop_size,
            self.down_sample_size,
            com
        )
        return params_recovered
    
    
    def __getitem__(self, idx):
        
        #params_file_path = self.datasetPath + "/" + self.patientList[idx] + "/saveDict.json"
        #/mnt/8tb_slot8/jonas/workingDirDatasets/brats-lucas-results/BraTS2021_00000/processed/growth_models/sbtc
        tumor_image_path = self.datasetPath + "/" + self.patientList[idx] + "/processed/growth_models/sbtc/sbtc_pred.nii.gz" 
        atlas_image_path = self.datasetPath + "/" + self.patientList[idx] + "/processed/tissue_segmentation/tissue_seg.nii.gz"

        tumorImg = nib.load(tumor_image_path).get_fdata()
        atlasImg = nib.load(atlas_image_path).get_fdata()

        tumorImg[atlasImg < 1.5] = 0


        img_center, com = compute_center_and_displacement(tumorImg)
        shifted_atlasImg, shifted_tumorImg = shift_images(atlasImg, tumorImg, img_center, com)
        downsampled_tumor = crop_and_downsample(shifted_tumorImg, img_center, crop_size=self.crop_size,         downsample_size=self.down_sample_size, mode = "trilinear")
        downsampled_atlas = crop_and_downsample(shifted_atlasImg, img_center, crop_size=self.crop_size, downsample_size=self.down_sample_size, mode = "nearest")
        
        coeff_list = [0,0,0,1,1]

        # this is done to correct for numerical differences from downsampling
        # so no tumor is in CSF or outside the brain
        downsampled_tumor[downsampled_atlas < 1.5] = 0

        return torch.tensor(downsampled_tumor).float().unsqueeze(0),  torch.tensor(downsampled_atlas).float().unsqueeze(0),  torch.tensor(coeff_list).float()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = brats_lucas_run_with_sbtc(length=1, crop_size=128, down_sample_size=128)
    downsampled_tumor, downsampled_atlas, param_list = dataset[0]
    
    plt.imshow(downsampled_atlas[0, :, :, downsampled_tumor.shape[2] // 2].numpy(), cmap="gray")
    plt.imshow(downsampled_tumor[0, :, :, downsampled_tumor.shape[2] // 2].numpy(), cmap="Reds", alpha=0.5)


# %% check
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #it is cropped first and downsampled second
    dataset = SyntheticDataset(length=1, crop_size=128, down_sample_size=200)
    downsampled_tumor, downsampled_atlas, param_list = dataset[0]

    # Plot the downsampled tumor image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(downsampled_tumor[:,:,downsampled_tumor.shape[2] // 2].numpy()[0], cmap="gray")
    plt.title("Downsampled Tumor Image")
    plt.axis("off")

    # Plot the downsampled atlas image
    plt.subplot(1, 2, 2)
    plt.imshow(downsampled_atlas[:,:,downsampled_atlas.shape[2] // 2].numpy()[0], cmap="gray")
    plt.title("Downsampled Atlas Image")
    plt.axis("off")

    plt.show()
    # invert

#%% dataset stats 
if __name__ == "__main__":
    dataset = SyntheticDataset(length=100, crop_size=128, down_sample_size=128)

    params = []
    for i in range(len(dataset)):
        downsampled_tumor, downsampled_atlas, param_list = dataset[i]
        params.append(param_list)
    
    params = np.array(params)

    #%%
    
    plt.plot(np.mean(params[:,:3], axis=0), label="mean")
    plt.plot(np.min(params[:,:3], axis=0), label="min")
    plt.plot(np.max(params[:,:3], axis=0), label="max")
    plt.title("Parameter Statistics")
    plt.xlabel("Parameter Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    plt.plot(np.mean(params[:,3:], axis=0), label="mean")
    plt.plot(np.min(params[:,3:], axis=0), label="min")
    plt.plot(np.max(params[:,3:], axis=0), label="max")
    plt.title("Parameter Statistics")
    plt.xlabel("Parameter Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    print("Mean parameters: ", np.mean(params, axis=0))
    print("Min parameters: ", np.min(params, axis=0))
    print("Max parameters: ", np.max(params, axis=0))

    #%% scatter mu and rho
    import matplotlib.pyplot as plt
    plt.scatter(params[:,3], params[:,4])
    plt.xlabel("muD")
    plt.ylabel("muRho")


#%% test inversion
if  __name__ == "__main__":
    #it is cropped first and downsampled second
    dataset = SyntheticDataset(length=1, crop_size=128, down_sample_size=128)

    patientNum = 1

    downsampled_tumor_dataset, _, param_list_dataset = dataset[patientNum]

    orginalTumor = nib.load(dataset.datasetPath + "/" + dataset.patientList[patientNum] + "/tumor_concentration.nii.gz").get_fdata()
    original_shape = orginalTumor.shape

    plt.imshow(orginalTumor[:,:,100], cmap="gray")
    plt.show()

    params_file_path = dataset.datasetPath + "/" + dataset.patientList[patientNum] + "/saveDict.json"

    params, D, rho, T = load_params(params_file_path)
    muD, muRho = compute_derived_params(D, rho, T)
    tumorImg = nib.load(dataset.datasetPath + "/" + dataset.patientList[patientNum] + "/tumor_concentration.nii.gz").get_fdata()
    img_center, com, x, y, z = compute_center_and_displacement(tumorImg, params)
    
    atlasImg = nib.load(dataset.atlas_image_path).get_fdata()
    shifted_atlasImg, shifted_tumorImg = shift_images(atlasImg, tumorImg, img_center, com)

    downsampled_tumor = crop_and_downsample(shifted_tumorImg, img_center, crop_size=dataset.crop_size,         downsample_size=dataset.down_sample_size, mode = "trilinear")
    downsampled_tumor_and_croped = downsampled_tumor.clone()

    downsampled_atlas = crop_and_downsample(shifted_atlasImg, img_center, crop_size=dataset.crop_size, downsample_size=dataset.down_sample_size, mode = "nearest")
    coeff_list = [x, y, z, muD, muRho]

    downsampled_tumor[downsampled_atlas < 1.5] = 0
    # Plot the downsampled tumor image
    plt.imshow(downsampled_tumor[:,:,downsampled_tumor.shape[2] // 2].numpy(), cmap="gray")

    plt.title("Downsampled Tumor Image")
    plt.show()
    # Plot the differences to dataset
    diff = downsampled_tumor - downsampled_tumor_dataset
    plt.imshow(diff[:,:,downsampled_tumor.shape[2] // 2].numpy()[0], cmap="bwr", vmin=-0.1, vmax=0.1)
    plt.title("Difference to Dataset")
    plt.colorbar()


    # inverted image form the function
    downsampled_tumor_dataset, _, param_list_dataset = dataset[patientNum]
    image_center = compute_center_and_displacement(orginalTumor, params)[0]

    inversely_shifted, restored_atlas, original_params_recovered  = invert_data_preprocess(
        downsampled_tumor_dataset[0], 
        downsampled_atlas, 
        param_list_dataset, 
        original_shape,
        image_center,
        dataset.crop_size,
        dataset.down_sample_size,
        com
    )
        #plt.imshow(inversely_shifted[:,:,inversely_shifted.shape[2] // 2].numpy(), cmap="gray")
    diff = inversely_shifted - orginalTumor
    plt.show()
    plt.imshow(diff[:,:,100].numpy(), cmap="bwr", vmin=-1, vmax=1)  

    plt.title("Difference to Original Tumor")     

    #%% print origianal params and recovered params
    print("original params: ", params)
    print("recovered params: ", original_params_recovered)

    # run the simulation again with the recovered params

    #%% and compare the images

    def runForwardSimulation(params):
        from TumorGrowthToolkit.FK import Solver as FKSolver
    
        atlasPath = "/home/jonas/workspace/programs/neural-surrogate-in-med/data/sub-mni152_tissues_space-sri.nii.gz"
        atlasTissue = nib.load(atlasPath).get_fdata()
        affine = nib.load(atlasPath).affine

        wm_data = atlasTissue == 3
        gm_data = atlasTissue == 2

        # Set up parameters
        parameters = {
            'Dw': params["Dw"],          # Diffusion coefficient for white matter
            'rho':  params["rho"],        # Growth rate
            'RatioDw_Dg': 10,  # Ratio of diffusion coefficients in white and grey matter
            'gm': gm_data,      # Grey matter data
            'wm': wm_data,      # White matter data
            'NxT1_pct': params["NxT1_pct"],    # tumor position [%]
            'NyT1_pct': params["NyT1_pct"],
            'NzT1_pct': params["NzT1_pct"],
            'init_scale': 1., #scale of the initial gaussian
            'resolution_factor': 1, #resultion scaling for calculations
            'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
            'verbose': True, #printing timesteps 
            'time_series_solution_Nt': None,#64, #64, # number of timesteps in the output
            #'stopping_volume' : stopping_volume, #stop when the volume of the tumor is less than this value
            'stopping_time' : 100, #stop when the time is greater than this values
        }        

        #Run the FK_solver and plot the results
        import time
        start_time = time.time()
        fk_solver = FKSolver(parameters)
        result = fk_solver.solve()
        end_time = time.time()  # Store the end time
        execution_time = int(end_time - start_time)  # Calculate the difference
        print(f"Execution time: {execution_time} seconds")
        # close all figures
        plt.close("all")
        return result["final_state"]
    
    reSim = runForwardSimulation(original_params_recovered)


    diff = reSim - orginalTumor
    plt.title("Difference to Original Tumor")
    plt.imshow(diff[:,:,reSim.shape[2] // 2], cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    maxDiff = np.max(np.abs(diff))
    print("maxDiff: ", maxDiff)
    # %%
    a= 1
    # %%
