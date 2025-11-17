# my_project/utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def gradient(input_tensor, onlyAssymetric = False):
    if onlyAssymetric:
        gradient_x_minus = (torch.roll(input_tensor, shifts=-1, dims=2) - input_tensor) 
        gradient_y_minus = (torch.roll(input_tensor, shifts=-1, dims=3) - input_tensor)
        gradient_z_minus = (torch.roll(input_tensor, shifts=-1, dims=4) - input_tensor)

        gradient_x_plus = (input_tensor - torch.roll(input_tensor, shifts=1, dims=2))
        gradient_y_plus = (input_tensor - torch.roll(input_tensor, shifts=1, dims=3))
        gradient_z_plus = (input_tensor - torch.roll(input_tensor, shifts=1, dims=4))

        gradient_x = gradient_x_minus.abs() + gradient_x_plus.abs()
        gradient_y = gradient_y_minus.abs() + gradient_y_plus.abs()
        gradient_z = gradient_z_minus.abs() + gradient_z_plus.abs()

        return gradient_x, gradient_y, gradient_z

    else:
        gradient_x = (torch.roll(input_tensor, shifts=-1, dims=2) -  torch.roll(input_tensor, shifts=1, dims=2)) / 2 
        gradient_y = (torch.roll(input_tensor, shifts=-1, dims=3) -  torch.roll(input_tensor, shifts=1, dims=3)) / 2
        gradient_z = (torch.roll(input_tensor, shifts=-1, dims=4) -  torch.roll(input_tensor, shifts=1, dims=4)) / 2

    return gradient_x, gradient_y, gradient_z

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
    
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]

        if input_tensor.max() == 0:
            return 0
        
        gradient_x, gradient_y, gradient_z = gradient(input_tensor, True)

        # Compute per-sample gradient magnitude mean
        loss_per_sample = []
        for i in range(batch_size):
            grad_mag = torch.sqrt(torch.clamp(
                gradient_x[i] ** 2 + gradient_y[i] ** 2 + gradient_z[i] ** 2, 
                min=0.00001, max=1000.0
            ))
            valid_mask = input_tensor[i] > 0.001
            loss_per_sample.append(torch.mean(grad_mag[valid_mask] ** 2))

        loss_per_sample = torch.stack(loss_per_sample)
        
        # Compute batch-wise average
        gradient_magnitude_mean = torch.mean(loss_per_sample)
        

        # Better NaN check
        if torch.isnan(gradient_magnitude_mean):
            raise ValueError("NaN detected in GradientLoss computation!")

        return gradient_magnitude_mean
        

class WaveFrontLossFirstOrder(nn.Module):
    def __init__(self, wm, gm):
        super(WaveFrontLossFirstOrder, self).__init__()
        self.wm = wm
        self.gm = gm

    def gradient_magnitude(self, input_tensor):

        gradient_x, gradient_y, gradient_z = gradient(input_tensor)

        # Compute the magnitude of the gradients
        gradient_magnitude = torch.sqrt(torch.clamp(gradient_x ** 2 + gradient_y ** 2 + gradient_z ** 2, min=0.00001, max=1000.0))

        #check if nan is in tensor
        if torch.isnan(input_tensor).any():
            print("nan in tensor")
            print(input_tensor)
            print(torch.isnan(input_tensor))
            exit()

        return gradient_magnitude

    def forward(self, predictions, constantFactor = 1, returnVoxelwise = False, mask = None):
        # Compute the gradient magnitude of the predictions
        grad_magnitude = self.gradient_magnitude(predictions)

        if mask is None:
            mask = predictions > 0.001
        else:
            mask = mask > 0.5

        # no loss on pixel at the border to the mask
        # Define a 3x3x3 kernel for 3D erosion
        kernel = torch.ones(1, 1, 3, 3, 3).float().to(mask.device)
        padded_mask = F.pad(mask, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        # Apply 3D convolution to perform erosion (with padding=0)

        eroded_mask = F.conv3d(padded_mask.float(), kernel, padding=0)

        eroded_mask = (eroded_mask == 27).float()

        #voxelLoss = predictions * (1-predictions) * constantFactor / D - grad_magnitude  this should be the correct one
        voxelLoss = (predictions * (1-predictions) * constantFactor  - grad_magnitude )/ constantFactor# (grad_magnitude +0.000000000001)  #/constantFactor

        voxelLoss = voxelLoss * eroded_mask
        
        loss = torch.mean(torch.abs(voxelLoss)**2) # 
        if returnVoxelwise:
            return loss, voxelLoss

        return loss
    

class DiceLoss(nn.Module):
    def __init__(self, default_threshold=0.5, epsilon=1e-6, steepness=500):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.default_threshold = default_threshold
        self.steepness = steepness

    def contFunction(self, x, threshold):
        # Use torch.sigmoid rather than F.sigmoid (the latter is deprecated)
        return torch.sigmoid((x - threshold) * self.steepness)

    def forward(self, predictions, targets, threshold=None):
        if threshold is None:
            threshold = self.default_threshold

        # Apply the continuous function to get probabilities
        probs = self.contFunction(predictions, threshold)

        # Compute the soft Dice coefficient per sample in the batch.
        # We assume the tensor shape is (B, ...). The following will sum over all dimensions except the batch.
        dims = tuple(range(1, predictions.ndimension()))
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice_coeff = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        dice_loss = 1.0 - dice_coeff

        # Return the average Dice loss over the batch
        return dice_loss.mean()


class PetLoss(nn.Module):
    def __init__(self):
        super(PetLoss, self).__init__()
    
    def forward(self, input_tensor, pet_image):

        mask = (pet_image > 0.0001 ) 

        if torch.sum(mask) < 0.000000001 or torch.sum(pet_image[mask]) < 0.001:
            return 0

        flatten_input = input_tensor[mask].view(-1)
        flatten_pet = pet_image[mask].view(-1)
        

        flatten_input_mean = flatten_input - torch.mean(flatten_input)
        flatten_pet_mean = flatten_pet - torch.mean(flatten_pet)

        correlation = torch.mean(flatten_input_mean * flatten_pet_mean) / (
            (torch.std(flatten_input) * torch.std(flatten_pet) + 0.001)
        )
    
        return 1-correlation

class MaskedTVLoss3D(nn.Module):
    """
    Masked Total Variation Loss for 3D images.
    
    This loss encourages smoothness only within a specified mask. It computes the
    absolute differences between neighboring voxels in the depth, height, and width
    dimensions, but only for voxel pairs that are both inside the mask.
    
    Args:
        weight (float): Weighting factor for the loss.
        eps (float): A small constant to avoid division by zero.
    """
    def __init__(self, weight=1.0, eps=1e-8):
        super(MaskedTVLoss3D, self).__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input 3D image tensor of shape (N, C, D, H, W).
            mask (torch.Tensor): Binary mask tensor indicating where to compute the loss.
                                 Expected shape is either (N, 1, D, H, W) or (N, D, H, W).
        Returns:
            torch.Tensor: The masked total variation loss.
        """
        # Ensure mask has a channel dimension
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)  # now shape is (N, 1, D, H, W)
        
        # Compute differences along the depth axis
        d_diff = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        d_mask = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
        d_valid = d_mask.sum()
        d_loss = (d_diff * d_mask).sum() / (d_valid + self.eps) if d_valid > 0 else 0.0

        # Compute differences along the height axis
        h_diff = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        h_mask = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
        h_valid = h_mask.sum()
        h_loss = (h_diff * h_mask).sum() / (h_valid + self.eps) if h_valid > 0 else 0.0

        # Compute differences along the width axis
        w_diff = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        w_mask = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]
        w_valid = w_mask.sum()
        w_loss = (w_diff * w_mask).sum() / (w_valid + self.eps) if w_valid > 0 else 0.0

        # Combine the losses from each spatial dimension and apply the weighting factor
        loss = self.weight * (d_loss + h_loss + w_loss)
        return loss