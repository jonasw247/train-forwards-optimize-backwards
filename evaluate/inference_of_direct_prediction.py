#%%

import sys
sys.path.append('/home/jonas/workspace/programs/neural-surrogate-in-med/')
import torch
from torch.utils.data import DataLoader
import numpy as np
from utils.datasets import SyntheticDataset
from model.directInverseConvNext import ConvNextEncoderForCoeffs

#def main():
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# model setup (match training config)
model = ConvNextEncoderForCoeffs(
    in_channels=2,
    num_coeffs=5,
    n_spatial_dims=3,
    spatial_resolution=(128, 128, 128),
    stages=4,
    blocks_per_stage=1,
    blocks_at_neck=1,
    init_features=32,
    gradient_checkpointing=False
).to(device)

# load pretrained weights
#modelName = "model_xindi-maquis-4" # this is the first test... not so good...
#ckpt_path = "/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-direct-inverse/checkpoint/"+modelName+"/model_epoch_3.pt"

modelName = "model_rural-wood-5"
ckpt_path = "/mnt/8tb_slot8/jonas/checkpoints/neural-surrogate-direct-inverse/checkpoint/"+modelName+"/model_epoch_5.pt"
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

# prepare test set
ds = SyntheticDataset(length=30000, crop_size=128, down_sample_size=128)
test_indices = range(len(ds) - 1000, len(ds))
test_loader = DataLoader(
    torch.utils.data.Subset(ds, test_indices),
    batch_size=1, shuffle=False, num_workers=2, pin_memory=True
)

mse_loss = torch.nn.MSELoss()
total_loss = 0.0
preds, trues = [], []
testIdx, fullDatasetIdx = [], []

with torch.no_grad():
    for i, (down_tumor, down_atlas, coeff) in enumerate(test_loader):
        index = test_loader.dataset.indices[i]
        testIdx.append(i)
        fullDatasetIdx.append(index)
        down_tumor = down_tumor.to(device)
        down_atlas = down_atlas.to(device)
        coeff = coeff.to(device)
        inputs = torch.cat([down_tumor, down_atlas], dim=1)
        output = model(inputs)

        total_loss += mse_loss(output, coeff).item()
        preds.append(output.cpu().numpy())
        trues.append(coeff.cpu().numpy())
        #if i> 100:
        #    print(f"Processed {i} samples")
        #    break

preds = np.vstack(preds)
trues = np.vstack(trues)
print(f"Test MSE: {total_loss / len(test_loader):.6f}")

#%%
savePath = "/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/" + modelName
os.makedirs(savePath, exist_ok=True)
# save predictions and ground truth
results = {
    "preds": preds,
    "trues": trues,
    "testIdx": testIdx,
    "fullDatasetIdx": fullDatasetIdx,
    "labels": ["x", "y", "z", "muD", "muRho"],
}
np.save(savePath + "/test_results.npy", results)



# %%
# load /mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_xindi-maquis-4

a = np.load("/mnt/8tb_slot8/jonas/workingDirDatasets/neural-surrogate-in-med/synthetic_FK_Michals_solver_smaller/direct_prediction/model_xindi-maquis-4/test_results.npy", allow_pickle=True).item()
# %%
a["testIdx"]

# %%
