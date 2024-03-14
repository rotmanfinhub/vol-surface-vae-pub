import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import *
from eval_scripts.eval_utils import *
import os, sys

set_seeds(0)
torch.set_default_dtype(torch.float64)
num_epochs = 500
ctx_len = 3
start_day = 3
days_to_generate = 5300
num_vaes = 1000

data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
level_data = data["levels"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[...,np.newaxis], skew_data[...,np.newaxis], slope_data[...,np.newaxis]], axis=-1)
print(ex_data.shape)
train_dataset = VolSurfaceDataSetRand(vol_surf_data[:4000])
valid_dataset = VolSurfaceDataSetRand(vol_surf_data[4000:5000])
test_dataset = VolSurfaceDataSetRand(vol_surf_data[5000:])
train_batch_sampler = CustomBatchSampler(train_dataset, 64)
valid_batch_sampler = CustomBatchSampler(valid_dataset, 16)
test_batch_sampler = CustomBatchSampler(test_dataset, 16)
train_simple = DataLoader(train_dataset, batch_sampler=train_batch_sampler)
valid_simple = DataLoader(valid_dataset, batch_sampler=valid_batch_sampler)
test_simple = DataLoader(test_dataset, batch_sampler=test_batch_sampler)

train_dataset2 = VolSurfaceDataSetRand((vol_surf_data[:4000], ex_data[:4000]))
valid_dataset2 = VolSurfaceDataSetRand((vol_surf_data[4000:5000], ex_data[4000:5000]))
test_dataset2 = VolSurfaceDataSetRand((vol_surf_data[5000:], ex_data[5000:]))
train_batch_sampler2 = CustomBatchSampler(train_dataset2, 64)
valid_batch_sampler2 = CustomBatchSampler(valid_dataset2, 16)
test_batch_sampler2 = CustomBatchSampler(test_dataset2, 16)
train_ex = DataLoader(train_dataset2, batch_sampler=train_batch_sampler2)
valid_ex = DataLoader(valid_dataset2, batch_sampler=valid_batch_sampler2)
test_ex = DataLoader(test_dataset2, batch_sampler=test_batch_sampler2)

if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = "test_spx/ex_3_feats/conv_mem_spx_ex3_0.pt"
file_name = os.path.splitext(os.path.basename(file_path))[0]

model_data = torch.load(file_path) # latent_dim=5, surface_hidden=[5,5,5], mem_hidden=100
model_config = model_data["model_config"]
model = CVAEMemRand(model_config)
model.load_weights(dict_to_load=model_data)
print(model)

if model_config["re_feat_weight"] > 0:
    check_ex_feats = True
else:
    check_ex_feats = False
print("check_ex_feats:", check_ex_feats)
if check_ex_feats:
    for i in range(ctx_len, ctx_len+1):
        set_seeds(0)
        print(f"currently generating {i}")
        if not os.path.exists(f"simpaths/{file_name}_gen{i}.npz"):
            os.makedirs("simpaths", exist_ok=True)
            surfaces, ex_feats = generate_surfaces_multiday(model_data = model_data,
                    ex_data = ex_data, vol_surface_data = vol_surf_data,
                    start_day=start_day, days_to_generate=days_to_generate, num_vaes = num_vaes,
                    model_type = CVAEMemRand, check_ex_feats=check_ex_feats, ctx_len=i)
            np.savez(f"simpaths/{file_name}_gen{i}.npz", surfaces=surfaces, ex_feats=ex_feats)
else:
    for i in range(ctx_len, ctx_len+1):
        set_seeds(0)
        print(f"currently generating {i}")
        if not os.path.exists(f"simpaths/{file_name}_gen{i}.npy"):
            os.makedirs("simpaths", exist_ok=True)
            surfaces, _ = generate_surfaces_multiday(model_data = model_data,
                    ex_data = ex_data, vol_surface_data = vol_surf_data,
                    start_day=start_day, days_to_generate=days_to_generate, num_vaes = num_vaes,
                    model_type = CVAEMemRand, check_ex_feats=check_ex_feats, ctx_len=i)
            np.save(f"simpaths/{file_name}_gen{i}.npy", surfaces)