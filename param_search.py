import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from vae.datasets_randomized import VolSurfaceDataSetRand, CustomBatchSampler
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import *
import os

set_seeds(0)
torch.set_default_dtype(torch.float64)
num_epochs = 500

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


conv_param_grid = {
    "latent_dim": 5,
    "surface_hidden": [5, 5, 5],
    "mem_hidden": 100,
    "kl_weight": 1e-5,
}

base_folder_name = "test_spx/2024_11_09"

df = {
    "fn": [],
    "latent_dim": [],
    "surface_hidden": [],
    "mem_hidden": [],
    "kl_weight": [],
    "dev_loss": [],
    "dev_re_surface": [],
    "dev_re_ex_feats": [],
    "dev_re_loss": [],
    "dev_kl_loss": [],
    "test_loss": [],
    "test_re_surface": [],
    "test_re_ex_feats": [],
    "test_re_loss": [],
    "test_kl_loss": [],
}

params = [
    {"model_name": "no_ex", "train_data": train_simple, "valid_data": valid_simple, "test_data": test_simple},
    {"model_name": "ex_no_loss", "train_data": train_ex, "valid_data": valid_ex, "test_data": test_ex},
    {"model_name": "ex_loss", "train_data": train_ex, "valid_data": valid_ex, "test_data": test_ex},
]

for i, param in enumerate(params):
    set_seeds(0)
    
    model_name = param["model_name"] + ".pt"
    train_data = param["train_data"]
    valid_data = param["valid_data"]
    test_data = param["test_data"]
    print(model_name)

    config = {
        "feat_dim": (5, 5),
        "latent_dim": conv_param_grid["latent_dim"],
        "device": "cuda",
        "kl_weight": conv_param_grid["kl_weight"],
        "re_feat_weight": 1.0 if param["model_name"] == "ex_loss" else 0.0,
        "surface_hidden": conv_param_grid["surface_hidden"],
        "ex_feats_dim": 0 if param["model_name"] == "no_ex" else ex_data.shape[-1],
        "ex_feats_hidden": None,
        "mem_type": "lstm",
        "mem_hidden": conv_param_grid["mem_hidden"],
        "mem_layers": 2,
        "mem_dropout": 0.2,
        "ctx_surface_hidden": conv_param_grid["surface_hidden"],
        "ctx_ex_feats_hidden": None,
        "interaction_layers": None,
        "use_dense_surface": False,
        "compress_context": True,
        "ex_loss_on_ret_only": True,  # assume that the ret is the first feature in the tensor
        "ex_feats_loss_type": "l2",
    }
    
    model = CVAEMemRand(config)
    if not os.path.exists(f"{base_folder_name}/{model_name}"):
        train(model, train_data, valid_data, epochs=num_epochs, lr=1e-05, model_dir=base_folder_name, file_name=model_name)
    dev_losses, test_losses = test(model, valid_data, test_data, f"{base_folder_name}/{model_name}")
    df["fn"].append(model_name)
    df["latent_dim"].append(conv_param_grid["latent_dim"])
    df["surface_hidden"].append(conv_param_grid["surface_hidden"])
    df["mem_hidden"].append(conv_param_grid["mem_hidden"])
    df["kl_weight"].append(conv_param_grid["kl_weight"])
    df["dev_loss"].append(dev_losses["loss"])
    df["dev_re_surface"].append(dev_losses["re_surface"])
    df["dev_re_ex_feats"].append(dev_losses["re_ex_feats"])
    df["dev_re_loss"].append(dev_losses["reconstruction_loss"])
    df["dev_kl_loss"].append(dev_losses["kl_loss"])
    df["test_loss"].append(test_losses["loss"])
    df["test_re_surface"].append(test_losses["re_surface"])
    df["test_re_ex_feats"].append(test_losses["re_ex_feats"])
    df["test_re_loss"].append(test_losses["reconstruction_loss"])
    df["test_kl_loss"].append(test_losses["kl_loss"])

df = pd.DataFrame(df)
df.to_csv(f"{base_folder_name}/results.csv", index=False)