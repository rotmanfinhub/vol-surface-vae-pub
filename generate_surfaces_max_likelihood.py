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


def generate_surfaces_mle(model: CVAEMemRand, ex_data, vol_surface_data, day, ctx_len, num_vaes, use_ex_feats, check_ex_feats):
    torch.cuda.empty_cache()
    z = torch.zeros((num_vaes, ctx_len + 1, model.config["latent_dim"]))
    vae_surfaces = np.zeros((num_vaes, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    if use_ex_feats and check_ex_feats:
        vae_ex_feats = np.zeros((num_vaes, ex_data.shape[1]))
    else:
        vae_ex_feats = None

    surf_data = torch.from_numpy(vol_surface_data[day - ctx_len:day])
    if use_ex_feats:
        ex_data = torch.from_numpy(ex_data[day - ctx_len:day])
        if len(ex_data.shape) == 1:
            ex_data = ex_data.unsqueeze(1)
        ctx_data = {
            "surface": surf_data.unsqueeze(0).repeat(num_vaes, 1, 1, 1), # (T, 5, 5)
            "ex_feats": ex_data.unsqueeze(0).repeat(num_vaes, 1, 1) # (T, 3)
        }
    else:
        ctx_data = {
            "surface": surf_data.unsqueeze(0).repeat(num_vaes, 1, 1, 1), # (T, 5, 5)
        }

    # for i in range(num_vaes):
    if use_ex_feats:
        surf, ex_feat = model.get_surface_given_conditions(ctx_data, z=z)
        ex_feat = ex_feat.detach().cpu().numpy().reshape((ex_data.shape[1],))
    else:
        surf = model.get_surface_given_conditions(ctx_data, z=z)
    surf = surf.detach().cpu().numpy().reshape((vol_surface_data.shape[1], vol_surface_data.shape[2]))
    vae_surfaces[...] = surf
    if use_ex_feats and check_ex_feats:
        vae_ex_feats[...] = ex_feat
    
    return vae_surfaces, vae_ex_feats

def generate_surfaces_multiday_mle(model_data, ex_data, vol_surface_data, 
                                start_day, days_to_generate, num_vaes,
                                model_type: Union[CVAE, CVAEMem, CVAEMemRand] = CVAEMemRand,
                                check_ex_feats=False, ctx_len=None):
    model_config = model_data["model_config"]
    model = model_type(model_config)
    model.load_weights(dict_to_load=model_data)
    if ctx_len is None:
        seq_len = model_config["seq_len"]
        ctx_len = model_config["ctx_len"]

    if "ex_feats_dim" in model_config:
        use_ex_feats = model_config["ex_feats_dim"] > 0
    else:
        use_ex_feats = False
    print("use_ex_feats is: ",use_ex_feats)

    all_day_surfaces = np.zeros((days_to_generate, num_vaes, vol_surface_data.shape[1], vol_surface_data.shape[2]))
    all_day_ex_feats = np.zeros((days_to_generate, num_vaes, ex_data.shape[1]))

    for day in range(start_day, start_day+days_to_generate):
        if day % 500 == 0:
            print(f"Generating day {day}")
        vae_surfaces, vae_ex_feats = generate_surfaces_mle(model, ex_data, vol_surface_data, day, ctx_len, num_vaes, use_ex_feats, check_ex_feats)
        all_day_surfaces[day - start_day, ...] = vae_surfaces
        if vae_ex_feats is not None:
            all_day_ex_feats[day - start_day, ...] = vae_ex_feats
    
    return all_day_surfaces, all_day_ex_feats

set_seeds(0)
torch.set_default_dtype(torch.float64)
num_epochs = 500
ctx_len = 5
start_day = 5
days_to_generate = 5810
num_vaes = 1

data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
level_data = data["levels"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[...,np.newaxis], skew_data[...,np.newaxis], slope_data[...,np.newaxis]], axis=-1)

base_folder = "test_spx/2024_11_09"
for (file_path, use_ex, return_ex) in [
    (f"{base_folder}/no_ex.pt", False, False),
    (f"{base_folder}/ex_no_loss.pt", True, False),
    (f"{base_folder}/ex_loss.pt", True, True),
]:
    print(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    model_data = torch.load(file_path) # latent_dim=5, surface_hidden=[5,5,5], mem_hidden=100
    model_config = model_data["model_config"]
    model_config["mem_dropout"] = 0.
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)
    model.eval()
    print(model)

    i = ctx_len
    gen_fn = f"{base_folder}/{file_name}_mle_gen{i}.npz"
    if not os.path.exists(gen_fn):
        if return_ex:
            surfaces, ex_feats = generate_surfaces_multiday_mle(model_data = model_data,
                    ex_data = ex_data, vol_surface_data = vol_surf_data,
                    start_day=i, days_to_generate=days_to_generate, num_vaes = num_vaes,
                    model_type = CVAEMemRand, check_ex_feats=return_ex, ctx_len=i)
            np.savez(gen_fn, surfaces=surfaces, ex_feats=ex_feats)
        else:
            
            surfaces, _ = generate_surfaces_multiday_mle(model_data = model_data,
                    ex_data = ex_data, vol_surface_data = vol_surf_data,
                    start_day=i, days_to_generate=days_to_generate, num_vaes = num_vaes,
                    model_type = CVAEMemRand, check_ex_feats=return_ex, ctx_len=i)
            np.savez(gen_fn, surfaces=surfaces)
