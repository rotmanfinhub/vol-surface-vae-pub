import torch
import numpy as np
from typing import Union
from vae.cvae import CVAE
from vae.cvae_with_mem import CVAEMem
from vae.cvae_with_mem_randomized import CVAEMemRand

def generate_surfaces(model, ex_data, vol_surface_data, day, ctx_len, num_vaes, use_ex_feats, check_ex_feats):
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
            "surface": surf_data,
            "ex_feats": ex_data
        }
    else:
        ctx_data = {
            "surface": surf_data,
        }
    
    for i in range(num_vaes):
        if use_ex_feats:
            surf, ex_feat = model.get_surface_given_conditions(ctx_data)
            ex_feat = ex_feat.detach().cpu().numpy().reshape((ex_data.shape[1],))
        else:
            surf = model.get_surface_given_conditions(ctx_data)
        surf = surf.detach().cpu().numpy().reshape((vol_surface_data.shape[1], vol_surface_data.shape[2]))
        vae_surfaces[i, ...] = surf
        if use_ex_feats and check_ex_feats:
            vae_ex_feats[i, ...] = ex_feat
    
    return vae_surfaces, vae_ex_feats

def generate_surfaces_multiday(model_data, ex_data, vol_surface_data, 
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
        vae_surfaces, vae_ex_feats = generate_surfaces(model, ex_data, vol_surface_data, day, ctx_len, num_vaes, use_ex_feats, check_ex_feats)
        all_day_surfaces[day - start_day, ...] = vae_surfaces
        if vae_ex_feats is not None:
            all_day_ex_feats[day - start_day, ...] = vae_ex_feats
    
    return all_day_surfaces, all_day_ex_feats