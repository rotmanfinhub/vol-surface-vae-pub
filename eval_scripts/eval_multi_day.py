import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
from vae.cvae import CVAE
from vae.cvae_with_mem import CVAEMem
from vae.cvae_with_mem_randomized import CVAEMemRand
from eval_scripts.eval_utils import *

def generate_multiple_surfaces_day_evolution(model_data, ex_data, vol_surface_data, 
                                             start_day, days_to_generate, row, col, num_vaes,
                                             model_type: Union[CVAE, CVAEMem, CVAEMemRand] = CVAEMemRand, 
                                             check_ex_feats=False, bin_count=15, ctx_len=None):
    '''
    Input:
    - model: trained model for vae vol surface
    - ex_data, vol_surface_data: pre-generated latent samples return/price and surfaces data. If SABR data has multiple paths, need to select the path before passing in.
    - start_day: MUST start_day > ctx_len
    - days_to_generate: how many days we want to check the evolution, e.g. 30 day for SABR
    - num_vaes: number of vaes of any choosing
    - row, col: point on vae surfaces wish to view. 0<=row<num_ttm, 0<=col<num_moneyness
    - check_ex_feats: whether or not we check the distribution of extra features like return/price, default False,  if True, will plot 2 separate distributions
    - bin_count: number of bins to characterize the histogram, default: 15

    Output:
    - z_score_list: list of z-scores for a realized point vs the distribution of VAEs on that day for 30 days
    - quartile_list: return distribution of quartile ranges the realized point s found on the day
    --plots the data as well
    '''
    model_config = model_data["model_config"]
    model = model_type(model_config)
    model.load_weights(dict_to_load=model_data)
    if ctx_len is None:
        seq_len = model_config["seq_len"]
        ctx_len = model_config["ctx_len"]
    moneyness_grid=[0.7, 0.85, 1, 1.15, 1.3]
    ttm_grid=[0.08333, 0.25, 0.5, 1, 2]

    vae_surface_z_scores = []
    vae_surface_quartiles = []
    vae_ex_feat_z_scores = []
    vae_ex_feat_quartiles = []
    if "ex_feats_dim" in model_config:
        use_ex_feats = model_config["ex_feats_dim"] > 0
    else:
        use_ex_feats = False
    print("use_ex_feats is: ",use_ex_feats)

    quartile_labels = ["Q1:[0,0.25)", "Q2:[0.25,0.5)", "Q3:[0.5,0.75)", "Q4:(0.75,1]"]
    for day in range(start_day, start_day+days_to_generate):
        vae_surfaces, vae_ex_feats = generate_surfaces(model, ex_data, vol_surface_data, day, ctx_len, num_vaes, use_ex_feats, check_ex_feats)

        realized_value = vol_surface_data[day, row, col]
        vae_values = vae_surfaces[:, row, col]

        z_score = (realized_value - np.mean(vae_values)) / np.std(vae_values)
        quartiles = np.percentile(vae_values, [25, 50, 75])
        quartile_idx = np.searchsorted(quartiles, realized_value)
        quartile_region = quartile_labels[quartile_idx]
        vae_surface_z_scores.append(z_score)
        vae_surface_quartiles.append(quartile_region)

        if use_ex_feats and check_ex_feats:
            realized_ex_feats = ex_data[day]
            z_score = (realized_ex_feats - np.mean(vae_ex_feats)) / np.std(vae_ex_feats)
            quartiles = np.percentile(vae_ex_feats, [25, 50, 75])
            quartile_idx = np.searchsorted(quartiles, realized_ex_feats)
            quartile_region = quartile_labels[quartile_idx]
            vae_ex_feat_z_scores.append(z_score)
            vae_ex_feat_quartiles.append(quartile_region)
    
    if use_ex_feats and check_ex_feats:
        plot_args = [{
            "values": pd.DataFrame({x: [vae_surface_quartiles.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
            "xlabel": "Quartile Range",
            "ylabel": "Frequency",
            "title": f"Histogram of where Realized Point lies in daily VAE distribution for {days_to_generate} Days: \nVAE Point (Moneyness={moneyness_grid[col]}, TTM={ttm_grid[row]})",
        }, {
            "values": vae_surface_z_scores,
            "xlabel": "Z-score",
            "ylabel": "Frequency",
            "title": f"Z-score of Realized Point distribution for {days_to_generate} Days: \nVAE Point (Moneyness={moneyness_grid[col]}, TTM={ttm_grid[row]})",
            "bins": bin_count,
        }, {
            "values": pd.DataFrame({x: [vae_ex_feat_quartiles.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
            "xlabel": "Quartile Range",
            "ylabel": "Frequency",
            "title": f"Histogram of where Realized Point lies in daily VAE distribution for {days_to_generate} Days: \nEx feature",
        }, {
            "values": vae_ex_feat_z_scores,
            "xlabel": "Z-score",
            "ylabel": "Frequency",
            "title": f"Z-score of Realized Point distribution for {days_to_generate} Days: \nEx feature",
            "bins": bin_count,
        }]
        fig, ax = plt.subplots(2, 2, figsize=(11, 12))
        for i in range(2):
            for j in range(2):
                curr_ax = ax[i][j]
                plot_arg = plot_args[2*i + j]
                if "bins" not in plot_arg:
                    sns.barplot(plot_arg["values"], x=plot_arg["xlabel"], y=plot_arg["ylabel"], ax=curr_ax)
                else:
                    sns.histplot(plot_arg["values"], bins=plot_arg["bins"], ax=curr_ax)
                    curr_ax.set_xlabel(plot_arg["xlabel"])
                    curr_ax.set_ylabel(plot_arg["ylabel"])
                curr_ax.set_title(plot_arg["title"])
        plt.tight_layout()
        plt.show()
        return vae_surface_quartiles, vae_surface_z_scores, vae_ex_feat_quartiles, vae_ex_feat_z_scores
    else:
        plot_args = [{
            "values": pd.DataFrame({x: [vae_surface_quartiles.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
            "xlabel": "Quartile Range",
            "ylabel": "Frequency",
            "title": f"Histogram of where Realized Point lies in daily VAE distribution for {days_to_generate} Days: \nVAE Point (Moneyness={moneyness_grid[col]}, TTM={ttm_grid[row]})",
        }, {
            "values": vae_surface_z_scores,
            "xlabel": "Z-score",
            "ylabel": "Frequency",
            "title": f"Z-score of Realized Point distribution for {days_to_generate} Days: \nVAE Point (Moneyness={moneyness_grid[col]}, TTM={ttm_grid[row]})",
            "bins": bin_count,
        }]
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        for i in range(2):
            curr_ax = ax[i]
            plot_arg = plot_args[i]
            if "bins" not in plot_arg:
                    sns.barplot(plot_arg["values"], x=plot_arg["xlabel"], y=plot_arg["ylabel"], ax=curr_ax)
            else:
                sns.histplot(plot_arg["values"], bins=plot_arg["bins"], ax=curr_ax)
                curr_ax.set_xlabel(plot_arg["xlabel"])
                curr_ax.set_ylabel(plot_arg["ylabel"])
            curr_ax.set_title(plot_arg["title"])
        plt.tight_layout()
        plt.show()
        return vae_surface_quartiles, vae_surface_z_scores
    
def generate_multiple_surfaces_day_evolution_pre_gen(pre_generated_data, vol_surface_data, ex_data, row, col, check_ex_feats=False, bin_count=15):
    '''
        Input:
            pre_generated_data: a dictionary {
                "surface": np.array of shape (num_day, num_vae, H, W),

                "ex_feats": np.array of shape (num_day, num_vae, num_feats), can be none if we don't check it
            }
    '''
    all_vae_surfaces = pre_generated_data["surface"]
    if "ex_feats" in pre_generated_data and pre_generated_data["ex_feats"] is not None:
        all_vae_ex_feats = pre_generated_data["ex_feats"]
        use_ex_feats = True
    else:
        use_ex_feats = False

    num_days = all_vae_surfaces.shape[0]
    vae_surface_z_scores = []
    vae_surface_quartiles = []
    vae_ex_feat_z_scores = []
    vae_ex_feat_quartiles = []
    moneyness_grid=[0.7, 0.85, 1, 1.15, 1.3]
    ttm_grid=[0.08333, 0.25, 0.5, 1, 2]

    quartile_labels = ["Q1:[0,0.25)", "Q2:[0.25,0.5)", "Q3:[0.5,0.75)", "Q4:(0.75,1]"]
    for day in range(num_days):
        vae_surfaces = all_vae_surfaces[day]
        realized_value = vol_surface_data[day, row, col]
        vae_values = vae_surfaces[:, row, col]

        z_score = (realized_value - np.mean(vae_values)) / np.std(vae_values)
        quartiles = np.percentile(vae_values, [25, 50, 75])
        quartile_idx = np.searchsorted(quartiles, realized_value)
        quartile_region = quartile_labels[quartile_idx]
        vae_surface_z_scores.append(z_score)
        vae_surface_quartiles.append(quartile_region)

        if use_ex_feats and check_ex_feats:
            vae_ex_feats = all_vae_ex_feats[day]
            realized_ex_feats = ex_data[day]
            z_score = (realized_ex_feats - np.mean(vae_ex_feats)) / np.std(vae_ex_feats)
            quartiles = np.percentile(vae_ex_feats, [25, 50, 75])
            quartile_idx = np.searchsorted(quartiles, realized_ex_feats)
            quartile_region = quartile_labels[quartile_idx]
            vae_ex_feat_z_scores.append(z_score)
            vae_ex_feat_quartiles.append(quartile_region)
    
    if use_ex_feats and check_ex_feats:
        plot_args = [{
            "values": pd.DataFrame({x: [vae_surface_quartiles.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
            "xlabel": "Quartile Range",
            "ylabel": "Frequency",
            "title": f"Histogram of where Realized Point lies in daily distribution: \n$x[{ttm_grid[row]}, {moneyness_grid[col]}]$, {num_days} days",
        }, {
            "values": vae_surface_z_scores,
            "xlabel": "Z-score",
            "ylabel": "Frequency",
            "title": f"Z-score of Realized Point: \n$x[{ttm_grid[row]}, {moneyness_grid[col]}]$, {num_days} days",
            "bins": bin_count,
        }, {
            "values": pd.DataFrame({x: [vae_ex_feat_quartiles.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
            "xlabel": "Quartile Range",
            "ylabel": "Frequency",
            "title": f"Histogram of where Realized Point lies in daily distribution: \nReturn, {num_days} Days",
        }, {
            "values": vae_ex_feat_z_scores,
            "xlabel": "Z-score",
            "ylabel": "Frequency",
            "title": f"Z-score of Realized Point: \nReturn, {num_days} days",
            "bins": bin_count,
        }]
        fig, ax = plt.subplots(2, 2, figsize=(11, 12))
        for i in range(2):
            for j in range(2):
                curr_ax = ax[i][j]
                plot_arg = plot_args[2*i + j]
                if "bins" not in plot_arg:
                    sns.barplot(plot_arg["values"], x=plot_arg["xlabel"], y=plot_arg["ylabel"], ax=curr_ax)
                else:
                    sns.histplot(plot_arg["values"], bins=plot_arg["bins"], ax=curr_ax)
                    curr_ax.set_xlabel(plot_arg["xlabel"])
                    curr_ax.set_ylabel(plot_arg["ylabel"])
                curr_ax.set_title(plot_arg["title"])
        plt.tight_layout()
        plt.show()
        return vae_surface_quartiles, vae_surface_z_scores, vae_ex_feat_quartiles, vae_ex_feat_z_scores
    else:
        plot_args = [{
            "values": pd.DataFrame({x: [vae_surface_quartiles.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
            "xlabel": "Quartile Range",
            "ylabel": "Frequency",
            "title": f"Histogram of where Realized Point lies in daily distribution: \n$x[{ttm_grid[row]}, {moneyness_grid[col]}]$, {num_days} days",
        }, {
            "values": vae_surface_z_scores,
            "xlabel": "Z-score",
            "ylabel": "Frequency",
            "title": f"Z-score of Realized Point: \n$x[{ttm_grid[row]}, {moneyness_grid[col]}]$, {num_days} days",
            "bins": bin_count,
        }]
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        for i in range(2):
            curr_ax = ax[i]
            plot_arg = plot_args[i]
            if "bins" not in plot_arg:
                    sns.barplot(plot_arg["values"], x=plot_arg["xlabel"], y=plot_arg["ylabel"], ax=curr_ax)
            else:
                sns.histplot(plot_arg["values"], bins=plot_arg["bins"], ax=curr_ax)
                curr_ax.set_xlabel(plot_arg["xlabel"])
                curr_ax.set_ylabel(plot_arg["ylabel"])
            curr_ax.set_title(plot_arg["title"])
        plt.tight_layout()
        plt.show()
        return vae_surface_quartiles, vae_surface_z_scores
    
def skew_and_slope_daily_evolution(model_data, ex_data, vol_surface_data,
                                             start_day, days_to_generate, num_vaes,
                                             model_type: Union[CVAE, CVAEMem] = CVAEMem,
                                             check_ex_feats=False, bin_count=15, ctx_len=None):
    '''
    Input:
    - model: trained model for vae vol surface
    - ex_data, vol_surface_data: pre-generated latent samples return/price and surfaces data. If SABR data has multiple paths, need to select the path before passing in.
    - start_day: MUST start_day > ctx_len
    - days_to_generate: how many days we want to check the evolution, e.g. 30 day for SABR
    - num_vaes: number of vaes of any choosing
    - check_ex_feats: whether or not we check the distribution of extra features like return/price, default False,  if True, will plot 2 separate distributions
    - bin_count: number of bins to characterize the histogram, default: 15

    Output:
    - z_score_list: list of z-scores for a realized slope and skew vs the distribution of VAEs on that day for 30 days
    - quartile_list: return distribution of quartile ranges the slope and skews point  found on the day
    --plots the data as well
    '''
    model_config = model_data["model_config"]
    model = model_type(model_config)
    model.load_weights(dict_to_load=model_data)
    if ctx_len is None:
        seq_len = model_config["seq_len"]
        ctx_len = model_config["ctx_len"]
    moneyness_grid=[0.7, 0.85, 1, 1.15, 1.3]
    ttm_grid=[0.08333, 0.25, 0.5, 1, 2]

    vae_surface_z_scores_slope = []
    vae_surface_quartiles_slope = []
    vae_surface_z_scores_skew = []
    vae_surface_quartiles_skew = []

    if "ex_feats_dim" in model_config:
        use_ex_feats = model_config["ex_feats_dim"] > 0
    else:
        use_ex_feats = False
    print("use_ex_feats is: ",use_ex_feats)

    quartile_labels = ["Q1:[0,0.25)", "Q2:[0.25,0.5)", "Q3:[0.5,0.75)", "Q4:(0.75,1]"]
    for day in range(start_day, start_day+days_to_generate):
        vae_surfaces, vae_ex_feats = generate_surfaces(model, ex_data, vol_surface_data, day, ctx_len, num_vaes, use_ex_feats, check_ex_feats)

        # calculate the slope as (iv[2 ytm, K/S=1] - iv[3 mtm, K/S=1])
        realized_value_slope = vol_surface_data[day, 4, 2] - vol_surface_data[day,1,2]
        vae_values_slope = vae_surfaces[:, 4, 2] - vae_surfaces[:, 1, 2]

        # calculate the skew as (iv[1 ytm, K/S=0.85] + iv[1 ytm, K/S=1.15])/2 - iv[1 ytm, K/S=1]
        realized_value_skew = (vol_surface_data[day,3, 1] + vol_surface_data[day,3, 3]) / 2 - vol_surface_data[day,3, 2]
        vae_values_skew = (vae_surfaces[:, 3, 1] + vae_surfaces[:, 3, 3]) / 2 - vae_surfaces[:, 3, 2]

        #Slope Calculations for distribution
        z_score_slope = (realized_value_slope - np.mean(vae_values_slope)) / np.std(vae_values_slope)
        quartiles_slope = np.percentile(vae_values_slope, [25, 50, 75])
        quartile_idx_slope = np.searchsorted(quartiles_slope, realized_value_slope)
        quartile_region_slope = quartile_labels[quartile_idx_slope]
        vae_surface_z_scores_slope.append(z_score_slope)
        vae_surface_quartiles_slope.append(quartile_region_slope)

        #Skew Calculations
        z_score_skew = (realized_value_skew - np.mean(vae_values_skew)) / np.std(vae_values_skew)
        quartiles_skew = np.percentile(vae_values_skew, [25, 50, 75])
        quartile_idx_skew = np.searchsorted(quartiles_skew, realized_value_skew)
        quartile_region_skew = quartile_labels[quartile_idx_skew]
        vae_surface_z_scores_skew.append(z_score_skew)
        vae_surface_quartiles_skew.append(quartile_region_skew)


    plot_args = [{
        "values": pd.DataFrame({x: [vae_surface_quartiles_skew.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
        "xlabel": "Quartile Range",
        "ylabel": "Frequency",
        "title": f"Histogram of where Realized Skew lies in daily distribution: {days_to_generate} Days",
    }, {
        "values": vae_surface_z_scores_skew,
        "xlabel": "Z-score",
        "ylabel": "Frequency",
        "title": f"Z-score of Realized Skew distribution: {days_to_generate} Days",
        "bins": bin_count,
    }, {
        "values": pd.DataFrame({x: [vae_surface_quartiles_slope.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
        "xlabel": "Quartile Range",
        "ylabel": "Frequency",
        "title": f"Histogram of where Realized Slope lies in daily distribution: {days_to_generate} Days",
    }, {
        "values": vae_surface_z_scores_slope,
        "xlabel": "Z-score",
        "ylabel": "Frequency",
        "title": f"Z-score of Realized Slope distribution: {days_to_generate} Days",
        "bins": bin_count,
    }]
    fig, ax = plt.subplots(2, 2, figsize=(11, 12))
    for i in range(2):
        for j in range(2):
            curr_ax = ax[i][j]
            plot_arg = plot_args[2*i + j]
            if "bins" not in plot_arg:
                sns.barplot(plot_arg["values"], x=plot_arg["xlabel"], y=plot_arg["ylabel"], ax=curr_ax)
            else:
                sns.histplot(plot_arg["values"], bins=plot_arg["bins"], ax=curr_ax)
                curr_ax.set_xlabel(plot_arg["xlabel"])
                curr_ax.set_ylabel(plot_arg["ylabel"])
            curr_ax.set_title(plot_arg["title"])
    plt.tight_layout()
    plt.show()
    return vae_surface_quartiles_skew, vae_surface_z_scores_skew, vae_surface_quartiles_slope, vae_surface_z_scores_slope

def skew_and_slope_daily_evolution_pre_gen(pre_generated_data, vol_surface_data, bin_count=15):
    '''
        Input:
            pre_generated_data: a dictionary {
                "surface": np.array of shape (num_day, num_vae, H, W),

                "ex_feats": np.array of shape (num_day, num_vae, num_feats), can be none if we don't check it
            }
    '''
    all_vae_surfaces = pre_generated_data["surface"]
    num_days = all_vae_surfaces.shape[0]

    vae_surface_z_scores_slope = []
    vae_surface_quartiles_slope = []
    vae_surface_z_scores_skew = []
    vae_surface_quartiles_skew = []

    quartile_labels = ["Q1:[0,0.25)", "Q2:[0.25,0.5)", "Q3:[0.5,0.75)", "Q4:(0.75,1]"]
    for day in range(num_days):
        vae_surfaces = all_vae_surfaces[day]

        # calculate the slope as (iv[2 ytm, K/S=1] - iv[3 mtm, K/S=1])
        realized_value_slope = vol_surface_data[day, 4, 2] - vol_surface_data[day,1,2]
        vae_values_slope = vae_surfaces[:, 4, 2] - vae_surfaces[:, 1, 2]

        # calculate the skew as (iv[1 ytm, K/S=0.85] + iv[1 ytm, K/S=1.15])/2 - iv[1 ytm, K/S=1]
        realized_value_skew = (vol_surface_data[day,3, 1] + vol_surface_data[day,3, 3]) / 2 - vol_surface_data[day,3, 2]
        vae_values_skew = (vae_surfaces[:, 3, 1] + vae_surfaces[:, 3, 3]) / 2 - vae_surfaces[:, 3, 2]

        #Slope Calculations for distribution
        z_score_slope = (realized_value_slope - np.mean(vae_values_slope)) / np.std(vae_values_slope)
        quartiles_slope = np.percentile(vae_values_slope, [25, 50, 75])
        quartile_idx_slope = np.searchsorted(quartiles_slope, realized_value_slope)
        quartile_region_slope = quartile_labels[quartile_idx_slope]
        vae_surface_z_scores_slope.append(z_score_slope)
        vae_surface_quartiles_slope.append(quartile_region_slope)

        #Skew Calculations
        z_score_skew = (realized_value_skew - np.mean(vae_values_skew)) / np.std(vae_values_skew)
        quartiles_skew = np.percentile(vae_values_skew, [25, 50, 75])
        quartile_idx_skew = np.searchsorted(quartiles_skew, realized_value_skew)
        quartile_region_skew = quartile_labels[quartile_idx_skew]
        vae_surface_z_scores_skew.append(z_score_skew)
        vae_surface_quartiles_skew.append(quartile_region_skew)


    plot_args = [{
        "values": pd.DataFrame({x: [vae_surface_quartiles_skew.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
        "xlabel": "Quartile Range",
        "ylabel": "Frequency",
        "title": f"Histogram of where Realized Skew lies in daily distribution: {num_days} Days",
    }, {
        "values": vae_surface_z_scores_skew,
        "xlabel": "Z-score",
        "ylabel": "Frequency",
        "title": f"Z-score of Realized Skew distribution: {num_days} Days",
        "bins": bin_count,
    }, {
        "values": pd.DataFrame({x: [vae_surface_quartiles_slope.count(x)] for x in quartile_labels}).melt(var_name="Quartile Range", value_name="Frequency"),
        "xlabel": "Quartile Range",
        "ylabel": "Frequency",
        "title": f"Histogram of where Realized Slope lies in daily distribution: {num_days} Days",
    }, {
        "values": vae_surface_z_scores_slope,
        "xlabel": "Z-score",
        "ylabel": "Frequency",
        "title": f"Z-score of Realized Slope distribution: {num_days} Days",
        "bins": bin_count,
    }]
    fig, ax = plt.subplots(2, 2, figsize=(11, 12))
    for i in range(2):
        for j in range(2):
            curr_ax = ax[i][j]
            plot_arg = plot_args[2*i + j]
            if "bins" not in plot_arg:
                sns.barplot(plot_arg["values"], x=plot_arg["xlabel"], y=plot_arg["ylabel"], ax=curr_ax)
            else:
                sns.histplot(plot_arg["values"], bins=plot_arg["bins"], ax=curr_ax)
                curr_ax.set_xlabel(plot_arg["xlabel"])
                curr_ax.set_ylabel(plot_arg["ylabel"])
            curr_ax.set_title(plot_arg["title"])
    plt.tight_layout()
    plt.show()
    return vae_surface_quartiles_skew, vae_surface_z_scores_skew, vae_surface_quartiles_slope, vae_surface_z_scores_slope