import os
import warnings
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier

from eval_scripts.eval_utils import *
from table_making import *
from vae.cvae_with_mem_randomized import CVAEMemRand
from vae.utils import *

warnings.filterwarnings("ignore")

CTX_LEN = 5
START_DAY = 5
DAYS_TO_GENERATE = 5810

PERIODS = [
    pd.date_range("2001-04-01", "2001-11-30"),
    pd.date_range("2008-01-01", "2009-06-30"),
    pd.date_range("2020-03-01", "2020-04-30"),
]

def generate_latents_single_day(model: CVAEMemRand, x: Dict[str, torch.Tensor]):
    surface = x["surface"].to(model.device)
    if len(surface.shape) == 3:
        # unbatched data
        surface = surface.unsqueeze(0)
    B = surface.shape[0]
    T = surface.shape[1]
    C = T - 1
    ctx_surface = surface[:, :C, :, :] # c
    ctx_encoder_input = {"surface": ctx_surface}

    encoder_input = {"surface": surface}
    if "ex_feats" in x:
        ex_feats = x["ex_feats"].to(model.device)
        if len(ex_feats.shape) == 2:
            ex_feats = ex_feats.unsqueeze(0)
        ctx_ex_feats = ex_feats[:, :C, :]
        ctx_encoder_input["ex_feats"] = ctx_ex_feats
        encoder_input["ex_feats"] = ex_feats

    ctx_embedding = model.ctx_encoder(ctx_encoder_input).detach().cpu().numpy() # embedded c (B, C, n)
    z_mean, z_log_var, z = model.encoder(encoder_input) # P(z|c,x), (B, T, latent_dim)

    return {
        "ctx": ctx_embedding, 
        "z_mean": z_mean.detach().cpu().numpy(), 
        "z_log_var": z_log_var.detach().cpu().numpy()
    }

def generate_latents_multi_day(model, fp, surf_data, ex_data=None, 
                               ctx_len=CTX_LEN, start_day=START_DAY, days_to_generate=DAYS_TO_GENERATE):
    if not os.path.exists(fp):
        all_day_ctx_embedding = np.zeros((days_to_generate, ctx_len, model.config["latent_dim"]))
        all_day_z_mean = np.zeros((days_to_generate, ctx_len + 1, model.config["latent_dim"]))
        all_day_z_log_var = np.zeros((days_to_generate, ctx_len + 1, model.config["latent_dim"]))

        for day in range(start_day, start_day+days_to_generate):
            if day % 1000 == 0:
                print(f"Generating day {day}")
                
            x = {
                "surface": torch.tensor(surf_data[day-start_day:day+1]),
            }
            if model.config["ex_feats_dim"] > 0:
                x["ex_feats"] = torch.tensor(ex_data[day-start_day:day+1])
            output_dict = generate_latents_single_day(model, x)
            all_day_ctx_embedding[day - start_day, ...] = output_dict["ctx"].reshape((ctx_len, model.config["latent_dim"]))
            all_day_z_mean[day - start_day, ...] = output_dict["z_mean"].reshape((ctx_len + 1, model.config["latent_dim"]))
            all_day_z_log_var[day - start_day, ...] = output_dict["z_log_var"].reshape((ctx_len + 1, model.config["latent_dim"]))
        np.savez(fp,
            ctx_emb=all_day_ctx_embedding,
            z_mean=all_day_z_mean,
            z_log_var=all_day_z_log_var)
    else:
        gen_data = np.load(fp)
        all_day_ctx_embedding = gen_data["ctx_emb"]
        all_day_z_mean = gen_data["z_mean"]
        all_day_z_log_var = gen_data["z_log_var"]
    
    return all_day_ctx_embedding, all_day_z_mean, all_day_z_log_var

def compute_surface_pca(spx_df: pd.DataFrame):
    df = spx_df.copy()
    all_cols = ["ttm_one_month_moneyness_pt_seven", "ttm_one_month_moneyness_pt_eightfive", "ttm_one_month_moneyness_pt_one", "ttm_one_month_moneyness_pt_oneonefive", "ttm_one_month_moneyness_pt_onethree",
        "ttm_three_month_moneyness_pt_seven", "ttm_three_month_moneyness_pt_eightfive", "ttm_three_month_moneyness_pt_one", "ttm_three_month_moneyness_pt_oneonefive", "ttm_three_month_moneyness_pt_onethree",
        "ttm_six_month_moneyness_pt_seven", "ttm_six_month_moneyness_pt_eightfive", "ttm_six_month_moneyness_pt_one", "ttm_six_month_moneyness_pt_oneonefive", "ttm_six_month_moneyness_pt_onethree",
        "ttm_one_year_moneyness_pt_seven", "ttm_one_year_moneyness_pt_eightfive", "ttm_one_year_moneyness_pt_one", "ttm_one_year_moneyness_pt_oneonefive", "ttm_one_year_moneyness_pt_onethree",   
        "ttm_two_year_moneyness_pt_seven", "ttm_two_year_moneyness_pt_eightfive", "ttm_two_year_moneyness_pt_one", "ttm_two_year_moneyness_pt_oneonefive", "ttm_two_year_moneyness_pt_onethree"
    ]
    pca_out_cols = [f"surf_dim{i}" for i in range(10)]
    all_cols_m1 = [f"{col}_m1" for col in all_cols]
    all_cols_m2 = [f"{col}_m2" for col in all_cols]
    all_cols_m3 = [f"{col}_m3" for col in all_cols]
    all_cols_m4 = [f"{col}_m4" for col in all_cols]
    all_cols_m5 = [f"{col}_m5" for col in all_cols]
    pca_cols = all_cols_m1 + all_cols_m2 + all_cols_m3 + all_cols_m4 + all_cols_m5

    for i, cols in enumerate([all_cols_m1, all_cols_m2, all_cols_m3, all_cols_m4, all_cols_m5]):
        df[cols] = df[all_cols].shift(i+1)
    df_to_pca = df[["date"] + pca_cols].copy()
    df_to_pca.dropna(inplace=True)
    pca = PCA(n_components=10, random_state=0).fit(df_to_pca[pca_cols].dropna().values)
    print(pca.explained_variance_ratio_)
    print(np.sum(pca.explained_variance_ratio_))
    pca_ctx_emb = pca.transform(df_to_pca[pca_cols].dropna().values)
    df_to_pca[pca_out_cols] = pca_ctx_emb
    return df_to_pca[["date"] + pca_out_cols].copy()

def build_regress_dataset(start_day=START_DAY, days_to_generate=DAYS_TO_GENERATE):
    spx_df = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
    df = spx_df[["date", "price"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["NBER"] = (df["date"].isin(PERIODS[0]) | df["date"].isin(PERIODS[1]) | df["date"].isin(PERIODS[2])).astype(int)
    for forecast_dates, suffix in [(30, "1m"), (90, "3m"), (180, "6m"), (365, "12m")]:
        for i in range(len(df) - forecast_dates):
            df.loc[i, f"NBER_ind_{suffix}"] = int(np.any(df["NBER"].values[i:i+forecast_dates+1] > 0))
        df[f"NBER_ind_{suffix}"].fillna(0, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df["ret"] = np.log(df["price"]) - np.log(df["price"].shift(1))
    df["ret2"] = np.log(df["price"].shift(-1)) - np.log(df["price"])
    df["ret_5d"] = np.log(df["price"].shift(-4)) - np.log(df["price"].shift(1))
    df["ret_5d2"] = np.log(df["price"].shift(-5)) - np.log(df["price"])
    for i, ret_col in enumerate(["ret", "ret2", "ret_5d", "ret_5d2"]):
        df[f"ret_ind{i}"] = np.where(df[ret_col] > 0, 1, -1)
    for i in range(len(df) - 5):
        df.loc[i, "vol_5d"] = np.std(df.loc[i:i+4, "ret"])
    df = df.loc[start_day:start_day+days_to_generate-1, 
                ["date", "price", "NBER", "ret", "ret2", "ret_5d", "ret_5d2", "vol_5d"] + 
                [f"ret_ind{i}" for i in range(4)] +
                [f"NBER_ind_{suffix}" for suffix in ["1m", "3m", "6m", "12m"]]
            ].reset_index(drop=True).copy()
    pca_df = compute_surface_pca(spx_df)
    df = pd.merge(df, pca_df, on="date", how="left")
    return df

def compute_pca_embedding(base_embedding, gen_embedding, df):
    df = df.copy()
    ctx_emb = base_embedding.reshape((-1, base_embedding.shape[1] * base_embedding.shape[2]))
    pca = PCA(n_components=2, random_state=0).fit(ctx_emb)
    print(pca.explained_variance_ratio_)
    pca_ctx_emb = pca.transform(ctx_emb)
    print(pca_ctx_emb.shape)
    ctx_emb2 = gen_embedding.reshape((-1, gen_embedding.shape[1] * gen_embedding.shape[2]))
    pca_ctx_emb2 = pca.transform(ctx_emb2)
    print(pca_ctx_emb2.shape)
    df["ctx_dim0"] = pca_ctx_emb[:, 0]
    df["ctx_dim1"] = pca_ctx_emb[:, 1]
    df.loc[START_DAY:, "gen_ctx_dim0"] = pca_ctx_emb2[:len(pca_ctx_emb2), 0]
    df.loc[START_DAY:, "gen_ctx_dim1"] = pca_ctx_emb2[:len(pca_ctx_emb2), 1]
    return df

def get_pca_plot(ctx_embedding, ctx_embedding2, start_day=START_DAY, days_to_generate=DAYS_TO_GENERATE,
                             base_folder="tables/iv_all", model_type="loss"):
    os.makedirs(f"{base_folder}/plots", exist_ok=True)
    
    df = build_regress_dataset(start_day, days_to_generate)
    df = compute_pca_embedding(ctx_embedding, ctx_embedding2, df)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    subset_df = df[(df["date"].isin(PERIODS[0]) | df["date"].isin(PERIODS[1]) | df["date"].isin(PERIODS[2]))].copy()
    ax[0].scatter(subset_df["ctx_dim0"], subset_df["ctx_dim1"], s=10, c="red", label="NBER recession", marker="^")
    ax[1].scatter(subset_df["gen_ctx_dim0"], subset_df["gen_ctx_dim1"], s=10, c="red", label="NBER recession", marker="^")

    subset_df = df[~(df["date"].isin(PERIODS[0]) | df["date"].isin(PERIODS[1]) | df["date"].isin(PERIODS[2]))].copy()
    ax[0].scatter(subset_df["ctx_dim0"], subset_df["ctx_dim1"], alpha=0.3, s=10, c="blue", label="NBER normal")
    ax[1].scatter(subset_df["gen_ctx_dim0"], subset_df["gen_ctx_dim1"], alpha=0.3, s=10, c="blue", label="NBER normal")

    xlimits = ax[0].get_xlim()
    ylimits = ax[0].get_ylim()
    ax[1].set_xlim(xlimits)
    ax[1].set_ylim(ylimits)

    ax[0].set_xlabel("First Principal Component")
    ax[0].set_ylabel("Second Principal Component")
    ax[0].set_title("Encoded Original")
    ax[0].legend(loc="lower right")
    ax[1].set_xlabel("First Principal Component")
    ax[1].set_ylabel("Second Principal Component")
    ax[1].set_title("Encoded MLE")
    ax[1].legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(f"{base_folder}/plots/pca_{model_type}.jpg")
    plt.close()

def parse_model_summaries(summaries: Dict[str, Any], coeff_names: List[str]):
    cols = [""] + list(summaries.keys())
    row_count = (len(coeff_names) + 1) * 2 + 2
    summary_df = pd.DataFrame(index=list(range(row_count)), columns=cols)
    for i, coeff_name in enumerate(coeff_names):
        summary_df.loc[2 * i, ""] = coeff_name
        summary_df.loc[2 * i + 1, ""] = ""
    summary_df.loc[row_count-4, ""] = "Intercept"
    summary_df.loc[row_count-3, ""] = ""
    summary_df.loc[row_count-2, ""] = "R-squared Adj."
    summary_df.loc[row_count-1, ""] = "N"

    def get_coeff_format(param, pval):
        coeff_str = "{0:.3f}".format(param)
        if pval < 0.1:
            coeff_str += "*"
        if pval < 0.05:
            coeff_str += "*"
        if pval < 0.01:
            coeff_str += "*"
        return coeff_str


    for k, summ in summaries.items():
        params = summ.params
        pvals = summ.pvalues
        ses = summ.bse
        prsquared = summ.prsquared
        nobs = summ.nobs
        for i, coeff_name in enumerate(coeff_names):
            try:
                summary_df.loc[2 * i, k] = get_coeff_format(params[coeff_name], pvals[coeff_name])
                summary_df.loc[2 * i + 1, k] = f"({ses[coeff_name]:.3f})"
            except:
                continue
        summary_df.loc[row_count-4, k] = get_coeff_format(params["Intercept"], pvals["Intercept"])
        summary_df.loc[row_count-3, k] =  f"({ses[0]:.3f})"
        summary_df.loc[row_count-2, k] = "{0:.3f}".format(prsquared)
        summary_df.loc[row_count-1, k] = str(nobs)
    
    summary_df = summary_df.set_index("")
    return summary_df

def baseline_reg(df: pd.DataFrame):
    summaries = {}
    result_dfs = {}

    for y_var in ["NBER_ind_1m"]:
        base_xvars = [f"surf_dim{i}" for i in range(10)]
        for i, x_vars in enumerate([base_xvars[0:2], base_xvars]):
            base_surface_reg_df = df[["date", y_var, "price"] + x_vars].copy().dropna()
            probit_model = smf.probit(f"{y_var}~1+{'+'.join(x_vars)}", data=base_surface_reg_df).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
            base_surface_reg_df.loc[:, "pred"] = probit_model.predict(base_surface_reg_df)
            result_dfs[f"Surface NBER{i+1}"] = base_surface_reg_df
            summaries[f"Surface NBER{i+1}"] = probit_model
    return summaries, result_dfs

def plot_prob_comparison_baseline(dfs: Dict[str, pd.DataFrame], fn):
    '''
    pd.date_range("2001-04-01", "2001-11-30"),
    pd.date_range("2008-01-01", "2009-06-30"),
    pd.date_range("2020-03-01", "2020-04-30"),
    '''
    # 
    era_list = [
        {"start": "2001-04-01", "stop": "2001-11-30"},
        {"start": "2008-01-01", "stop": "2009-06-30"},
        {"start": "2020-03-01", "stop": "2020-04-30"},
    ]
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    title = "Estimated Probability of Recession and SPX Price"
    df_base = dfs["Surface NBER1"]
    df_generated = dfs["Surface NBER2"]
    ax.plot(df_base["date"], df_base["pred"], color="blue", label="2 PCs")
    ax.plot(df_generated["date"], df_generated["pred"], color="red", label="10 PCs")
    twin_ax = ax.twinx()
    twin_ax.plot(df_base["date"], df_base["price"], color="orange", label="SPX Price")
    twin_ax.set_ylabel("SPX Price")
    twin_ax.set_ylim([-400, 6000])

    ax_lines, ax_labels = ax.get_legend_handles_labels()
    twinax_lines, twinax_labels = twin_ax.get_legend_handles_labels()
    ax.legend(ax_lines + twinax_lines, ax_labels + twinax_labels, loc="upper left")
    ax.set_xlabel("date")
    ax.set_ylabel("Estimated Probability")
    # ax.set_title(title)
    for k in range(len(era_list)):
        ax.axvspan(era_list[k]["start"], era_list[k]["stop"], facecolor='grey', alpha=.2)
    ax.set_ylim([-0.4, 1.6])
    plt.savefig(fn)
    plt.close()

def plot_prob_comparison(dfs: Dict[str, pd.DataFrame], fn):
    '''
    pd.date_range("2001-04-01", "2001-11-30"),
    pd.date_range("2008-01-01", "2009-06-30"),
    pd.date_range("2020-03-01", "2020-04-30"),
    '''
    # 
    era_list = [
        {"start": "2001-04-01", "stop": "2001-11-30"},
        {"start": "2008-01-01", "stop": "2009-06-30"},
        {"start": "2020-03-01", "stop": "2020-04-30"},
    ]
    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    title = "Estimated Probability of Recession and SPX Price"
    df_base = dfs["Base NBER"]
    df_generated = dfs["Generated NBER"]
    ax.plot(df_base["date"], df_base["pred"], color="blue", label="Encoded Original")
    ax.plot(df_generated["date"], df_generated["pred"], color="red", label="Encoded MLE")
    twin_ax = ax.twinx()
    twin_ax.plot(df_base["date"], df_base["price"], color="orange", label="SPX Price")
    twin_ax.set_ylabel("SPX Price")
    twin_ax.set_ylim([-400, 6000])

    ax_lines, ax_labels = ax.get_legend_handles_labels()
    twinax_lines, twinax_labels = twin_ax.get_legend_handles_labels()
    ax.legend(ax_lines + twinax_lines, ax_labels + twinax_labels, loc="upper left")
    ax.set_xlabel("date")
    ax.set_ylabel("Estimated Probability")
    # ax.set_title(title)
    for k in range(len(era_list)):
        ax.axvspan(era_list[k]["start"], era_list[k]["stop"], facecolor='grey', alpha=.2)
    ax.set_ylim([-0.4, 1.6])
    plt.savefig(fn)
    plt.close()

def basic_regressions(df: pd.DataFrame):   
    summaries = {}
    result_dfs = {}

    for y_var in ["NBER_ind_1m"]:
        for x_var_config in [{"model_name": "Base NBER", "x_vars": {"ctx_dim0": "FPC", "ctx_dim1": "SPC"}}, 
                            {"model_name": "Generated NBER", "x_vars": {"gen_ctx_dim0": "FPC", "gen_ctx_dim1": "SPC"}}]:
            df_ = df[["date", y_var, "price"] + list(x_var_config["x_vars"].keys())].copy().rename(columns=x_var_config["x_vars"])
            df_ = df_.dropna()
            probit_model = smf.probit(f"{y_var}~1+{'+'.join(list(x_var_config['x_vars'].values()))}", data=df_).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
            df_.loc[:, "pred"] = probit_model.predict(df_)
            result_dfs[x_var_config["model_name"]] = df_
            summaries[x_var_config["model_name"]] = probit_model
    sm_df1 = parse_model_summaries(summaries, ["FPC", "SPC"])

    summaries2 = {}
    for y_var in ["vol_5d"]:
        for x_var_config in [{"model_name": "Base Vol5d", "x_vars": {"ctx_dim0": "FPC", "ctx_dim1": "SPC"}}, 
                            {"model_name": "Generated Vol5d", "x_vars": {"gen_ctx_dim0": "FPC", "gen_ctx_dim1": "SPC"}}]:
            df_ = df[[y_var, "price"] + list(x_var_config["x_vars"].keys())].copy().rename(columns=x_var_config["x_vars"])
            df_ = df_.dropna()
            reg_model = smf.ols(f"{y_var}~1+{'+'.join(list(x_var_config['x_vars'].values()))}", data=df_).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
            df_.loc[:, "pred"] = reg_model.predict(df_)
            summaries2[x_var_config["model_name"]] = reg_model
    sm_df2 = summary_col(list(summaries2.values()), model_names=list(summaries2.keys()), 
                stars=True, info_dict={"N": lambda x: "{0:d}".format(int(x.nobs))},
                regressor_order=["FPC", "SPC"]).tables[0]

    summ_df = pd.concat([sm_df1.reset_index(names=""), 
                        sm_df2.reset_index(drop=True)], axis=1).set_index("")
    ltx = summ_df.style.to_latex(hrules=True, column_format="l" + "c" * len(summ_df.columns))
    ltx = ltx.replace(" & Base NBER & Generated NBER & Base Vol5d & Generated Vol5d \\\\\n &  &  &  &  \\\\\n\\midrule",
    r""" & \multicolumn{2}{c}{NBER} & \multicolumn{2}{c}{Vol5d} \\
    & Encoded Original & Encoded MLE & Encoded Original & Encoded MLE \\
    & (1) & (2) & (3) & (4)\\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5}""")
    ltx = ltx.replace("R-squared Adj.", "$R^2$")
    return ltx, summ_df, result_dfs


def incremental_regressions(df: pd.DataFrame):   
    summaries = {}
    result_dfs = {}

    for y_var in ["NBER_ind_1m"]:
        base_xvars = [f"surf_dim{i}" for i in range(10)]
        base_surface_reg_df = df[["date", y_var, "price"] + base_xvars].copy().dropna()
        probit_model = smf.probit(f"{y_var}~1+{'+'.join(base_xvars[0:2])}", data=base_surface_reg_df).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        base_surface_reg_df.loc[:, "pred"] = probit_model.predict(base_surface_reg_df)
        result_dfs["Surface NBER1"] = base_surface_reg_df
        summaries["Surface NBER1"] = probit_model

        probit_model = smf.probit(f"{y_var}~1+{'+'.join(base_xvars)}", data=base_surface_reg_df).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        base_surface_reg_df.loc[:, "pred"] = probit_model.predict(base_surface_reg_df)
        result_dfs["Surface NBER2"] = base_surface_reg_df
        summaries["Surface NBER2"] = probit_model
        
        for x_var_config in [{"model_name": "Base NBER", "x_vars": {"ctx_dim0": "FPC", "ctx_dim1": "SPC"}}, 
                            {"model_name": "Generated NBER", "x_vars": {"gen_ctx_dim0": "FPC", "gen_ctx_dim1": "SPC"}}]:
            df_ = df[["date", y_var, "price"] + base_xvars + list(x_var_config["x_vars"].keys())].copy().rename(columns=x_var_config["x_vars"])
            df_ = df_.dropna()
            for i, base_reg_vars in enumerate([base_xvars[0:2], base_xvars]): 
                probit_model = smf.probit(f"{y_var}~1+{'+'.join(base_reg_vars + list(x_var_config['x_vars'].values()))}", data=df_).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
                df_.loc[:, "pred"] = probit_model.predict(df_)
                result_dfs[x_var_config["model_name"] + str(i)] = df_
                summaries[x_var_config["model_name"] + str(i)] = probit_model
    sm_df1 = parse_model_summaries(summaries, ["surf_dim0", "surf_dim1", "FPC", "SPC"])

    summaries2 = {}
    for y_var in ["vol_5d"]:
        base_xvars = [f"surf_dim{i}" for i in range(10)]
        base_surface_reg_df = df[["date", y_var, "price"] + base_xvars].copy().dropna()
        reg_model = smf.ols(f"{y_var}~1+{'+'.join(base_xvars[0:2])}", data=base_surface_reg_df).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        base_surface_reg_df.loc[:, "pred"] = reg_model.predict(base_surface_reg_df)
        summaries2["Surface Vol1"] = reg_model
        reg_model = smf.ols(f"{y_var}~1+{'+'.join(base_xvars)}", data=base_surface_reg_df).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
        base_surface_reg_df.loc[:, "pred"] = reg_model.predict(base_surface_reg_df)
        summaries2["Surface Vol2"] = reg_model

        for x_var_config in [{"model_name": "Base Vol5d", "x_vars": {"ctx_dim0": "FPC", "ctx_dim1": "SPC"}}, 
                            {"model_name": "Generated Vol5d", "x_vars": {"gen_ctx_dim0": "FPC", "gen_ctx_dim1": "SPC"}}]:
            df_ = df[[y_var, "price"] + base_xvars + list(x_var_config["x_vars"].keys())].copy().rename(columns=x_var_config["x_vars"])
            df_ = df_.dropna()
            for i, base_reg_vars in enumerate([base_xvars[0:2], base_xvars]): 
                reg_model = smf.ols(f"{y_var}~1+{'+'.join(base_reg_vars + list(x_var_config['x_vars'].values()))}", data=df_).fit(cov_type="HAC", cov_kwds={"maxlags": 30})
                df_.loc[:, "pred"] = reg_model.predict(df_)
                summaries2[x_var_config["model_name"] + str(i)] = reg_model
    sm_df2 = summary_col(list(summaries2.values()), model_names=list(summaries2.keys()), 
                stars=True, info_dict={"N": lambda x: "{0:d}".format(int(x.nobs))},
                regressor_order=["surf_dim0", "surf_dim1", "FPC", "SPC", "Intercept"], drop_omitted=True).tables[0]

    ltx1 = sm_df1.style.to_latex(hrules=True, column_format="l" + "c" * len(sm_df1.columns))
    ltx1 = ltx1.replace(" & Surface NBER1 & Surface NBER2 & Base NBER0 & Base NBER1 & Generated NBER0 & Generated NBER1 \\\\\n &  &  &  &  &  &  \\\\\n\\midrule",
    r""" & \multicolumn{2}{c}{Original} & \multicolumn{2}{c}{Encoded Original} & \multicolumn{2}{c}{Encoded MLE} \\
    & 2PCs & 10PCs & 2PCs & 10PCs & 2PCs & 10PCs \\
    & (1) & (2) & (3) & (4) & (5) & (6) \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}""")
    ltx1 = ltx1.replace("R-squared Adj.", "$R^2$")
    ltx1 = ltx1.replace("nan", "")
    ltx1 = ltx1.replace("surf_dim0", r"$\text{PC}_1$")
    ltx1 = ltx1.replace("surf_dim1", r"$\text{PC}_2$")

    ltx2 = sm_df2.style.to_latex(hrules=True, column_format="l" + "c" * len(sm_df2.columns))
    ltx2 = ltx2.replace(" & Surface Vol1 & Surface Vol2 & Base Vol5d0 & Base Vol5d1 & Generated Vol5d0 & Generated Vol5d1 \\\\\n\\midrule",
    r""" & \multicolumn{2}{c}{Original} & \multicolumn{2}{c}{Encoded Original} & \multicolumn{2}{c}{Encoded MLE} \\
    & 2PCs & 10PCs & 2PCs & 10PCs & 2PCs & 10PCs \\
    & (1) & (2) & (3) & (4) & (5) & (6) \\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}""")
    ltx2 = ltx2.replace("R-squared Adj.", "$R^2$")
    ltx2 = ltx2.replace("nan", "")
    ltx2 = ltx2.replace("surf_dim0", r"$\text{PC}_1$")
    ltx2 = ltx2.replace("surf_dim1", r"$\text{PC}_2$")
    return ltx1, ltx2


def classification_task(df: pd.DataFrame):
    df = df.copy()
    indices = [[], []]
    for y in [f"ret_ind{i}" for i in range(4)]:
        for data_target in ["train (Encoded MLE)", "test (Encoded Original)"]:
            indices[0].append(y)
            indices[1].append(data_target)

    report_df = pd.DataFrame(index=pd.MultiIndex.from_arrays(indices), columns=["accuracy", "precision", "recall", "f1-score", "auc"])

    for y_var in [f"ret_ind{i}" for i in range(4)]:
        train_df = df[["gen_ctx_dim0", "gen_ctx_dim1", y_var]].copy().dropna().rename(columns={"gen_ctx_dim0": "FPC", 
                                                                                                    "gen_ctx_dim1": "SPC",
                                                                                                    y_var: "y"})
        test_df = df[["ctx_dim0", "ctx_dim1", y_var]].copy().dropna().rename(columns={"ctx_dim0": "FPC", 
                                                                                        "ctx_dim1": "SPC",
                                                                                        y_var: "y"})
        classifier = MLPClassifier(random_state=0)
        clf = classifier.fit(train_df[["FPC", "SPC"]].values, train_df["y"].values)
        pred_train_y = clf.predict(train_df[["FPC", "SPC"]].values)
        pred_test_y = clf.predict(test_df[["FPC", "SPC"]].values)
        report_train = classification_report(train_df["y"].values, pred_train_y, output_dict=True)
        report_test = classification_report(test_df["y"].values, pred_test_y, output_dict=True)
        auc_train_y = roc_auc_score(train_df["y"].values, clf.predict_proba(train_df[["FPC", "SPC"]].values)[:, 1])
        auc_test_y = roc_auc_score(test_df["y"].values, clf.predict_proba(test_df[["FPC", "SPC"]].values)[:, 1])
        '''
        Use 'weighted'
        :
        Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). 
        This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
        '''
        weighted_report_train = report_train["weighted avg"]
        weighted_report_test = report_test["weighted avg"]
        weighted_report_train["accuracy"] = report_train["accuracy"]
        weighted_report_test["accuracy"] = report_test["accuracy"]
        weighted_report_train["auc"] = auc_train_y
        weighted_report_test["auc"] = auc_test_y
        for k in report_df.columns:
            report_df.loc[(y_var, "train (Encoded MLE)"), k] = "{0:.3f}".format(weighted_report_train[k])
            report_df.loc[(y_var, "test (Encoded Original)"), k] = "{0:.3f}".format(weighted_report_test[k])
    ltx = report_df.style.to_latex(column_format="ll" + "c" * len(report_df.columns), multirow_align="t", hrules=True)
    ltx = ltx.replace("ret_ind0", r"$\text{Ret}_{t-1:t}$")
    ltx = ltx.replace("ret_ind1", r"$\text{Ret}_{t:t+1}$")
    ltx = ltx.replace("ret_ind2", r"$\text{Ret}_{t-1:t+4}$")
    ltx = ltx.replace("ret_ind3", r"$\text{Ret}_{t:t+5}$")
    ltx = ltx.replace(" &  & accuracy & precision & recall & f1-score & auc \\\\\n\\midrule",
    r""" & & Accuracy & Precision (Weighted) & Recall (Weighted) & F1-score (Weighted) & AUC \\
    \cmidrule(lr){3-7}""")
    return ltx, report_df

def classification_task_is_oos(df: pd.DataFrame):
    df = df.copy()
    indices = [[], []]
    for y in [f"ret_ind{i}" for i in range(4)]:
        for data_target in ["train (2000-2015)", "test (2016-2023)"]:
            indices[0].append(y)
            indices[1].append(data_target)

    report_df = pd.DataFrame(index=pd.MultiIndex.from_arrays(indices), columns=["accuracy", "precision", "recall", "f1-score", "auc"])

    for y_var in [f"ret_ind{i}" for i in range(4)]:
        df_ = df[["date", "gen_ctx_dim0", "gen_ctx_dim1", y_var]].copy().dropna().rename(columns={"gen_ctx_dim0": "FPC", 
                                                                                                    "gen_ctx_dim1": "SPC",
                                                                                                    y_var: "y"})
        train_df = df_[df_["date"] < "2016-01-01"].copy()
        test_df = df_[df_["date"] >= "2016-01-01"].copy()
        classifier = MLPClassifier(random_state=0)
        clf = classifier.fit(train_df[["FPC", "SPC"]].values, train_df["y"].values)
        pred_train_y = clf.predict(train_df[["FPC", "SPC"]].values)
        pred_test_y = clf.predict(test_df[["FPC", "SPC"]].values)
        report_train = classification_report(train_df["y"].values, pred_train_y, output_dict=True)
        report_test = classification_report(test_df["y"].values, pred_test_y, output_dict=True)
        auc_train_y = roc_auc_score(train_df["y"].values, clf.predict_proba(train_df[["FPC", "SPC"]].values)[:, 1])
        auc_test_y = roc_auc_score(test_df["y"].values, clf.predict_proba(test_df[["FPC", "SPC"]].values)[:, 1])
        '''
        Use 'weighted'
        :
        Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). 
        This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
        '''
        weighted_report_train = report_train["weighted avg"]
        weighted_report_test = report_test["weighted avg"]
        weighted_report_train["accuracy"] = report_train["accuracy"]
        weighted_report_test["accuracy"] = report_test["accuracy"]
        weighted_report_train["auc"] = auc_train_y
        weighted_report_test["auc"] = auc_test_y
        for k in report_df.columns:
            report_df.loc[(y_var, "train (2000-2015)"), k] = "{0:.3f}".format(weighted_report_train[k])
            report_df.loc[(y_var, "test (2016-2023)"), k] = "{0:.3f}".format(weighted_report_test[k])
    ltx = report_df.style.to_latex(column_format="ll" + "c" * len(report_df.columns), multirow_align="t", hrules=True)
    ltx = ltx.replace("ret_ind0", r"$\text{Ret}_{t-1:t}$")
    ltx = ltx.replace("ret_ind1", r"$\text{Ret}_{t:t+1}$")
    ltx = ltx.replace("ret_ind2", r"$\text{Ret}_{t-1:t+4}$")
    ltx = ltx.replace("ret_ind3", r"$\text{Ret}_{t:t+5}$")
    ltx = ltx.replace(" &  & accuracy & precision & recall & f1-score & auc \\\\\n\\midrule",
    r""" & & Accuracy & Precision (Weighted) & Recall (Weighted) & F1-score (Weighted) & AUC \\
    \cmidrule(lr){3-7}""")
    return ltx, report_df