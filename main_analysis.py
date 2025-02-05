from analysis_code import *
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.size"] = 15

set_seeds(0)
torch.set_default_dtype(torch.float64)

BASE_MODEL_DIR = "test_spx/2024_11_09"
BASE_TABLE_DIR = "tables/2024_1213"
os.makedirs(BASE_TABLE_DIR, exist_ok=True)

SUFFIX_TAG_MAP = {
    "no_ex": "No EX", 
    "ex_no_loss": "EX No Loss", 
    "ex_loss": "EX Loss"
}

CTX_LEN = 5
START_DAY = 5
DAYS_TO_GENERATE = 5810

SPX_DF = pd.read_parquet("data/spx_vol_surface_history_full_data_fixed.parquet")
print(len(SPX_DF), SPX_DF["date"].min(), SPX_DF["date"].max())

## Real Data
data = np.load("data/vol_surface_with_ret.npz")
vol_surf_data = data["surface"]
ret_data = data["ret"]
level_data = data["levels"]
skew_data = data["skews"]
slope_data = data["slopes"]
ex_data = np.concatenate([ret_data[...,np.newaxis], skew_data[...,np.newaxis], slope_data[...,np.newaxis]], axis=-1)
print(ex_data.shape)

## Loss
print("{0:=^80}".format("Loss"))
loss_table = get_loss_table(f"{BASE_MODEL_DIR}/results.csv")
with open(f"{BASE_TABLE_DIR}/loss.tex", "w") as f:
    f.write(loss_table)

base_df = build_regress_dataset(START_DAY, DAYS_TO_GENERATE)
summ_df, result_dfs = baseline_reg(base_df)
plot_prob_comparison_baseline(result_dfs, f"{BASE_TABLE_DIR}/plots/probit_reg_baseline.png")
arbitrage_simulated_data = {}
for k in SUFFIX_TAG_MAP:
    print(f"{k:=^80}")
    simulation = np.load(f"{BASE_MODEL_DIR}/{k}_gen5.npz")
    simulated_surfaces = simulation["surfaces"]

    simulation_mle = np.load(f"{BASE_MODEL_DIR}/{k}_mle_gen5.npz")
    simulated_surfaces_mle = simulation_mle["surfaces"] # (num_days, 1, 5, 5), because we only have one simulation
    simulated_surfaces_mle_reduced = np.mean(simulated_surfaces_mle, axis=1)

    if k == "ex_loss":
        simulated_ret = simulation_mle["ex_feats"][:, :, :1]
        simulated_ret = np.mean(simulated_ret, axis=1)
    else:
        simulated_ret = ret_data[START_DAY:START_DAY+simulated_surfaces_mle.shape[0], np.newaxis]
    simulated_skews = (simulated_surfaces_mle_reduced[:, 3, 1] + simulated_surfaces_mle_reduced[:, 3, 3]) / 2 - simulated_surfaces_mle_reduced[:, 3, 2]
    simulated_slopes = simulated_surfaces_mle_reduced[:, 4, 2] - simulated_surfaces_mle_reduced[:, 1, 2]
    simulated_ex_data = np.concatenate([simulated_ret, 
                                    simulated_skews[...,np.newaxis], 
                                    simulated_slopes[...,np.newaxis]], axis=-1)

    # Regression
    print("{0:=^40}".format("Regression"))
    regression_surface_grids_mle(simulated_surfaces_mle, SPX_DF.copy(), 
                            ctx_len=CTX_LEN, days_to_generate=DAYS_TO_GENERATE, 
                            base_folder=BASE_TABLE_DIR, model_type=k)
    plot_surface_grids(simulated_surfaces_mle, SPX_DF.copy(), 
                        ctx_len=CTX_LEN, days_to_generate=DAYS_TO_GENERATE, 
                        base_folder=BASE_TABLE_DIR, model_type=k)
    benchmark_with_rmse(simulated_surfaces_mle, SPX_DF.copy(), 
                        ctx_len=CTX_LEN, days_to_generate=DAYS_TO_GENERATE,
                        base_folder=BASE_TABLE_DIR, model_type=k)
    
    # Latent PCA
    print("{0:=^40}".format("PCA"))
    file_path = f"{BASE_MODEL_DIR}/{k}.pt"
    model_data = torch.load(file_path) # latent_dim=5, surface_hidden=[5,5,5], mem_hidden=100
    model_config = model_data["model_config"]
    model = CVAEMemRand(model_config)
    model.load_weights(dict_to_load=model_data)

    ctx_embedding, z_mean, z_log_var = generate_latents_multi_day(model, 
                                                                f"{BASE_MODEL_DIR}/{k}_gen5_base_latent.npz",
                                                                vol_surf_data, ex_data,
                                                                    CTX_LEN, START_DAY, DAYS_TO_GENERATE)
    ctx_embedding2, z_mean2, z_log_var2 = generate_latents_multi_day(model, 
                                                                    f"{BASE_MODEL_DIR}/{k}_gen5_latent.npz",
                                                                    simulated_surfaces_mle_reduced, simulated_ex_data,
                                                                    CTX_LEN, START_DAY, DAYS_TO_GENERATE-5)
    base_df = build_regress_dataset(START_DAY, DAYS_TO_GENERATE)
    reg_df = compute_pca_embedding(ctx_embedding, ctx_embedding2, base_df)
    get_pca_plot(ctx_embedding, ctx_embedding2, START_DAY, DAYS_TO_GENERATE, BASE_TABLE_DIR, k)
    ltx, summ_df, result_dfs = basic_regressions(reg_df)
    with open(f"{BASE_TABLE_DIR}/prediction_{k}.tex", "w") as f:
        f.write(ltx)
    ltx1, ltx2 = incremental_regressions(reg_df)
    with open(f"{BASE_TABLE_DIR}/prediction_{k}_incremental_nber.tex", "w") as f:
        f.write(ltx1)
    with open(f"{BASE_TABLE_DIR}/prediction_{k}_incremental_vol.tex", "w") as f:
        f.write(ltx2)
    plot_prob_comparison(result_dfs, f"{BASE_TABLE_DIR}/plots/probit_reg_{k}.png")
    ltx, report_df = classification_task(reg_df)
    with open(f"{BASE_TABLE_DIR}/classification_{k}.tex", "w") as f:
        f.write(ltx)
    ltx, report_df = classification_task_is_oos(reg_df)
    with open(f"{BASE_TABLE_DIR}/classification_is_oos_{k}.tex", "w") as f:
        f.write(ltx)

    # Arbitrage
    arbitrage_simulated_data[SUFFIX_TAG_MAP[k]] = simulated_surfaces
print("{0:=^80}".format("Arbitrage"))
ltx = get_arbitrage_table(vol_surf_data[START_DAY:START_DAY+DAYS_TO_GENERATE], arbitrage_simulated_data)
with open(f"{BASE_TABLE_DIR}/arbitrage.tex", "w") as f:
    f.write(ltx)