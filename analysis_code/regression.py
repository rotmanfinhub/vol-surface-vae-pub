import numpy as np
import pandas as pd
import yfinance
import statsmodels.formula.api as smf
import statsmodels.api as smi
from table_making import summary_col
import matplotlib.pyplot as plt
import re
import os
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


cols = [
    [
        "ttm_one_month_moneyness_pt_seven",
        "ttm_one_month_moneyness_pt_eightfive",
        "ttm_one_month_moneyness_pt_one",
        "ttm_one_month_moneyness_pt_oneonefive",
        "ttm_one_month_moneyness_pt_onethree",
    ],
    [
        "ttm_three_month_moneyness_pt_seven",
        "ttm_three_month_moneyness_pt_eightfive",
        "ttm_three_month_moneyness_pt_one",
        "ttm_three_month_moneyness_pt_oneonefive",
        "ttm_three_month_moneyness_pt_onethree",
    ],
    [
        "ttm_six_month_moneyness_pt_seven",
        "ttm_six_month_moneyness_pt_eightfive",
        "ttm_six_month_moneyness_pt_one",
        "ttm_six_month_moneyness_pt_oneonefive",
        "ttm_six_month_moneyness_pt_onethree",
    ],
    [
        "ttm_one_year_moneyness_pt_seven",
        "ttm_one_year_moneyness_pt_eightfive",
        "ttm_one_year_moneyness_pt_one",
        "ttm_one_year_moneyness_pt_oneonefive",
        "ttm_one_year_moneyness_pt_onethree",
    ],   
    [
        "ttm_two_year_moneyness_pt_seven",
        "ttm_two_year_moneyness_pt_eightfive",
        "ttm_two_year_moneyness_pt_one",
        "ttm_two_year_moneyness_pt_oneonefive",
        "ttm_two_year_moneyness_pt_onethree",
    ],
]

def latex_format(sm, row=0, delta=False, mean=True):
    ltx = sm.as_latex()
    ltx = re.sub(r" +", " ", ltx)
    row_correspondence = ["1/12", "1/4", "1/2", "1", "2"]
    col_correspondence = ["0.7", "0.85", "1", "1.15", "1.3"]
    row_val = row_correspondence[row]
    models = ""
    if delta:
        for col in col_correspondence:
            models += f"& $|\\Delta x_t[{row_val}, {col}]|$ "
    else:
        for col in col_correspondence:
            models += f"& $x_t[{row_val}, {col}]$ "
    ltx = ltx.replace(" & col0 & col1 & col2 & col3 & col4 \\\\\n\\hline", " " + models + """ \\\\
 & (1) & (2)  & (3) & (4) & (5) \\\\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} \cmidrule(lr){4-4} \cmidrule(lr){5-5} \cmidrule(lr){6-6}""")
    ltx = ltx.replace(" & log\_ret & abs\_ret \\\\\n\\hline", """ & $r_t$ & $|\\Delta r_t|$ \\\\
 & (1) & (2) \\\\
\cmidrule(lr){2-2} \cmidrule(lr){3-3} """)
    ltx = ltx.replace("ret\_std", r"$\sigma(\hat{r_t})$")
    ltx = ltx.replace("ret\_mean", r"$\mu(\hat{r_t})$")
    if mean:
        ltx = ltx.replace("iv", r"$\mu(\hat{x})$")
    else:
        ltx = ltx.replace("iv", r"$\sigma(\hat{x})$")
    ltx = ltx.replace("R-squared Adj.", r"\\hline\n$R^2$")
    ltx = ltx.replace("\\$N\\$", r"$N$")
    ltx = ltx.replace("""llllll""", """lccccc""")
    ltx = ltx.replace("lll", "lcc")
    ltx = ltx.replace("cc}\n\\hline", "cc}\n\\hline\\hline")
    ltx = ltx.replace("\\begin{table}\n\\caption{}\n\\label{}\n\\begin{center}\n", "")
    ltx = ltx.replace("\\end{center}\n\\end{table}", "")
    return ltx

def latex_format_grid_points(df: pd.DataFrame):
    ltx = df.style.to_latex(column_format="l" + "c" * len(df.columns))
    ltx = ltx.replace("nan", " ")
    ltx = ltx.replace("K/S=1.3 \\\\\n", "K/S=1.3 \\\\\n\\cmidrule{2-6}\n")
    ltx = ltx.replace("cc}\n", "cc}\n\\toprule\n")
    ltx = ltx.replace("\\\\\n\\end{tabular}", "\\\\\n\\bottomrule\n\\end{tabular}")
    return ltx


def regression_surface_grids_mle(simulated_surfaces, spx_df: pd.DataFrame, ctx_len=3, days_to_generate = 5819,
                             base_folder="tables/iv_all", model_type="loss"):
    os.makedirs(f"{base_folder}", exist_ok=True)
    df_cols = ["K/S=0.7", "K/S=0.85", "K/S=1", "K/S=1.15", "K/S=1.3"]
    df_rows = ["1 month", "3 month", "6 month", "1 year", "2 year"]
    val_df_coeff = pd.DataFrame(index=df_rows, columns=df_cols)
    val_df_const = pd.DataFrame(index=df_rows, columns=df_cols)
    val_df_r2 = pd.DataFrame(index=df_rows, columns=df_cols)

    for row in range(5):
        models = []
        for col in range(5):
            curr_grid = cols[row][col]
            regression_df = spx_df.copy()
            regression_df[f"delta_{curr_grid}"] = np.abs(regression_df[curr_grid] - regression_df[curr_grid].shift(1))
            regression_df = regression_df.loc[ctx_len:ctx_len+days_to_generate-1, ["date", curr_grid, f"delta_{curr_grid}"]].reset_index(drop=True)
            regression_df1 = regression_df.copy()
            regression_df1["iv"] = np.mean(simulated_surfaces[:, :, row, col], axis=1)

            model = smf.ols(f"{curr_grid}~1+iv", data=regression_df1).fit(cov_type="HC3")
            models.append(model)

            model_iv_param = model.params["iv"]
            model_iv_pval = model.pvalues["iv"]
            model_iv_param_str = "%.4f" % model_iv_param
            if model_iv_pval < .1:
                model_iv_param_str += "*"
            if model_iv_pval < .05:
                model_iv_param_str += "*"
            if model_iv_pval < .01:
                model_iv_param_str += "*"

            model_const_param = model.params["Intercept"]
            model_const_pval = model.pvalues["Intercept"]
            model_const_param_str = "%.4f" % model_const_param
            if model_const_pval < .1:
                model_const_param_str += "*"
            if model_const_pval < .05:
                model_const_param_str += "*"
            if model_const_pval < .01:
                model_const_param_str += "*"

            model_rsquared_adj = "%.3f" % model.rsquared_adj
            val_df_coeff.loc[df_rows[row], df_cols[col]] = model_iv_param_str
            val_df_const.loc[df_rows[row], df_cols[col]] = model_const_param_str
            val_df_r2.loc[df_rows[row], df_cols[col]] = model_rsquared_adj

    with open(f"{base_folder}/reg_{model_type}_mle_coeff.tex", "w") as f:
        f.write(latex_format_grid_points(val_df_coeff))
    with open(f"{base_folder}/reg_{model_type}_mle_const.tex", "w") as f:
        f.write(latex_format_grid_points(val_df_const))
    with open(f"{base_folder}/reg_{model_type}_mle_r2.tex", "w") as f:
        f.write(latex_format_grid_points(val_df_r2))

def regression_surface_grids_std(simulated_surfaces, spx_df: pd.DataFrame, ctx_len=3, days_to_generate = 5819,
                             base_folder="tables/iv_all", model_type="loss"):
    os.makedirs(f"{base_folder}", exist_ok=True)
    df_cols = ["K/S=0.7", "K/S=0.85", "K/S=1", "K/S=1.15", "K/S=1.3"]
    df_rows = ["1 month", "3 month", "6 month", "1 year", "2 year"]
    delta_df_coeff = pd.DataFrame(index=df_rows, columns=df_cols)
    delta_df_const = pd.DataFrame(index=df_rows, columns=df_cols)
    delta_df_r2 = pd.DataFrame(index=df_rows, columns=df_cols)

    for row in range(5):
        delta_models = []
        for col in range(5):
            curr_grid = cols[row][col]
            regression_df = spx_df.copy()
            regression_df[f"delta_{curr_grid}"] = np.abs(regression_df[curr_grid] - regression_df[curr_grid].shift(1))
            regression_df = regression_df.loc[ctx_len:ctx_len+days_to_generate-1, ["date", curr_grid, f"delta_{curr_grid}"]].reset_index(drop=True)
            regression_df1 = regression_df.copy()
            regression_df1["iv"] = np.std(simulated_surfaces[:, :, row, col], axis=1)

            delta_model = smf.ols(f"delta_{curr_grid}~1+iv", data=regression_df1).fit(cov_type="HC3")
            delta_models.append(delta_model)

            delta_model_iv_param = delta_model.params["iv"]
            delta_model_iv_pval = delta_model.pvalues["iv"]
            delta_model_iv_param_str = "%.4f" % delta_model_iv_param
            if delta_model_iv_pval < .1:
                delta_model_iv_param_str += "*"
            if delta_model_iv_pval < .05:
                delta_model_iv_param_str += "*"
            if delta_model_iv_pval < .01:
                delta_model_iv_param_str += "*"

            delta_model_const_param = delta_model.params["Intercept"]
            delta_model_const_pval = delta_model.pvalues["Intercept"]
            delta_model_const_param_str = "%.4f" % delta_model_const_param
            if delta_model_const_pval < .1:
                delta_model_const_param_str += "*"
            if delta_model_const_pval < .05:
                delta_model_const_param_str += "*"
            if delta_model_const_pval < .01:
                delta_model_const_param_str += "*"
            
            delta_model_rsquared_adj = "%.3f" % delta_model.rsquared_adj
            delta_df_coeff.loc[df_rows[row], df_cols[col]] = delta_model_iv_param_str
            delta_df_const.loc[df_rows[row], df_cols[col]] = delta_model_const_param_str
            delta_df_r2.loc[df_rows[row], df_cols[col]] = delta_model_rsquared_adj

        with open(f"{base_folder}/reg_{model_type}_delta_coeff.tex", "w") as f:
            f.write(latex_format_grid_points(delta_df_coeff))
        with open(f"{base_folder}/reg_{model_type}_delta_const.tex", "w") as f:
            f.write(latex_format_grid_points(delta_df_const))
        with open(f"{base_folder}/reg_{model_type}_delta_r2.tex", "w") as f:
            f.write(latex_format_grid_points(delta_df_r2))

def plot_surface_grids(simulated_surfaces, spx_df: pd.DataFrame, 
                       ctx_len=3, days_to_generate = 5819,
                       base_folder="tables/iv_all", model_type="loss"):
    os.makedirs(f"{base_folder}/plots", exist_ok=True)
    fig, ax = plt.subplots(5, 5, figsize=(30, 30))
    # fig_std, ax_std = plt.subplots(5, 5, figsize=(30, 30))
    df_cols = ["K/S=0.7", "K/S=0.85", "K/S=1", "K/S=1.15", "K/S=1.3"]
    df_rows = ["1 month", "3 month", "6 month", "1 year", "2 year"]
    for row in range(5):
        for col in range(5):
            curr_grid = cols[row][col]
            regression_df = spx_df.copy()
            regression_df[f"delta_{curr_grid}"] = np.abs(regression_df[curr_grid] - regression_df[curr_grid].shift(1))
            regression_df = regression_df.loc[ctx_len:ctx_len+days_to_generate-1, ["date", curr_grid, f"delta_{curr_grid}"]].reset_index(drop=True)
            regression_df1 = regression_df.copy()
            regression_df1["iv"] = np.mean(simulated_surfaces[:, :, row, col], axis=1)

            model = smf.ols(f"{curr_grid}~1+iv", data=regression_df1).fit(cov_type="HC3")
            regression_df1[f"{curr_grid}_pred"] = model.predict(regression_df1)
            x = np.linspace(0, 1, 50)
            ax[row][col].scatter(regression_df1["iv"], regression_df1[curr_grid], alpha=0.3, s=10, color="blue")
            ax[row][col].plot(list(regression_df1["iv"]) + [0], 
                              list(regression_df1[f"{curr_grid}_pred"]) + [model.params["Intercept"]], 
                              color="orange", linewidth=5)
            ax[row][col].plot(x, x, color="red", linestyle="-.", linewidth=5)
            ax[row][col].set_title(f"Mean: {df_rows[row]}, {df_cols[col]}")
            ax[row][col].set_xlabel("Generated")
            ax[row][col].set_ylabel("Realized")
    plt.tight_layout()
    fig.savefig(f"{base_folder}/plots/reg_mle_{model_type}.jpg")
    plt.close()
                
def regression_return(simulated_returns_mle, simulated_returns, spx_df: pd.DataFrame, 
                       ctx_len=3, days_to_generate = 5819, 
                      base_folder="tables/iv_all"):
    os.makedirs(f"{base_folder}", exist_ok=True)
    stock_price = pd.read_csv("data/GSPC.csv").rename(columns={"Date": "date"})
    stock_price["date"] = pd.to_datetime(stock_price["date"])
    stock_price["log_ret"] = np.log(stock_price["Adj Close"]) - np.log(stock_price["Adj Close"].shift(1))
    stock_price.loc[0, "log_ret"] = 0
    stock_price["abs_ret"] = np.abs(stock_price["log_ret"])
    regression_df = pd.merge(spx_df.loc[ctx_len:ctx_len+days_to_generate-1, ["date"]], stock_price, how="left", on="date")[["date", "log_ret", "abs_ret"]].reset_index(drop=True)
    regression_df1 = regression_df.copy()
    regression_df1["ret_mean"] = np.mean(simulated_returns_mle, axis=1)
    regression_df1["ret_std"] = np.std(simulated_returns, axis=1)
    
    models = []
    for y in ["log_ret", "abs_ret"]:
        model = smf.ols(f"{y}~1+ret_std", data=regression_df1).fit(cov_type="HC3")
        models.append(model)
    sm = summary_col(models, model_names=["log_ret", "abs_ret"], stars=True,
                            regressor_order=["ret_std", "Intercept"],
                            info_dict={'$N$':lambda x: "{0:d}".format(int(x.nobs))})
    with open(f"{base_folder}/reg_std_loss_ret.tex", "w") as f:
        f.write(latex_format(sm))

    models = []
    for y in ["log_ret", "abs_ret"]:
        model = smf.ols(f"{y}~1+ret_mean", data=regression_df1).fit(cov_type="HC3")
        models.append(model)
    sm = summary_col(models, model_names=["log_ret", "abs_ret"], stars=True,
                            regressor_order=["ret_mean", "Intercept"],
                            info_dict={'$N$':lambda x: "{0:d}".format(int(x.nobs))})
    with open(f"{base_folder}/reg_mle_loss_ret.tex", "w") as f:
        f.write(latex_format(sm))

def benchmark_with_rmse(simulated_surfaces, spx_df: pd.DataFrame, 
                       ctx_len=3, days_to_generate = 5819,
                             base_folder="tables/iv_all", model_type="loss"):
    os.makedirs(base_folder, exist_ok=True)
    df_cols = ["K/S=0.7", "K/S=0.85", "K/S=1", "K/S=1.15", "K/S=1.3"]
    df_rows = ["1 month", "3 month", "6 month", "1 year", "2 year"]
    rmse_df = pd.DataFrame(index=df_rows, columns=df_cols)
    for row in range(5):
        for col in range(5):
            curr_grid = cols[row][col]
            regression_df = spx_df.copy()
            regression_df["iv_prev_day"] = regression_df[curr_grid].shift(1)
            regression_df = regression_df.loc[ctx_len:ctx_len+days_to_generate-1, ["date", curr_grid, "iv_prev_day"]].rename(columns={curr_grid: "iv"}).reset_index(drop=True)
            regression_df1 = regression_df.copy()
            regression_df1["iv_mean"] = np.mean(simulated_surfaces[:, :, row, col], axis=1)
            
            rmse = np.sqrt(np.mean((regression_df1["iv"] - regression_df1["iv_mean"])**2))
            rmse_df.loc[df_rows[row], df_cols[col]] = "%.4f" % rmse
            
    with open(f"{base_folder}/rmse_{model_type}.tex", "w") as f:
        f.write(latex_format_grid_points(rmse_df))