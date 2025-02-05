from typing import Dict

import numpy as np
import pandas as pd

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

moneyness = np.array([0.7, 0.85, 1, 1.15, 1.3])
ttm = np.array([0.08333, 0.25, 0.5, 1, 2])


def compute_arbitrage_conds(surfaces):
    ttm_reshaped = ttm.reshape((1, 1, len(ttm), 1))
    moneyness_reshaped = moneyness.reshape((1, 1, 1, len(moneyness)))
    total_iv_surf = ttm_reshaped * surfaces # w(ttm, K/S) = ttm * iv[ttm, K/S]
    # gradient calculation
    dw_dt = np.gradient(total_iv_surf, ttm, axis=-2, edge_order=2)
    dw_dx = np.gradient(total_iv_surf, moneyness, axis=-1, edge_order=2)
    dw_dx_sec = np.gradient(dw_dx, moneyness, axis=-1, edge_order=2)
    butterfly = (1.0 - moneyness_reshaped * dw_dx / (2 * total_iv_surf)) ** 2 \
        - dw_dx / 4 * (1 / total_iv_surf + 1/4) \
        + dw_dx_sec / 2
    
    return dw_dt, butterfly

def compare_fn(real: Dict[str, np.ndarray], simulated: Dict[str, Dict[str, np.ndarray]], ci=0.05):
    calendar_real = real["calendar"] # (num_day, 5, 5)
    butterfly_real = real["butterfly"] # (num_day, 5, 5)
    
    simulated_cols = list(simulated.keys())
    calendar_df = pd.DataFrame(index=list(range(8)), columns=["real"] + simulated_cols)
    butterfly_df = pd.DataFrame(index=list(range(8)), columns=["real"] + simulated_cols)

    # calendar
    neg_cal_real = np.where(calendar_real < 0, 1, 0) # (num_day, 5, 5)
    for count in range(8):
        non_cal_arbitrage_free_real = np.where(np.sum(np.sum(neg_cal_real, axis=-1), axis=-1) > count, 1, 0) # (num_day,)
        calendar_df.loc[count, "real"] = np.sum(non_cal_arbitrage_free_real)
    # butterfly
    neg_but_real = np.where(butterfly_real < 0, 1, 0) # (num_day, 5, 5)
    for count in range(8):
        non_but_free_real = np.where(np.sum(np.sum(neg_but_real, axis=-1), axis=-1) > count, 1, 0) # (num_day,)
        butterfly_df.loc[count, "real"] = np.sum(non_but_free_real)

    for sim_type, sim_res in simulated.items():
        calendar_sim = sim_res["calendar"] # (num_day, num_sim, 5, 5)
        butterfly_sim = sim_res["butterfly"] # (num_day, num_sim, 5, 5)
        thres = ci*butterfly_sim.shape[1]

        neg_cal_sim = np.where(calendar_sim < 0, 1, 0) # (num_day, num_sim, 5, 5)
        for count in range(8):
            non_cal_arbitrage_free_sim = np.where(np.sum(np.sum(neg_cal_sim, axis=-1), axis=-1) > count, 1, 0) # (num_day, num_sim)
            non_cal_arbitrage_free_sim = np.where(np.sum(non_cal_arbitrage_free_sim, axis=-1) > thres, 1, 0) # (num_day,)
            calendar_df.loc[count, sim_type] = np.sum(non_cal_arbitrage_free_sim) 

        neg_but_sim = np.where(butterfly_sim < 0, 1, 0) # (num_day, num_sim, 5, 5)
        for count in range(8):
            non_but_free_sim = np.where(np.sum(np.sum(neg_but_sim, axis=-1), axis=-1) > count, 1, 0) # (num_day, num_sim)
            non_but_free_sim = np.where(np.sum(non_but_free_sim, axis=-1) > thres, 1, 0) # (num_day,)
            butterfly_df.loc[count, sim_type] = np.sum(non_but_free_sim)

    return {
        "calendar": calendar_df,
        "butterfly": butterfly_df
    }

def get_arbitrage_table(real_data: np.ndarray, simulated_data: Dict[str, np.ndarray]):
    dw_dt_real, butterfly_real = compute_arbitrage_conds(real_data)
    real = {
        "calendar": dw_dt_real,
        "butterfly": butterfly_real,
    }

    sim = {}
    for k, v in simulated_data.items():
        dw_dt_sim, butterfly_sim = compute_arbitrage_conds(v)
        sim[k] = {
            "calendar": dw_dt_sim,
            "butterfly": butterfly_sim,
        }
    dfs = compare_fn(real, sim)
    d = {}
    d["Calendar"] = dfs["calendar"]
    d["Butterfly"] = dfs["butterfly"]
    count_df = pd.concat(d, axis=1)
    ltx = count_df.style.to_latex(column_format="l" + "c" * (2 + 2 * len(simulated_data)), multicol_align="c", hrules=True)
    return ltx