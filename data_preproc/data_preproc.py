import pandas as pd
import numpy as np
import scipy.optimize as opt
import yfinance as yf
from datetime import date

def clean_up_and_merge_dfs(df, stock_price):
    print(df["secid"].unique()) # [108105]
    print(df["cp_flag"].unique()) # ['P' 'C']
    print(df["cfadj"].unique()) # [1]
    print(df["ss_flag"].unique()) # [0]
    print(df["index_flag"].unique()) # [1]
    print(df["issuer"].unique()) # ['CBOE S&P 500 INDEX']
    print(df["exercise_style"].unique()) # ['E']
    print(df["am_set_flag"].unique()) # [nan]

    df = df.copy()

    df = df[(df["cp_flag"] == "C") & (df["cfadj"] == 1) & (df["ss_flag"] == 0) 
        & (df["index_flag"] == 1) & (df["exercise_style"] == "E")]
    # reduce the size of the raw dataframe by dropping columns with single unique values
    df = df.drop(columns=["secid", "cp_flag", "cfadj", "ss_flag", 
                        "index_flag", "issuer", "index_flag", "exercise_style", "am_set_flag"])
    trading_days = df[["date"]].drop_duplicates().sort_values(by=["date"])
    trading_days["next_date"] = trading_days["date"].shift(-1)
    print(trading_days.head(10))
    print(trading_days.tail(10))
    trading_days = trading_days.dropna()

    # drop rows if impl_volatility or delta is NaN
    df = df.dropna(subset=['impl_volatility', 'delta'])

    # work on the raw data a bit more
    df["mid_market"] = (df.loc[:, "best_bid"] + df.loc[:, "best_offer"]) / 2

    # calculate day_to_maturity; all options are AM settled, so minus 1 more day
    df["day_to_maturity"] = (df["exdate"] - df["date"]).dt.days - 1
    print(df["day_to_maturity"])

    # merge with trading_days to get next_date
    df = df.merge(trading_days, left_on='date', right_on='date', how='left')
    df = df.dropna(subset=["next_date"])

    # self-join to pair two rows with the same optionid and left_next_date=right_date (i.e. two consecutive trading days)
    df = df.merge(df, left_on=['optionid', 'next_date'], right_on=['optionid', 'date'], how='left', suffixes=('', '_next'))
    df = df.dropna(subset=['date_next'])
    df = df.drop(columns=['date_next', 'exdate_next', 'strike_price_next', 'best_bid_next', 'best_offer_next',
                            'day_to_maturity_next', 'next_date_next'])
    
    df = df.merge(stock_price, on="date")
    df = df[(df["delta"] >= 0.05)&(df["delta"] <= 0.95)&(df['day_to_maturity'] >= 14)]
    df['index_pc_change'] = df['return']
    df['time_to_maturity'] = df['day_to_maturity'] / 250
    df['moneyness'] = df['delta']
    df['actual_change'] = df['impl_volatility_next'] - df['impl_volatility']

    df['x1'] = df['index_pc_change'] / np.sqrt(df['time_to_maturity'])
    df['x2'] = df['index_pc_change'] / np.sqrt(df['time_to_maturity']) * df['moneyness']
    df['x3'] = df['x2'] * df['moneyness']
    return df

def objective_function(x,weight,implied_vol):       
        return sum((np.sum(weight * x,axis = 1) - implied_vol)**2)

def ineq_constraint(x):
    """constrain all elements of x to be >= 0"""
    return x

def cal_arb_constraint(x):
    '''Let w(ttm, moneyness) = ttm* iv(ttm, moneyness), constrain dw/dt >= 0'''
    moneyness = np.array([0.7, 0.85, 1, 1.15, 1.3])
    ttm = np.array([0.08333, 0.25, 0.5, 1, 2])
    ttm_reshape = ttm.reshape(len(ttm), 1)
    w = ttm_reshape * x.reshape((len(ttm), len(moneyness)))
    dw_dt = np.gradient(w, ttm, axis=0, edge_order=2)
    return dw_dt.reshape(len(ttm)*len(moneyness))

def but_arb_constraint(x):
    '''Let w(ttm, moneyness) = ttm* iv(ttm, moneyness), 
    constrain g(ttm, moneyness) = (1-(moneyness*dw/dx)/(2w))^2 -(dw/dx)/4 * (1/w+1/4) + d^2w/dx^2/2 >= 0'''
    moneyness = np.array([0.7, 0.85, 1, 1.15, 1.3])
    ttm = np.array([0.08333, 0.25, 0.5, 1, 2])
    ttm_reshape = ttm.reshape(len(ttm), 1)
    w = ttm_reshape * x.reshape((len(ttm), len(moneyness)))
    dw_dt = np.gradient(w, ttm, axis=0, edge_order=2)
    dw_dx = np.gradient(w, moneyness, axis=-1, edge_order=2)
    dw_dx_sec = np.gradient(dw_dx, moneyness, axis=-1, edge_order=2)
    moneyness_reshaped = moneyness.reshape((1, len(moneyness)))
    butterfly = (1.0 - moneyness_reshaped * dw_dx / (2 * (w+1e-8))) ** 2 \
        - dw_dx / 4 * (1 / (w+1e-8) + 1/4) \
        + dw_dx_sec / 2
    return butterfly.reshape(len(ttm)*len(moneyness))

def value_above(num,standard_value):
    cloest_diff = min([value - num for value in standard_value if (value - num) >0])
    return standard_value[[value - num for value in standard_value].index(cloest_diff)]

def value_below(num,standard_value):
    cloest_diff = max([value - num for value in standard_value if (value - num) <0])

    return standard_value[[value - num for value in standard_value].index(cloest_diff)]

def process(df, cal_arb_cons=False, but_arb_cons=False):
    date_range = df.date.unique()

    standard_maturity = [0.08333,0.25,0.5,1,2]
    standard_moneyness = [0.7,0.85,1,1.15,1.3]

    maturity_columns = ["one_month","three_month","six_month","one_year","two_year"]
    moneyness_columns = ["point_seven","point_eight_five","one","one_point_one_five","one_point_three"]
    maturity_numpy = [0,1,2,3,4]
    moneyness_numpy = [0,1,2,3,4]

    col = ["ttm_one_month_moneyness_pt_seven","ttm_one_month_moneyness_pt_eightfive",
            "ttm_one_month_moneyness_pt_one","ttm_one_month_moneyness_pt_oneonefive",
            "ttm_one_month_moneyness_pt_onethree","ttm_three_month_moneyness_pt_seven",
            "ttm_three_month_moneyness_pt_eightfive","ttm_three_month_moneyness_pt_one",
            "ttm_three_month_moneyness_pt_oneonefive","ttm_three_month_moneyness_pt_onethree",
            "ttm_six_month_moneyness_pt_seven",
            "ttm_six_month_moneyness_pt_eightfive","ttm_six_month_moneyness_pt_one",
            "ttm_six_month_moneyness_pt_oneonefive","ttm_six_month_moneyness_pt_onethree",
            "ttm_one_year_moneyness_pt_seven",
            "ttm_one_year_moneyness_pt_eightfive","ttm_one_year_moneyness_pt_one",
            "ttm_one_year_moneyness_pt_oneonefive","ttm_one_year_moneyness_pt_onethree",   
            "ttm_two_year_moneyness_pt_seven",
            "ttm_two_year_moneyness_pt_eightfive","ttm_two_year_moneyness_pt_one",
            "ttm_two_year_moneyness_pt_oneonefive","ttm_two_year_moneyness_pt_onethree","r_squared","mean_error","mean_absolute_error","observation"]

    output = pd.DataFrame(np.zeros((len(date_range),len(standard_maturity)*len(standard_moneyness)+4)),columns = col,index = date_range)

    for date in date_range:
        print(date)
        fitted_point = np.zeros((len(standard_maturity),len(standard_moneyness)))
        subset = df[df.date == date].dropna(axis = 1, how = "all")
        subset = subset.dropna(axis = 0, how = "all")
        subset = subset[subset['impl_volatility'].notna()]
        subset = subset[subset.open_interest != 0]
        if len(subset) == 0:
            output.loc[date] = np.NaN
            continue
        subset["ttm"] = subset["exdate"] - subset["date"]
        subset["ttm"] = subset["ttm"].apply(lambda x: x.days/365)
        subset["moneyness"] = (subset["strike_price"])/(subset["Close"]*1000)
        subset = subset[(subset["ttm"] > 0.08333)&(subset["ttm"]<2)]
        subset = subset[(subset["moneyness"] > 0.7)&(subset["moneyness"]<1.3)]

        subset["standard_value_below_maturity"] = [value_below(maturity,standard_maturity) for maturity in subset["ttm"] ]
        subset["standard_value_above_maturity"] = [value_above(maturity,standard_maturity) for maturity in subset["ttm"] ]
        
        subset["weight_below_maturity"] = (subset["standard_value_above_maturity"] - subset["ttm"])/(subset["standard_value_above_maturity"] - subset["standard_value_below_maturity"])
        subset["weight_above_maturity"] = 1-subset["weight_below_maturity"] 
        
        subset["standard_value_below_moneyness"] = [value_below(moneyness,standard_moneyness) for moneyness in subset["moneyness"] ]
        subset["standard_value_above_moneyness"] = [value_above(moneyness,standard_moneyness) for moneyness in subset["moneyness"] ]
        subset["weight_below_moneyness"] = (subset["standard_value_above_moneyness"] - subset["moneyness"])/(subset["standard_value_above_moneyness"] - subset["standard_value_below_moneyness"])
        subset["weight_above_moneyness"] = 1-subset["weight_below_moneyness"] 
        
        subset["below_below"] = subset["weight_below_maturity"]*subset["weight_below_moneyness"] 
        subset["below_above"] = subset["weight_below_maturity"]*subset["weight_above_moneyness"]
        subset["above_below"] = subset["weight_above_maturity"]*subset["weight_below_moneyness"]
        subset["above_above"] = subset["weight_above_maturity"]*subset["weight_above_moneyness"]
        
        subset["standard_value_below_maturity_lookup"] = [maturity_numpy[standard_maturity.index(maturity)] for maturity in subset["standard_value_below_maturity"] ]
        subset["standard_value_above_maturity_lookup"] = [maturity_numpy[standard_maturity.index(maturity)] for maturity in subset["standard_value_above_maturity"] ] 
        subset["standard_value_below_moneyness_lookup"] = [moneyness_numpy[standard_moneyness.index(moneyness)] for moneyness in subset["standard_value_below_moneyness"] ] 
        subset["standard_value_above_moneyness_lookup"] = [moneyness_numpy[standard_moneyness.index(moneyness)] for moneyness in subset["standard_value_above_moneyness"] ] 
        
        weight = np.zeros((len(subset["standard_value_below_maturity_lookup"]),len(standard_maturity)*len(standard_moneyness),))
        
        for i in range(len(subset["standard_value_below_maturity_lookup"])):
            fitted_point = np.zeros((len(standard_maturity),len(standard_moneyness)))
            fitted_point[subset["standard_value_below_maturity_lookup"].values[i],subset["standard_value_below_moneyness_lookup"].values[i]] = subset["below_below"].values[i]
            fitted_point[subset["standard_value_below_maturity_lookup"].values[i],subset["standard_value_above_moneyness_lookup"].values[i]] = subset["below_above"].values[i]
            fitted_point[subset["standard_value_above_maturity_lookup"].values[i],subset["standard_value_below_moneyness_lookup"].values[i]] = subset["above_below"].values[i]
            fitted_point[subset["standard_value_above_maturity_lookup"].values[i],subset["standard_value_above_moneyness_lookup"].values[i]] = subset["above_above"].values[i]
            weight[i,:] = fitted_point.flatten()
        
        from numpy import random
        x = random.rand(len(standard_maturity)*len(standard_moneyness))
        x = np.zeros(len(standard_maturity)*len(standard_moneyness))
        
        cons=[{'type': 'ineq', 'fun': ineq_constraint}]
        if cal_arb_cons:
            cons.append({'type': 'ineq', 'fun': cal_arb_constraint})
        
        if but_arb_cons:
            cons.append({'type': 'ineq', 'fun': but_arb_constraint})
        
        objective_function(x,weight,subset["impl_volatility"].values)
        res = opt.minimize(objective_function,x, method='SLSQP',args = (weight,subset["impl_volatility"].values),constraints = cons)
        solution = res.x
        # solution = res.x.reshape(5,5)
        
        count = len(subset)
        r_squared = 1 - np.var(np.sum(weight * res.x,axis = 1) - subset["impl_volatility"].values)/np.var(subset["impl_volatility"].values)
        mean_error = np.average(np.sum(weight * res.x,axis = 1) - subset["impl_volatility"].values)
        mean_abs_error = np.average(abs(np.sum(weight * res.x,axis = 1) - subset["impl_volatility"].values))
        solution = np.append(solution,[r_squared,mean_error,mean_abs_error,count])
        output.loc[date] = solution
    return output