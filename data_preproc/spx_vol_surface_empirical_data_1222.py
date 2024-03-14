# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:46:08 2020

@author: jacky
"""
import scipy.optimize as opt
import pandas as pd
import os
import datetime as dt
import numpy as np

def objective_function(x,weight,implied_vol):       
        return sum((np.sum(weight * x,axis = 1) - implied_vol)**2)

def ineq_constraint(x):
    """constrain all elements of x to be >= 0"""
    return x

def value_above(num,standard_value):
    cloest_diff = min([value - num for value in standard_value if (value - num) >0])
    return standard_value[[value - num for value in standard_value].index(cloest_diff)]

def value_below(num,standard_value):
    cloest_diff = max([value - num for value in standard_value if (value - num) <0])

    return standard_value[[value - num for value in standard_value].index(cloest_diff)]


path = "./Data/"

df = pd.read_pickle(os.path.join(path,"Option_Data.pkl"))

# %%
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
    subset["ttm"] = subset["exdate"] - subset["date"]
    subset["ttm"] = subset["ttm"].apply(lambda x: x.days/365)    
    subset["moneyness"] = (subset["strike_price"])/(subset["Close"]*1000)
    subset = subset[(subset.ttm > 0.08333)&(subset.ttm<2)]
    subset = subset[(subset.moneyness > 0.7)&(subset.moneyness<1.3)]

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
    
    
output.to_csv("spx_vol_surface_history.csv")
output["indicator"] = ((output.iloc[:,:25] < 0.05)|(output.iloc[:,:25] > 1)).sum(axis = 1)
output = output[output["indicator"] ==0]
output.to_csv("spx_vol_surface_history_cleaned.csv")
output1 = output[output["observation"] >300]

output = pd.read_csv("spx_vol_surface_history_full_data_1222.csv",index_col = "date")