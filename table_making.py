# modified from statsmodels iolib to allow different formatting for different items
from statsmodels.iolib.summary2 import Summary, _make_unique, _col_info, summary_params
from functools import reduce
import pandas as pd
import numpy as np

# Vertical summary instance for multiple models
def _col_params(result, float_format:dict={"coeff": '%.4f', "se": "%.2f", "R2": "%.3f"}, stars=True):
    """Stack coefficients and standard errors in single column
    """

    # Extract parameters
    res = summary_params(result)
    # Format float
    res.iloc[:, 0] = res.iloc[:, 0].apply(lambda x: float_format["coeff"] % x)
    res.iloc[:, 1] = res.iloc[:, 1].apply(lambda x: float_format["se"] % x)
    # Std.Errors in parentheses
    res.iloc[:, 1] = '(' + res.iloc[:, 1] + ')'
    # Significance stars
    if stars:
        idx = res.iloc[:, 3] < .1
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
        idx = res.iloc[:, 3] < .05
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
        idx = res.iloc[:, 3] < .01
        res.loc[idx, res.columns[0]] = res.loc[idx, res.columns[0]] + '*'
    # Stack Coefs and Std.Errors
    res = res.iloc[:, :2]
    res = res.stack()

    rsquared_adj = getattr(result, 'rsquared_adj', np.nan)
    r2 = pd.Series({('R-squared Adj.', ""): rsquared_adj})

    if r2.notnull().any():
        r2 = r2.apply(lambda x: float_format["R2"] % x)
        res = pd.concat([res, r2], axis=0)
    res = pd.DataFrame(res)
    res.columns = [str(result.model.endog_names)]
    return res

def summary_col(results, float_format:dict={"coeff": '%.4f', "se": "%.2f", "R2": "%.3f"}, model_names=(), stars=False,
                info_dict=None, regressor_order=(), drop_omitted=False):
    """
    Summarize multiple results instances side-by-side (coefs and SEs)

    Parameters
    ----------
    results : statsmodels results instance or list of result instances
    float_format : dict[str, str], optional, formatting for coefficient (default %.4f), standard error (default %.2f) and R^2 (default %.3f).
    model_names : list[str], optional
        Must have same length as the number of results. If the names are not
        unique, a roman number will be appended to all model names
    stars : bool
        print significance stars
    info_dict : dict, default None
        dict of functions to be applied to results instances to retrieve
        model info. To use specific information for different models, add a
        (nested) info_dict with model name as the key.
        Example: `info_dict = {"N":lambda x:(x.nobs), "R2": ..., "OLS":{
        "R2":...}}` would only show `R2` for OLS regression models, but
        additionally `N` for all other results.
        Default : None (use the info_dict specified in
        result.default_model_infos, if this property exists)
    regressor_order : list[str], optional
        list of names of the regressors in the desired order. All regressors
        not specified will be appended to the end of the list.
    drop_omitted : bool, optional
        Includes regressors that are not specified in regressor_order. If
        False, regressors not specified will be appended to end of the list.
        If True, only regressors in regressor_order will be included.
    """

    if not isinstance(results, list):
        results = [results]

    cols = [_col_params(x, stars=stars, float_format=float_format) for x in
            results]

    # Unique column names (pandas has problems merging otherwise)
    if model_names:
        colnames = _make_unique(model_names)
    else:
        colnames = _make_unique([x.columns[0] for x in cols])
    for i in range(len(cols)):
        cols[i].columns = [colnames[i]]

    def merg(x, y):
        return x.merge(y, how='outer', right_index=True,
                       left_index=True)

    summ = reduce(merg, cols)

    if regressor_order:
        varnames = summ.index.get_level_values(0).tolist()
        vc = pd.Series(varnames).value_counts()
        varnames = vc.loc[vc == 2].index.tolist()
        ordered = [x for x in regressor_order if x in varnames]
        unordered = [x for x in varnames if x not in regressor_order]
        new_order = ordered + unordered
        other = [x for x in summ.index.get_level_values(0)
                 if x not in new_order]
        new_order += other
        if drop_omitted:
            for uo in unordered:
                new_order.remove(uo)
        summ = summ.loc[new_order]

    idx = []
    index = summ.index.get_level_values(0)
    for i in range(0, index.shape[0], 2):
        idx.append(index[i])
        if (i + 1) < index.shape[0] and (index[i] == index[i + 1]):
            idx.append("")
        else:
            pass
            # idx.append(index[i + 1])
    summ.index = idx

    # add infos about the models.
    if info_dict:
        cols = [_col_info(x, info_dict.get(x.model.__class__.__name__,
                                           info_dict)) for x in results]
    else:
        cols = [_col_info(x, getattr(x, "default_model_infos", None)) for x in
                results]
    # use unique column names, otherwise the merge will not succeed
    for df, name in zip(cols, _make_unique([df.columns[0] for df in cols])):
        df.columns = [name]

    def merg(x, y):
        return x.merge(y, how='outer', right_index=True,
                       left_index=True)

    info = reduce(merg, cols)
    dat = pd.DataFrame(np.vstack([summ, info]))  # pd.concat better, but error
    dat.columns = summ.columns
    dat.index = pd.Index(summ.index.tolist() + info.index.tolist())
    summ = dat

    summ = summ.fillna('')

    smry = Summary()
    smry._merge_latex = True
    smry.add_df(summ, header=True, align='l')
    smry.add_text('Standard errors in parentheses.')
    if stars:
        smry.add_text('* p<.1, ** p<.05, ***p<.01')

    return smry
