import numpy as np
from vae.base import BaseVAE
from vae.cvae_with_mem import CVAEMem
from vae.cvae import CVAE
from typing import Union
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

def get_gradients_on_conditions(model: Union[CVAE, CVAEMem], conditions):
    '''
        conditions should have batch size = 1 or not batched
    '''
    if isinstance(conditions, dict):
        for k in conditions:
            conditions[k].requires_grad = True
        ex_feats_enabled = "ex_feats" in conditions
        if ex_feats_enabled:
            surface, ex_feats = model.get_surface_given_conditions(conditions)
        else:
            surface = model.get_surface_given_conditions(conditions)
        if len(conditions["surface"].shape) == 4:
            ctx_len = conditions["surface"].shape[1]
            pred_len = surface.shape[1]
            surface_sz1 = surface.shape[3]
            surface_sz2 = surface.shape[4]
            if ex_feats_enabled:
                ex_feats_sz = ex_feats.shape[2]
        else:
            ctx_len = conditions["surface"].shape[0]
            pred_len = surface.shape[0]
            surface_sz1 = surface.shape[2]
            surface_sz2 = surface.shape[3]
            if ex_feats_enabled:
                ex_feats_sz = ex_feats.shape[1]
        grad_mat_surface_surface = np.zeros((pred_len, surface_sz1, surface_sz2, ctx_len, surface_sz1, surface_sz2))
        surface_ctx = conditions["surface"]

        if ex_feats_enabled:
            grad_mat_ex_feats_surface = np.zeros((pred_len, ex_feats_sz, ctx_len, surface_sz1, surface_sz2))
            grad_mat_surface_ex_feats = np.zeros((pred_len, surface_sz1, surface_sz2, ctx_len, ex_feats_sz))
            grad_mat_ex_feats_ex_feats = np.zeros((pred_len, ex_feats_sz, ctx_len, ex_feats_sz))
            ex_feats_ctx = conditions["ex_feats"]

        for time in range(pred_len):
            for i in range(surface_sz1):
                for j in range(surface_sz2):
                    surface[0, time, i, j].backward(retain_graph=True)
                    grad_mat_surface_surface[time, i, j, :, :, :] = surface_ctx.grad[0].detach().cpu().numpy()
                    if ex_feats_enabled:
                        grad_mat_surface_ex_feats[time, i, j, :, :] = ex_feats_ctx.grad[0].detach().cpu().numpy()
                    surface_ctx.grad.zero_()
                    if ex_feats_enabled:
                        ex_feats_ctx.grad.zero_()
            if ex_feats_enabled:
                for feat in range(ex_feats_sz):
                    ex_feats[0, time, feat].backward(retain_graph=True)
                    grad_mat_ex_feats_surface[time, feat, :, :, :] = surface_ctx.grad[0].detach().cpu().numpy()
                    grad_mat_ex_feats_ex_feats[time, feat, :, :] = ex_feats_ctx.grad[0].detach().cpu().numpy()
                    surface_ctx.grad.zero_()
                    ex_feats_ctx.grad.zero_()

        if ex_feats_enabled:
            return {
                "surface-surface": grad_mat_surface_surface,
                "surface-ex_feat": grad_mat_surface_ex_feats,
                "ex_feat-surface": grad_mat_ex_feats_surface,
                "ex_feat-ex_feat": grad_mat_ex_feats_ex_feats,
            }
        else:
            return {"surface-surface": grad_mat_surface_surface}
    else:
        if len(conditions.shape) == 3:
            conditions = conditions.unsqueeze(0)
        conditions.requires_grad = True
        surface = model.get_surface_given_conditions(conditions)
        ctx_len = conditions.shape[1]
        pred_len = surface.shape[1]
        surface_sz1 = surface.shape[2]
        surface_sz2 = surface.shape[3]

        grad_mat_surface_surface = np.zeros((pred_len, surface_sz1, surface_sz2, ctx_len, surface_sz1, surface_sz2))
        for time in range(pred_len):
            for i in range(surface_sz1):
                for j in range(surface_sz2):
                    surface[0, time, i, j].backward(retain_graph=True)
                    grad_mat_surface_surface[time, i, j, :, :, :] = conditions.grad[0].detach().cpu().numpy()
                    conditions.grad.zero_()

        return {"surface-surface": grad_mat_surface_surface}
    
def plot_surface_surface_grad(surface_surface_grad):
    pred_len = surface_surface_grad.shape[0]
    sz0 = surface_surface_grad.shape[1]
    sz1 = surface_surface_grad.shape[2]
    ctx_len = surface_surface_grad.shape[3]
    nrows, ncols = pred_len * sz0, sz1
    fig, ax = plt.subplots(nrows=pred_len*sz0, ncols=sz1, subplot_kw={"projection": "3d"})

    moneyness_grid=[0.7, 0.85, 1, 1.15, 1.3]
    ttm_grid=[0.08333, 0.25, 0.5, 1, 2]
    ctx_grid = list(range(-ctx_len, 0))
    X, Y, Z = np.meshgrid(moneyness_grid, ttm_grid, ctx_grid)

    for pred_i in range(pred_len):
        for ttm in range(sz0):
            for moneyness in range(sz1):
                curr_ax = ax[pred_i*sz0 + ttm][moneyness]
                ctx_grad = surface_surface_grad[pred_i, ttm, moneyness, :, :, :]
                ctx_grad = np.transpose(ctx_grad, [1, 2, 0])
                color_map = cm.ScalarMappable(cmap=cm.Reds_r)
                color_map.set_array(ctx_grad)
                curr_ax.scatter(X, Y, Z, marker="s", s=40, color="green")
                # plt.colorbar(color_map, ax=curr_ax)
                curr_ax.set_title(f"Grad for pred={pred_i}, ttm={ttm_grid[ttm]:.2f}\nK/S={moneyness_grid[moneyness]:.2f}")
                curr_ax.set_xlabel("K/S")
                curr_ax.set_ylabel("ttm")
                curr_ax.set_zlabel("ctx")

    plt.subplots_adjust(right=3.5, top=3.5, hspace=0.3)
    plt.show()

def plot_ex_feat_grad(ex_feat_ex_feat_grad):
    pred_len = ex_feat_ex_feat_grad.shape[0]
    ctx_len = ex_feat_ex_feat_grad.shape[2]
    n_feats = ex_feat_ex_feat_grad.shape[1]
    fig, axes = plt.subplots(nrows=pred_len*n_feats)
    feat_names = ["ret"]
    ctx_labels = list(range(-ctx_len, 0))

    for pred_i in range(pred_len):
        for feat_i in range(n_feats):
            if pred_len*n_feats == 1:
                curr_ax = axes
            else:
                curr_ax = axes[pred_i*n_feats + feat_i]
            curr_grad = ex_feat_ex_feat_grad[pred_i][feat_i]
            if ctx_len > n_feats:
                sns.heatmap(curr_grad.T, ax=curr_ax, xticklabels=ctx_labels, yticklabels=feat_names)
                curr_ax.set_title(f"Grad for pred={pred_i}, {feat_names[feat_i]}")
                curr_ax.set_xlabel("ctx")
                curr_ax.set_ylabel("feats")
            else:
                sns.heatmap(curr_grad, ax=curr_ax, xticklabels=feat_names, yticklabels=ctx_labels)
                curr_ax.set_title(f"Grad for pred={pred_i}, {feat_names[feat_i]}")
                curr_ax.set_xlabel("feats")
                curr_ax.set_ylabel("ctx")
    plt.subplots_adjust(right=2.0, wspace=0.2, top=0.5, hspace=0.3)
    plt.show()