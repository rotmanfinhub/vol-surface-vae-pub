import random, time, os, json
import numpy as np
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
from tqdm import tqdm
from vae.base import BaseVAE
from vae.cvae_with_mem import CVAEMem
from vae.cvae import CVAE
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
from typing import Union

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(model: BaseVAE, dataloader):
    model.eval() # switch to eval model, will turn off randomness like dropout
    eval_losses = defaultdict(float)
    num_batches = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f"eval")):
        try:
            batch.to(model.device)
        except:
            pass

        losses = model.test_step(batch)

        for k, v in losses.items():
            eval_losses[k] += v.item()
        num_batches += 1

    for k, v in eval_losses.items():
        eval_losses[k] = v / num_batches

    return eval_losses
    

def train(model: BaseVAE, train_dataloader: DataLoader, valid_dataloader: DataLoader, 
          lr=1e-5, epochs=100, 
          model_dir="./", file_name="vanilla.pt"):
    model.train()
    optimizer = opt.AdamW(model.parameters(), lr)
    best_dev_loss = np.inf

    ## run for the specified number of epochs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if "." in file_name:
        file_prefix = file_name.split(".")[0]
    else:
        file_prefix = file_name
    log_file = open(f"{model_dir}/{file_prefix}-{epochs}-log.txt", "w", encoding="utf-8")

    print("Model config: ", file=log_file)
    print(json.dumps(model.config, indent=True), file=log_file)
    print(f"LR: {lr}", file=log_file)
    print(f"Epochs: {epochs}", file=log_file)
    print("", file=log_file)
    log_file.flush()
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"train-{epoch}")):
            try:
                batch.to(model.device)
            except:
                pass

            losses = model.train_step(batch, optimizer)

            for k, v in losses.items():
                train_losses[k] += v.item()
            num_batches += 1
        for k, v in train_losses.items():
            train_losses[k] = v / (num_batches)
        
        dev_losses = model_eval(model, valid_dataloader)

        if dev_losses["loss"] < best_dev_loss:
            best_dev_loss = dev_losses["loss"]
            model.save_weights(optimizer, model_dir, file_prefix)

        formatted_train_loss = ", ".join([f'{k}: {v:.3f}' for k, v in train_losses.items()])
        formatted_dev_loss = ", ".join([f'{k}: {v:.3f}' for k, v in dev_losses.items()])
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}")
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}", file=log_file)
    print(f"training finished, total time :: {time.time() - start_time}")
    print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
    log_file.close()
    return train_losses, dev_losses

def test(model: BaseVAE, valid_dataloader: DataLoader, test_dataloader: DataLoader, model_fn="./vanilla"):
    model.load_weights(f=model_fn)
    dev_losses = model_eval(model, valid_dataloader)
    test_losses = model_eval(model, test_dataloader)

    formatted_dev_loss = ", ".join([f'{k}: {v:.3f}' for k, v in dev_losses.items()])
    formatted_test_loss = ", ".join([f'{k}: {v:.3f}' for k, v in test_losses.items()])
    print(f"dev loss: {formatted_dev_loss}, \ntest_loss: {formatted_test_loss}")
    return dev_losses, test_losses

def plot_surface(original_data: np.ndarray, vae_output: np.ndarray):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = [0.7,0.85,1,1.15,1.3]
    y = [0.08333,0.25,0.5,1,2]
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, original_data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.plot_surface(x, y, vae_output, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("K/S")
    ax.set_ylabel("ttm")
    plt.show()

def plot_surface_separate(original_data: np.ndarray, vae_output: np.ndarray):
    fig, ax = plt.subplots(nrows=1, ncols=3, subplot_kw={"projection": "3d"})
    x = [0.7,0.85,1,1.15,1.3]
    y = [0.08333,0.25,0.5,1,2]
    x, y = np.meshgrid(x, y)

    ax[0].plot_surface(x, y, original_data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax[0].set_xlabel("K/S")
    ax[0].set_ylabel("ttm")
    ax[0].set_title("Original surface")
    
    ax[1].plot_surface(x, y, vae_output, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax[1].set_xlabel("K/S")
    ax[1].set_ylabel("ttm")
    ax[1].set_title("VAE surface")
    
    ax[2].plot_surface(x, y, original_data, cmap=cm.Blues, linewidth=0, antialiased=False)
    ax[2].plot_surface(x, y, vae_output, cmap=cm.Reds, linewidth=0, antialiased=False)
    ax[2].set_xlabel("K/S")
    ax[2].set_ylabel("ttm")
    ax[2].set_title("Both surfaces")
    
    plt.subplots_adjust(right=2.0, wspace=0.3)
    plt.show()

def generate_surface_path(surf_data, ex_data, model_data, path_idx=8000, model_type: Union[CVAE, CVAEMem] = CVAEMem):
    '''
        This function is for SABR data. Also this function applies an aggregated approach. 
        i.e. generate new surfaces based on previously generated data
    '''
    model_config = model_data["model_config"]
    model = model_type(model_config)
    model.load_weights(dict_to_load=model_data)
    seq_len = model_config["seq_len"]
    if "ex_feats_dim" in model_config:
        use_ex_feats = model_config["ex_feats_dim"] > 0
    else:
        use_ex_feats = False
    all_simulation = {
        "surface": [surf_data[path_idx][i] for i in range(seq_len-1)],
        "ex_feats": [ex_data[path_idx][i] for i in range(seq_len-1)] if use_ex_feats else None,
    }
    steps_to_sim = len(surf_data[path_idx]) + 1 - seq_len
    for i in range(steps_to_sim):
        ctx_data = {
            "surface": torch.from_numpy(np.array(all_simulation["surface"][i:(i+seq_len-1)])), 
            "ex_feats": torch.from_numpy(np.array(all_simulation["ex_feats"][i:(i+seq_len-1)])).unsqueeze(-1) if use_ex_feats else None
        }
        if use_ex_feats:
            surf, ex_feat = model.get_surface_given_conditions(ctx_data) 
        else:
            ctx_data.pop("ex_feats")
            surf = model.get_surface_given_conditions(ctx_data) 
        surf = surf.detach().cpu().numpy().reshape((5,5))
        all_simulation["surface"].append(surf)
        if use_ex_feats:
            ex_feat = ex_feat.detach().cpu().numpy().reshape((1,))[0]
            all_simulation["ex_feats"].append(ex_feat)

    all_simulation["surface"] = np.array(all_simulation["surface"])
    if use_ex_feats:
        all_simulation["ex_feats"] = np.array(all_simulation["ex_feats"])
    
    return all_simulation

def generate_surface_spx(surf_data, ex_data, model_data, start_time=5000, steps_to_sim=30, model_type: Union[CVAE, CVAEMem] = CVAEMem):
    '''
        This function is for S&P500 data. Also this function applies an aggregated approach. 
        i.e. generate new surfaces based on previously generated data

        This function uses the original return/price as ex features for simulation
    '''
    model_config = model_data["model_config"]
    model = model_type(model_config)
    model.load_weights(dict_to_load=model_data)
    seq_len = model_config["seq_len"]
    ctx_len = model_config["ctx_len"]
    if "ex_feats_dim" in model_config:
        use_ex_feats = model_config["ex_feats_dim"] > 0
    else:
        use_ex_feats = False
    all_simulation = {
        "surface": [surf_data[start_time+i] for i in range(ctx_len)],
        "ex_feats": [ex_data[start_time+i] for i in range(steps_to_sim + ctx_len)] if use_ex_feats else None,
    }
    for i in range(steps_to_sim):
        ctx_data = {
            "surface": torch.from_numpy(np.array(all_simulation["surface"][i:(i+ctx_len)])), 
            "ex_feats": torch.from_numpy(np.array(all_simulation["ex_feats"][i:(i+ctx_len)])).unsqueeze(-1) if use_ex_feats else None
        }
        if use_ex_feats:
            surf, _ = model.get_surface_given_conditions(ctx_data) 
        else:
            ctx_data.pop("ex_feats")
            surf = model.get_surface_given_conditions(ctx_data) 
        surf = surf.detach().cpu().numpy().reshape((5,5))
        all_simulation["surface"].append(surf)
        # if use_ex_feats:
        #     ex_feat = ex_feat.detach().cpu().numpy().reshape((1,))[0]
        #     all_simulation["ex_feats"].append(ex_feat)

    all_simulation["surface"] = np.array(all_simulation["surface"])
    if use_ex_feats:
        all_simulation["ex_feats"] = np.array(all_simulation["ex_feats"])
    
    return all_simulation

def plot_surface_time_series(vae_output, title_label="VAE"):
    if isinstance(vae_output, dict):
        surfaces = vae_output["surface"]
        ex_feats = vae_output["ex_feats"]
    else:
        surfaces = vae_output
        ex_feats = None
    nrows = surfaces.shape[0] // 5
    if surfaces.shape[0] % 5 != 0:
        nrows += 1
    fig, ax = plt.subplots(nrows=nrows, ncols=5, subplot_kw={"projection": "3d"})
    x = [0.7,0.85,1,1.15,1.3]
    y = [0.08333,0.25,0.5,1,2]
    x, y = np.meshgrid(x, y)

    for i in range(len(surfaces)):
        r = i // 5
        c = i % 5
        ax[r][c].plot_surface(x, y, surfaces[i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax[r][c].set_xlabel("K/S")
        ax[r][c].set_ylabel("ttm")
        if ex_feats is not None:
            ax[r][c].set_title(f"{title_label} surface on day {i},\nex_feat={ex_feats[i]:.4f}")
        else:
            ax[r][c].set_title(f"{title_label} surface on day {i}")
    
    plt.subplots_adjust(right=4.0, top=4.0, hspace=0.5)
    plt.show()