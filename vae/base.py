import os, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

class BaseEncoder(nn.Module):
    def __init__(self, config: dict):
        super(BaseEncoder, self).__init__()
        self.config = config
        self.device = config["device"]
    
    def forward(self, x):
        raise NotImplementedError()
    
class BaseDecoder(nn.Module):
    def __init__(self, config: dict):
        super(BaseDecoder, self).__init__()
        self.config = config
        self.device = config["device"]
    
    def forward(self, x):
        raise NotImplementedError()

class BaseVAE(nn.Module):
    def __init__(self, config: dict):
        '''
            Inputs:
                config: a dictionary that contains keys: seq_len, feat_dim, latent_dim, device, kl_weight
                seq_len: the time series sequence length
                feat_dim: the feature dimension of each time step, integer if 1D, tuple of integers if 2D
                kl_weight: weight \beta used for loss = RE + \beta * KL
        '''
        super(BaseVAE, self).__init__()
        self.config = config

        self.kl_weight = config["kl_weight"]
        self.device = config["device"]
        self.encoder: BaseEncoder = None
        self.decoder: BaseDecoder = None
    
    def check_input(self, config: dict):
        for req in ["seq_len", "feat_dim", "latent_dim"]:
            if req not in config:
                raise ValueError(f"config doesn't contain: {req}")
        if "device" not in config:
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if "kl_weight" not in config:
            config["kl_weight"] = 1.0

    def forward(self, x):
        '''
            Input:
                x: tensor of shape (B,T,H,W)
            Returns:
                a tuple of reconstruction, z_mean, z_log_var, z, 
                where z is sampled from distribution defined by z_mean and z_log_var
        '''
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return (reconstruction, z_mean, z_log_var, z)
    
    def count_parameters(self):
        modules = []
        params = []
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: 
                continue
            p_count = parameter.numel()
            modules.append(name)
            params.append(p_count)
            total_params += p_count
        modules.append("Total")
        params.append(total_params)
        df = pd.DataFrame({"module": modules, "num_params": params})
        return df
    
    def get_prior_samples(self, num_samples):
        z = torch.randn((num_samples, self.latent_dim)).to(self.device)
        samples = self.decode(z)
        return samples
    
    def get_prior_samples_given_z(self, z):
        z = z.to(self.device)
        return self.decode(z)
    
    def train_step(self, x, optimizer: torch.optim.Optimizer):
        x = x.to(self.device)
        optimizer.zero_grad()
        reconstruction, z_mean, z_log_var, z = self.forward(x)

        # RE = 1/M \sum_{i=1}^M (x_i - y_i)^2
        reconstruction_error = F.mse_loss(reconstruction, x)
        # KL = -1/2 \sum_{i=1}^M (1+log(\sigma_k^2) - \sigma_k^2 - \mu_k^2)
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=1))
        total_loss = reconstruction_error + self.kl_weight * kl_loss
        total_loss.backward()
        optimizer.step()

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }
    
    def test_step(self, x):
        x = x.to(self.device)
        reconstruction, z_mean, z_log_var, z = self.forward(x)

        # RE = 1/M \sum_{i=1}^M (x_i - y_i)^2
        reconstruction_error = F.mse_loss(reconstruction, x)
        # KL = -1/2 \sum_{i=1}^M (1+log(\sigma_k^2) - \sigma_k^2 - \mu_k^2)
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=1))
        total_loss = reconstruction_error + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }

    def save_weights(self, optimizer: torch.optim.Optimizer, model_dir, filename):
        dict_to_save = {
            "model": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": self.config,
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        }
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(dict_to_save, f"{model_dir}/{filename}.pt")
    
    def load_weights(self, f: str=None, dict_to_load: dict=None):
        '''
            Input:
                f: path-like, pointing to the saved state dictionary
                dict_to_load: a dictionary, loaded from the saved file
                One of them must not be None
        '''
        assert (f is not None) or (dict_to_load is not None), "One of file path or dict_to_load must not be None"
        if f is not None:
            dict_to_load = torch.load(f)
        self.load_state_dict(dict_to_load["model"])
