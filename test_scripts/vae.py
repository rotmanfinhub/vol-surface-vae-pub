import random, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from tqdm import tqdm

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class VolSurfaceDataSet(Dataset):
    def __init__(self, dataset, seq_len):
        self.dataset = torch.from_numpy(dataset).float()
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.dataset) - self.seq_len + 1

    def __getitem__(self, idx):
        ele = self.dataset[idx:idx+self.seq_len]
        return ele

class BaseVAE(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, device="cuda", kl_weight=1.0):
        '''
            Inputs:
                seq_len: the time series sequence length
                feat_dim: the feature dimension of each time step, integer if 1D, tuple of integers if 2D
                kl_weight: weight \beta used for loss = RE + \beta * KL
        '''
        super(BaseVAE, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.device = device
    
    def encode(self, x):
        raise NotImplementedError()

    def decode(self, x):
        raise NotImplementedError()

    def forward(self, x):
        z_mean, z_log_var, z = self.encode(x)
        reconstruction = self.decode(z)
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

    def save_weights(self, model_dir, file_prefix):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.state_dict(), f"{model_dir}/{file_prefix}_weights.pt")
    
    def load_weights(self, model_dir, file_prefix):
        self.load_state_dict(torch.load(f"{model_dir}/{file_prefix}_weights.pt"))

class VAEDense(BaseVAE):
    def __init__(self, seq_len, feat_dim, latent_dim, device="cuda", kl_weight=1, 
                 hidden_layer_sizes=[100]):
        super().__init__(seq_len, feat_dim, latent_dim, device, kl_weight)
        self.hidden_layer_sizes = hidden_layer_sizes

        # encoder
        encoder_layers = OrderedDict()
        # flatten the feature dimension seq_len * feat_dim
        encoder_layers["flatten"] = nn.Flatten()
        if isinstance(self.feat_dim, tuple):
            self.flattened_dim = self.seq_len * self.feat_dim[0] * self.feat_dim[1]
        else:
            self.flattened_dim = self.seq_len * self.feat_dim
        in_feats = self.flattened_dim
        for i, out_feats in enumerate(self.hidden_layer_sizes):
            encoder_layers[f"enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
            encoder_layers[f"enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        self.encoder_dense = nn.Sequential(encoder_layers)
        self.z_mean_layer = nn.Linear(in_feats, self.latent_dim)
        self.z_log_var_layer = nn.Linear(in_feats, self.latent_dim)

        # decoder
        decoder_layers = OrderedDict()
        in_feats = self.latent_dim
        for i, out_feats in enumerate(reversed(self.hidden_layer_sizes)):
            decoder_layers[f"dec_dense_{i}"] = nn.Linear(in_feats, out_feats)
            decoder_layers[f"dec_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        decoder_layers[f"dec_output"] = nn.Linear(in_feats, self.flattened_dim)
        self.decoder_dense = nn.Sequential(decoder_layers)
        self.to(self.device)

    def encode(self, x):
        x = self.encoder_dense(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        eps = torch.rand_like(z_log_var)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps
        return (z_mean, z_log_var, z)
    
    def decode(self, x):
        out = self.decoder_dense(x)
        if isinstance(self.feat_dim, tuple):
            out = out.view((-1, self.seq_len, self.feat_dim[0], self.feat_dim[1]))
        else:
            out = out.view((-1, self.seq_len, self.feat_dim))
        return out

class VAEConv2D(BaseVAE):
    def __init__(self, seq_len, feat_dim, latent_dim, device="cuda", kl_weight=1,
                 hidden_layer_sizes=[100]):
        '''
            The input size should be BxTx5x5, sequence length will be used as the C channel
            Idea from: https://github.com/AntixK/PyTorch-VAE
        '''
        super().__init__(seq_len, feat_dim, latent_dim, device, kl_weight)
        self.hidden_layer_sizes = hidden_layer_sizes

        # we want to keep the dimensions the same, out_dim = (in_dim - kernel_size + 2*padding) / stride + 1
        # so padding = (out_dim * stride + kernel_size - in_dim) // 2 where in_dim and out_dim are 5
        padding = ((feat_dim[-1] - 1) * 2 + 3 - feat_dim[-1]) // 2
        if ((feat_dim[-1] - 1) * 2 + 3 - feat_dim[-1]) % 2 == 1:
            padding1 = padding + 1
            deconv_output_padding = 1
        else:
            padding1 = padding
            deconv_output_padding = 0
        
        # encoder
        encoder_layers = OrderedDict()
        in_feats = self.seq_len
        for i, out_feats in enumerate(self.hidden_layer_sizes):
            
            encoder_layers[f"enc_conv_{i}"] = nn.Conv2d(
                in_feats, out_feats,
                kernel_size=3, stride=2, padding=padding1,
            )
            encoder_layers[f"enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        encoder_layers["flatten"] = nn.Flatten()
        self.encoder_conv = nn.Sequential(encoder_layers)
        self.z_mean_layer = nn.Linear(in_feats*feat_dim[0]*feat_dim[1], self.latent_dim)
        self.z_log_var_layer = nn.Linear(in_feats*feat_dim[0]*feat_dim[1], self.latent_dim)

        # decoder
        self.decoder_input = nn.Linear(self.latent_dim, in_feats*feat_dim[0]*feat_dim[1])
        decoder_layers = OrderedDict()
        for i, out_feats in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            decoder_layers[f"dec_deconv_{i}"] = nn.ConvTranspose2d(
                in_feats, out_feats,
                kernel_size=3, stride=2, padding=padding1, output_padding=deconv_output_padding,
                )
            decoder_layers[f"dec_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        
        # transform to the original size
        decoder_layers["dec_final"] = nn.ConvTranspose2d(
            in_feats, in_feats,
            kernel_size=3, stride=2, padding=padding1, output_padding=deconv_output_padding,
        )
        decoder_layers["dec_final_activation"] = nn.ReLU()
        decoder_layers["dec_output"] = nn.Conv2d(
            in_feats, self.seq_len,
            kernel_size=3, padding="same"
        )
        self.decoder_layers = nn.Sequential(decoder_layers)
        self.to(self.device)
    
    def encode(self, x):
        '''
            Input:
                x should be of shape BxTx5x5
        '''
        x = self.encoder_conv(x) # Bx(hidden[-1]x5x5)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        eps = torch.rand_like(z_log_var)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps
        return (z_mean, z_log_var, z)
    
    def decode(self, x):
        '''
            Input:
                x should be of shape Bxlatent dim
        '''
        x = self.decoder_input(x) # Bx(hidden[-1]x5x5)
        x = x.view(-1, self.hidden_layer_sizes[-1], self.feat_dim[0], self.feat_dim[1])
        out = self.decoder_layers(x)
        return out

# perform model evaluation in terms of the accuracy and f1 score.
def model_eval(model: BaseVAE, dataloader):
    model.eval() # switch to eval model, will turn off randomness like dropout
    eval_loss = 0
    num_batches = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f"eval")):
        batch.to(model.device)

        losses = model.test_step(batch)

        eval_loss += losses["loss"].item()
        num_batches += 1

    return eval_loss / num_batches
    

def train(model: BaseVAE, train_dataloader: DataLoader, valid_dataloader: DataLoader, 
          lr=1e-5, epochs=100, 
          model_dir="./", file_prefix="vanilla"):
    model.train()
    optimizer = opt.AdamW(model.parameters(), lr)
    best_dev_loss = np.inf

    ## run for the specified number of epochs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_file = open(f"{model_dir}/{file_prefix}-{epochs}-log.txt", "w", encoding="utf-8")
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"train-{epoch}")):
            batch.to(model.device)

            losses = model.train_step(batch, optimizer)

            train_loss += losses["loss"].item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        
        dev_loss = model_eval(model, valid_dataloader)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            model.save_weights(model_dir, file_prefix)

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}, time elapsed :: {time.time() - epoch_start_time}")
        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}, time elapsed :: {time.time() - epoch_start_time}", file=log_file)
    print(f"training finished, total time :: {time.time() - start_time}")
    print(f"training finished, total time :: {time.time() - start_time}", file=log_file)

def test(model: BaseVAE, valid_dataloader: DataLoader, test_dataloader: DataLoader, 
         model_dir="./", file_prefix="vanilla"):
    model.load_weights(model_dir, file_prefix)
    dev_loss = model_eval(model, valid_dataloader)
    test_loss = model_eval(model, test_dataloader)

    print(f"dev loss: {dev_loss :.3f}, test_loss: {test_loss :.3f}")