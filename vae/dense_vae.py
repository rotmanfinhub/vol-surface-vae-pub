import torch
import torch.nn as nn
from vae.base import BaseVAE, BaseDecoder, BaseEncoder
from collections import OrderedDict

class DenseEncoder(BaseEncoder):
    def __init__(self, config: dict):
        super(DenseEncoder, self).__init__(config)
        hidden_layer_sizes = config["hidden"]
        seq_len = config["seq_len"]
        feat_dim = config["feat_dim"]
        latent_dim = config["latent_dim"]
        # encoder
        encoder_layers = OrderedDict()
        # flatten the feature dimension seq_len * feat_dim
        encoder_layers["flatten"] = nn.Flatten()
        if isinstance(feat_dim, tuple):
            flattened_dim = seq_len * feat_dim[0] * feat_dim[1]
        else:
            flattened_dim = seq_len * feat_dim
        in_feats = flattened_dim
        for i, out_feats in enumerate(hidden_layer_sizes):
            encoder_layers[f"enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
            encoder_layers[f"enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        self.encoder_layers = nn.Sequential(encoder_layers)
        self.z_mean_layer = nn.Linear(in_feats, latent_dim)
        self.z_log_var_layer = nn.Linear(in_feats, latent_dim)
    
    def forward(self, x):
        x = self.encoder_layers(x)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        eps = torch.rand_like(z_log_var)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps
        return (z_mean, z_log_var, z)

class DenseDecoder(BaseDecoder):
    def __init__(self, config: dict):
        super(DenseDecoder, self).__init__(config)
        hidden_layer_sizes = config["hidden"]
        self.seq_len = config["seq_len"]
        self.feat_dim = config["feat_dim"]
        latent_dim = config["latent_dim"]

        if isinstance(self.feat_dim, tuple):
            flattened_dim = self.seq_len * self.feat_dim[0] * self.feat_dim[1]
        else:
            flattened_dim = self.seq_len * self.feat_dim

        decoder_layers = OrderedDict()
        in_feats = latent_dim
        for i, out_feats in enumerate(reversed(hidden_layer_sizes)):
            decoder_layers[f"dec_dense_{i}"] = nn.Linear(in_feats, out_feats)
            decoder_layers[f"dec_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        decoder_layers[f"dec_output"] = nn.Linear(in_feats, flattened_dim)
        self.decoder_layers = nn.Sequential(decoder_layers)
    
    def forward(self, x):
        out = self.decoder_layers(x)
        if isinstance(self.feat_dim, tuple):
            out = out.view((-1, self.seq_len, self.feat_dim[0], self.feat_dim[1]))
        else:
            out = out.view((-1, self.seq_len, self.feat_dim))
        return out

class VAEDense(BaseVAE):
    def __init__(self, config: dict):
        '''
            The Dense encoding and decoding version of VAE
            
            Input:
                config: must contain hidden, seq_len, feat_dim, latent_dim, device, kl_weight
                seq_len: the time series sequence length
                feat_dim: the feature dimension of each time step, integer if 1D, tuple of integers if 2D
                kl_weight: weight \beta used for loss = RE + \beta * KL
                hidden: hidden layer sizes
        '''
        super(VAEDense, self).__init__(config)
        self.check_input(config)
        self.encoder = DenseEncoder(config)
        self.decoder = DenseDecoder(config)
        self.to(self.device)
    
    def check_input(self, config: dict):
        super().check_input(config)
        if "hidden" not in config:
            raise "config doesn't contain: hidden (hidden layer sizes)"
        if not isinstance(config["hidden"], list):
            raise "hidden (hidden layer sizes) must be a list"