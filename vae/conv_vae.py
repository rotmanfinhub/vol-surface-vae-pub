import torch
import torch.nn as nn
from vae.base import BaseVAE, BaseDecoder, BaseEncoder
from collections import OrderedDict

class Conv2DEncoder(BaseEncoder):
    def __init__(self, config: dict):
        super(Conv2DEncoder, self).__init__(config)
        
        hidden_layer_sizes = config["hidden"]
        seq_len = config["seq_len"]
        feat_dim = config["feat_dim"]
        latent_dim = config["latent_dim"]
        padding = config["padding"]

        encoder_layers = OrderedDict()
        in_feats = seq_len
        for i, out_feats in enumerate(hidden_layer_sizes):
            
            encoder_layers[f"enc_conv_{i}"] = nn.Conv2d(
                in_feats, out_feats,
                kernel_size=3, stride=2, padding=padding,
            )
            encoder_layers[f"enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        encoder_layers["flatten"] = nn.Flatten()
        self.encoder_layers = nn.Sequential(encoder_layers)
        self.z_mean_layer = nn.Linear(in_feats*feat_dim[0]*feat_dim[1], latent_dim)
        self.z_log_var_layer = nn.Linear(in_feats*feat_dim[0]*feat_dim[1], latent_dim)

    def forward(self, x):
        '''
            Input:
                x should be of shape (B,T,H,W)
        '''
        x = self.encoder_layers(x) # (B,hidden[-1]xHxW)
        z_mean = self.z_mean_layer(x)
        z_log_var = self.z_log_var_layer(x)
        eps = torch.rand_like(z_log_var)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps
        return (z_mean, z_log_var, z)

class Conv2DDecoder(BaseDecoder):
    def __init__(self, config: dict):
        super(Conv2DDecoder, self).__init__(config)

        hidden_layer_sizes = config["hidden"]
        seq_len = config["seq_len"]
        feat_dim = config["feat_dim"]
        latent_dim = config["latent_dim"]
        padding = config["padding"]
        deconv_output_padding = config["deconv_output_padding"]

        # record the final hidden size for forward function
        in_feats = self.final_hidden_size = hidden_layer_sizes[-1]

        self.decoder_input = nn.Linear(latent_dim, in_feats*feat_dim[0]*feat_dim[1])
        decoder_layers = OrderedDict()
        for i, out_feats in enumerate(reversed(hidden_layer_sizes[:-1])):
            decoder_layers[f"dec_deconv_{i}"] = nn.ConvTranspose2d(
                in_feats, out_feats,
                kernel_size=3, stride=2, padding=padding, output_padding=deconv_output_padding,
                )
            decoder_layers[f"dec_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        
        # transform to the original size
        decoder_layers["dec_final"] = nn.ConvTranspose2d(
            in_feats, in_feats,
            kernel_size=3, stride=2, padding=padding, output_padding=deconv_output_padding,
        )
        decoder_layers["dec_final_activation"] = nn.ReLU()
        decoder_layers["dec_output"] = nn.Conv2d(
            in_feats, seq_len,
            kernel_size=3, padding="same"
        )
        self.decoder_layers = nn.Sequential(decoder_layers)
    
    def forward(self, x):
        '''
            Input:
                x should be of shape (B,latent_dim)
        '''
        feat_dim = self.config["feat_dim"]
        x = self.decoder_input(x) # (B,hidden[-1]xHxW)
        x = x.view(-1, self.final_hidden_size, feat_dim[0], feat_dim[1])
        out = self.decoder_layers(x)
        return out

class VAEConv2D(BaseVAE):
    def __init__(self, config: dict):
        '''
            The 2D Convolutional encoding and decoding version of VAE. 
            The input size should be (B,T,H,W), sequence length will be used as the C channel
            Idea from: https://github.com/AntixK/PyTorch-VAE
            
            Input:
                config: must contain hidden, seq_len, feat_dim, latent_dim, device, kl_weight
                seq_len: the time series sequence length
                feat_dim: the feature dimension of each time step, integer if 1D, tuple of integers if 2D
                kl_weight: weight \beta used for loss = RE + \beta * KL
                hidden: hidden layer sizes
            
        '''
        super(VAEConv2D, self).__init__(config)
        self.check_input(config)
        # we want to keep the dimensions the same, out_dim = (in_dim - kernel_size + 2*padding) / stride + 1
        # so padding = (out_dim * stride + kernel_size - in_dim) // 2 where in_dim and out_dim are 5
        feat_dim = config["feat_dim"]
        padding = ((feat_dim[-1] - 1) * 2 + 3 - feat_dim[-1]) // 2
        if ((feat_dim[-1] - 1) * 2 + 3 - feat_dim[-1]) % 2 == 1:
            padding += 1
            deconv_output_padding = 1
        else:
            deconv_output_padding = 0
        
        config["padding"] = padding
        config["deconv_output_padding"] = deconv_output_padding

        self.encoder = Conv2DEncoder(config)
        self.decoder = Conv2DDecoder(config)
        self.to(self.device)
    
    def check_input(self, config: dict):
        super().check_input(config)
        if "hidden" not in config:
            raise "config doesn't contain: hidden (hidden layer sizes)"
        if not isinstance(config["hidden"], list):
            raise "hidden (hidden layer sizes) must be a list"