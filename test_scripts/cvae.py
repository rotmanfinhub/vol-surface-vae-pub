from vae import BaseVAE
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConditionalVAEConv2D(BaseVAE):
    def __init__(self, seq_len, context_len,
                 feat_dim, latent_dim, device="cuda", kl_weight=1, 
                 hidden_layer_sizes=[100], ctx_hidden_layer_sizes=[100], ctx_embedding_size=100):
        '''
            Inputs:
                seq_len: the overall time sequence length
                context_len: the part of the sequence that should be treated as context, for now, we assume seq_len = context_len+1
                hidden_layer_sizes: the hidden layer sizes for the main encoder
                ctx_hidden_layer_sizes: the hidden layer sizes for the context encoder (for the convolutional hidden layers)
                ctx_embedding_size: the final embedding size of the context
        '''
        assert context_len + 1 == seq_len, "seq len > context_len + 1 is not supported"
        super().__init__(seq_len, feat_dim, latent_dim, device, kl_weight)
        self.context_len = context_len
        self.hidden_layer_sizes = hidden_layer_sizes
        self.ctx_hidden_layer_sizes = ctx_hidden_layer_sizes
        self.ctx_embedding_size = ctx_embedding_size

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
        # encodes the context and current input, so total num of features = seq_len
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

        # context encoder
        # There is no need for distribution sampling
        ctx_encoder_layers = OrderedDict()
        in_feats = self.context_len
        for i, out_feats in enumerate(self.ctx_hidden_layer_sizes):
            ctx_encoder_layers[f"ctx_enc_conv_{i}"] = nn.Conv2d(
                in_feats, out_feats,
                kernel_size=3, stride=2, padding=padding1,
            )
            ctx_encoder_layers[f"ctx_enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        ctx_encoder_layers["flatten"] = nn.Flatten()
        ctx_encoder_layers["ctx_enc_final_linear"] = nn.Linear(in_feats*feat_dim[0]*feat_dim[1], self.ctx_embedding_size)
        self.ctx_encoder = nn.Sequential(ctx_encoder_layers)

        # decoder
        # Inputs: 
        #   1. the latents generated by main encoder on seq_len input
        #   2. the embedding generated by context encoder on context_len input
        # Firstly, we try to regenerate the final output of the main encoder using decoder_input layer
        # Then, the deconvolution will reconstruct the (Bxinput_lenx5x5) input, where input_len = seq_len-context_len 
        in_feats = self.hidden_layer_sizes[-1]
        self.decoder_input = nn.Linear(self.latent_dim + self.ctx_embedding_size, in_feats*feat_dim[0]*feat_dim[1])
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
            in_feats, self.seq_len - self.context_len,
            kernel_size=3, padding="same"
        )
        self.decoder_layers = nn.Sequential(decoder_layers)
        self.to(self.device)
    
    def encode(self, x):
        '''
            Input:
                x should be of shape BxTx5x5, 
                x[:,:context_len,:,:] are the context surfaces (previous days),
                x[:,context_len:,:,:] is the surface to predict
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
                x should be of shape Bx(latent_dim+ctx_embedding_size)
        '''
        x = self.decoder_input(x) # Bx(hidden[-1]x5x5)
        x = x.view(-1, self.hidden_layer_sizes[-1], self.feat_dim[0], self.feat_dim[1])
        out = self.decoder_layers(x)
        return out

    def forward(self, x):
        '''
            Input:
                x should be either unbatched, (seq_len)x5x5 or batched Bx(seq_len)x5x5
        '''
        if len(x.shape) == 3 and x.shape[0] == self.seq_len:
            # unbatched data
            x = x.unsqueeze(0)
        ctx = x[:, :self.context_len, :, :] # c
        ctx_embedding = self.ctx_encoder(ctx) # embedded c
        z_mean, z_log_var, z = self.encode(x) # P(z|c,x)

        decoder_input = torch.cat([ctx_embedding, z], dim=1)
        reconstruction = self.decode(decoder_input) # P(x|c,z)
        return (reconstruction, z_mean, z_log_var, z)

    def train_step(self, x, optimizer: torch.optim.Optimizer):
        '''
            Input:
                x should be either unbatched, (seq_len)x5x5 or batched Bx(seq_len)x5x5
        '''
        if len(x.shape) == 3 and x.shape[0] == self.seq_len:
            # unbatched data
            x = x.unsqueeze(0)

        optimizer.zero_grad()
        x = x.to(self.device)
        input_x = x[:,self.context_len:, :, :] # x
        reconstruction, z_mean, z_log_var, z = self.forward(x)

        # RE = 1/M \sum_{i=1}^M (x_i - y_i)^2
        reconstruction_error = F.mse_loss(reconstruction, input_x)
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
        input_x = x[:,self.context_len:, :, :] # x
        reconstruction, z_mean, z_log_var, z = self.forward(x)

        # RE = 1/M \sum_{i=1}^M (x_i - y_i)^2
        reconstruction_error = F.mse_loss(reconstruction, input_x)
        # KL = -1/2 \sum_{i=1}^M (1+log(\sigma_k^2) - \sigma_k^2 - \mu_k^2)
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=1))
        total_loss = reconstruction_error + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }