import torch
import torch.nn as nn
import torch.nn.functional as F
from vae.base import BaseVAE, BaseDecoder, BaseEncoder
from collections import OrderedDict

class CVAEMemRandEncoder(BaseEncoder):
    def __init__(self, config: dict):
        '''
            Encoder for the main observation data.
            
            Given X=(B,T,H,W) the surface sequence and A=(B,T,n) the extra information, the encoder does the following:
            - Embed surface into (B,T,n_surface)
            - Embed extra features into (B,T,n_info) (Not necessary for now)
            - Concate both info, (B,T,n_surface+n_info)
            - Encode time features using memory (RNN/GRU/LSTM), (B,T,n_mem)
            - Map to latent space, (B, T, latent_dim), latents will be generated for each timestep, 
            as we don't want future information to be observed for current/previous timesteps
        '''
        super(CVAEMemRandEncoder, self).__init__(config)

        latent_dim = config["latent_dim"]

        surface_embedding_final_dim = self.__get_surface_embedding(config)
        if config["ex_feats_dim"] > 0:
            ex_feats_embedding_final_dim = self.__get_ex_feats_embedding(config)
        else:
            ex_feats_embedding_final_dim = 0
        
        self.__get_interaction_layers(config, surface_embedding_final_dim + ex_feats_embedding_final_dim)
        mem_final_dim = self.__get_mem(config, surface_embedding_final_dim + ex_feats_embedding_final_dim)

        self.z_mean_layer = nn.Linear(mem_final_dim, latent_dim)
        self.z_log_var_layer = nn.Linear(mem_final_dim, latent_dim)

    def __get_surface_embedding(self, config):
        '''Embedding for vol surface'''
        feat_dim = config["feat_dim"]

        surface_embedding_layers = config["surface_hidden"]

        surface_embedding = OrderedDict()
        if config["use_dense_surface"]:
            in_feats = feat_dim[0] * feat_dim[1]
            surface_embedding["flatten"] = nn.Flatten()
            for i, out_feats in enumerate(surface_embedding_layers):
                surface_embedding[f"enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
                surface_embedding[f"enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            
            self.surface_embedding = nn.Sequential(surface_embedding)
            final_dim = in_feats
        else:
            # convolutional layer
            padding = config["padding"]
            in_feats = 1 # encoding per vol surface
            for i, out_feats in enumerate(surface_embedding_layers):
                
                surface_embedding[f"enc_conv_{i}"] = nn.Conv2d(
                    in_feats, out_feats,
                    kernel_size=3, stride=1, padding=padding,
                )
                surface_embedding[f"enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            surface_embedding["flatten"] = nn.Flatten()
            self.surface_embedding = nn.Sequential(surface_embedding)

            final_dim = in_feats * feat_dim[0] * feat_dim[1]
        return final_dim

    def __get_ex_feats_embedding(self, config):
        '''Embedding for extra features'''
        ex_feats_dim = config["ex_feats_dim"]
        ex_feats_embedding_layers = config["ex_feats_hidden"]
        if ex_feats_embedding_layers is None:
            # we simply assume that there is no need for embedding the extra features
            self.ex_feats_embedding = nn.Identity()
            return ex_feats_dim
        
        # The following code is not used and not tested
        ex_feats_embedding = OrderedDict()
        in_feats = ex_feats_dim
        for i, out_feats in enumerate(ex_feats_embedding_layers):
            ex_feats_embedding[f"ex_enc_linear_{i}"] = nn.Linear(in_feats, out_feats)
            ex_feats_embedding[f"ex_enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        self.ex_feats_embedding = nn.Sequential(ex_feats_embedding)
        return in_feats # final dimension

    def __get_interaction_layers(self, config, input_size):
        if config["interaction_layers"] is not None and config["interaction_layers"] > 0:
            num_layers = config["interaction_layers"]
            interaction = OrderedDict()
            for i in range(num_layers):
                interaction[f"enc_interact_linear_{i}"] = nn.Linear(input_size, input_size)
                interaction[f"enc_interaction_activation_{i}"] = nn.ReLU()
            
            interaction["enc_interact_linear_final"] = nn.Linear(input_size, input_size)
            self.interaction = nn.Sequential(interaction)
        else:
            self.interaction = nn.Identity()

    def __get_mem(self, config, input_size):
        # Memory using LSTM/RNN/GRU
        mem_type = config["mem_type"]
        mem_args = {
            "input_size": input_size,
            "hidden_size": config["mem_hidden"],
            "num_layers": config["mem_layers"],
            "batch_first": True,
            "dropout": config["mem_dropout"],
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)
        return config["mem_hidden"]

    def forward(self, x):
        '''
            Input:
                x should be a dictionary with 2 keys:
                - surface: volatility surface of shape (B,T,H,W), 
                    surface[:,:context_len,:,:] are the context surfaces (previous days),
                    surface[:,context_len:,:,:] is the surface to predict
                - ex_feats: extra features of shape (B,T,n), this doesn't have to exist
        '''
        latent_dim = self.config["latent_dim"]
        surface: torch.Tensor = x["surface"] # (B, T, H, W)
        T = surface.shape[1]

        # Embed surface of each day individually
        surface = surface.reshape((surface.shape[0] * surface.shape[1], 1, surface.shape[2], surface.shape[3])) # (BxT, 1, H, W), this works for both dense and conv versions
        surface_embedding = self.surface_embedding(surface) # (BxT, surface_embedding_final_dim)
        surface_embedding = surface_embedding.reshape((-1, T, surface_embedding.shape[1])) # (B, T, surface_embedding_final_dim)
        
        if "ex_feats" in x:
            ex_feats: torch.Tensor = x["ex_feats"] # (B, T, n)
            # Embed features of each day individually
            ex_feats = ex_feats.reshape((ex_feats.shape[0] * ex_feats.shape[1], ex_feats.shape[2])) # (BxT, n)
            ex_feats_embedding = self.ex_feats_embedding(ex_feats) # (BxT, ex_feats_embedding_final_dim)
            ex_feats_embedding = ex_feats_embedding.reshape((-1, T, ex_feats_embedding.shape[1])) # (B, T, ex_feats_embedding_final_dim)
        
            # concat the embeddings and get the time features
            embeddings = torch.cat([surface_embedding, ex_feats_embedding], dim=-1)  # (B, T, surface_embedding_final_dim + ex_feats_embedding_final_dim)
        else:
            embeddings = surface_embedding

        embeddings = self.interaction(embeddings) # add some nonlinear interactions, nn layers act on the final dimension only
        embeddings, _ = self.mem(embeddings) # (B, T, n_lstm)

        embeddings = embeddings.reshape((embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2])) # (BxT, n_lstm)
        z_mean = self.z_mean_layer(embeddings).reshape((-1, T, latent_dim)) # (B, T, latent_dim)
        z_log_var = self.z_log_var_layer(embeddings).reshape((-1, T, latent_dim)) # (B, T, latent_dim)
        eps = torch.randn_like(z_log_var)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps
        return (z_mean, z_log_var, z)

class CVAECtxMemRandEncoder(BaseEncoder):
    def __init__(self, config: dict):
        '''
            Encoder for the context.
            
            Given X=(B,C,H,W) the surface sequence and A=(B,C,n) the extra information, the context encoder does the following:
            - Embed surface into (B,C,n_surface)
            - Embed extra features into (B,C,n_info) (Not necessary for now)
            - Concate both info, (B,C,n_surface+n_info)
            - Encode time features using LSTM, (B,C,n_lstm)
            - (Potentially) compress to (B,C,latent_dim)
        '''
        super(CVAECtxMemRandEncoder, self).__init__(config)

        # There is no need for distribution sampling
        surface_embedding_dim = self.__get_surface_embedding(config)
        if config["ex_feats_dim"] > 0:
            ex_feats_embedding_dim = self.__get_ex_feats_embedding(config)
        else:
            ex_feats_embedding_dim = 0
        
        self.__get_interaction_layers(config, surface_embedding_dim + ex_feats_embedding_dim)
        mem_hidden = self.__get_mem(config, surface_embedding_dim + ex_feats_embedding_dim)
        if config["compress_context"]:
            self.final_compression = nn.Linear(mem_hidden, config["latent_dim"])
        else:
            self.final_compression = nn.Identity()
    
    def __get_surface_embedding(self, config):
        '''Embedding for vol surface'''
        feat_dim = config["feat_dim"]

        ctx_surface_embedding_layers = config["ctx_surface_hidden"]
        
        ctx_surface_embedding = OrderedDict()
        if config["use_dense_surface"]:
            in_feats = feat_dim[0] * feat_dim[1]
            ctx_surface_embedding["flatten"] = nn.Flatten()
            for i, out_feats in enumerate(ctx_surface_embedding_layers):
                ctx_surface_embedding[f"ctx_enc_dense_{i}"] = nn.Linear(in_feats, out_feats)
                ctx_surface_embedding[f"ctx_enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            
            self.ctx_surface_embedding = nn.Sequential(ctx_surface_embedding)
            final_dim = in_feats
        else:
            padding = config["padding"]
            in_feats = 1 # encoding per vol surface
            for i, out_feats in enumerate(ctx_surface_embedding_layers):
                
                ctx_surface_embedding[f"ctx_enc_conv_{i}"] = nn.Conv2d(
                    in_feats, out_feats,
                    kernel_size=3, stride=1, padding=padding,
                )
                ctx_surface_embedding[f"ctx_enc_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            ctx_surface_embedding["flatten"] = nn.Flatten()
            self.ctx_surface_embedding = nn.Sequential(ctx_surface_embedding)

            final_dim = in_feats * feat_dim[0] * feat_dim[1]
        return final_dim

    def __get_ex_feats_embedding(self, config):
        '''Embedding for extra features'''
        ex_feats_dim = config["ex_feats_dim"]
        ctx_ex_feats_embedding_layers = config["ctx_ex_feats_hidden"]
        if ctx_ex_feats_embedding_layers is None:
            # we simply assume that there is no need for embedding the extra features
            self.ctx_ex_feats_embedding = nn.Identity()
            return ex_feats_dim
        
        # The following code is not used and not tested
        ctx_ex_feats_embedding = OrderedDict()
        in_feats = ex_feats_dim
        for i, out_feats in enumerate(ctx_ex_feats_embedding_layers):
            ctx_ex_feats_embedding[f"ctx_ex_enc_linear_{i}"] = nn.Linear(in_feats, out_feats)
            ctx_ex_feats_embedding[f"ctx_ex_enc_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        self.ctx_ex_feats_embedding = nn.Sequential(ctx_ex_feats_embedding)
        return in_feats # final dimension
    
    def __get_interaction_layers(self, config, input_size):
        if config["interaction_layers"] is not None and config["interaction_layers"] > 0:
            num_layers = config["interaction_layers"]
            interaction = OrderedDict()
            for i in range(num_layers):
                interaction[f"ctx_interact_linear_{i}"] = nn.Linear(input_size, input_size)
                interaction[f"ctx_interaction_activation_{i}"] = nn.ReLU()
            
            interaction["ctx_interact_linear_final"] = nn.Linear(input_size, input_size)
            self.interaction = nn.Sequential(interaction)
        else:
            self.interaction = nn.Identity()

    def __get_mem(self, config, input_size):
        # Memory using LSTM/RNN/GRU
        mem_type = config["mem_type"]
        mem_args = {
            "input_size": input_size,
            "hidden_size": config["mem_hidden"],
            "num_layers": config["mem_layers"],
            "batch_first": True,
            "dropout": config["mem_dropout"],
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)
        return config["mem_hidden"]

    def forward(self, x):
        '''
            Input:
                x should be a dictionary with 2 keys:
                - surface: volatility surface of shape (B,C,H,W), 
                    surface[:,:context_len,:,:] are the context surfaces (previous days),
                    surface[:,context_len:,:,:] is the surface to predict
                - ex_feats: extra features of shape (B,C,n)
        '''
        surface: torch.Tensor = x["surface"] # (B, C, H, W)
        C = surface.shape[1]

        # Embed surface of each day individually
        surface = surface.reshape((surface.shape[0] * surface.shape[1], 1, surface.shape[2], surface.shape[3])) # (BxC, 1, H, W), this works for both dense and conv versions
        surface_embedding = self.ctx_surface_embedding(surface)
        surface_embedding = surface_embedding.reshape((-1, C, surface_embedding.shape[1]))

        if "ex_feats" in x:
            ex_feats: torch.Tensor = x["ex_feats"] # (B, C, n)
            # Embed features of each day individually
            ex_feats = ex_feats.reshape((ex_feats.shape[0] * ex_feats.shape[1], ex_feats.shape[2])) # (BxT, n)
            ex_feats_embedding = self.ctx_ex_feats_embedding(ex_feats) # (BxT, ex_feats_embedding_final_dim)
            ex_feats_embedding = ex_feats_embedding.reshape((-1, C, ex_feats_embedding.shape[1])) # (B, T, ex_feats_embedding_final_dim)

            ctx_embeddings = torch.cat([surface_embedding, ex_feats_embedding], dim=-1)
        else:
            ctx_embeddings = surface_embedding
        ctx_embeddings = self.interaction(ctx_embeddings) # linear acts on the final layers only
        ctx_embeddings, _ = self.mem(ctx_embeddings)
        ctx_embeddings = self.final_compression(ctx_embeddings)
        return ctx_embeddings

class CVAEMemRandDecoder(BaseDecoder):
    def __init__(self, config: dict):
        '''
            Inputs to this module: 1. the latents generated by main encoder on seq_len input. (B, T, latent_dim) 2. the embedding generated by context encoder on context_len input (B, C, mem_hidden) padded with zeros to (B, T, mem_hidden)
            
            The decoder does the following:
            - Memory map back to (B, T, n_surface+n_info)
            - Deconv on (B, T, n_surface) to reconstruct the surface
            - Dense mapping on (B, T, n_info) to reconstruct the extra features
        '''

        super(CVAEMemRandDecoder, self).__init__(config)

        surface_embedding_layers = config["surface_hidden"]
        ex_feats_embedding_layers = config["ex_feats_hidden"]
        feat_dim = config["feat_dim"]
        latent_dim = config["latent_dim"]
        if config["compress_context"]:
            ctx_embedding_dim = config["latent_dim"]
        else:
            ctx_embedding_dim = config["mem_hidden"]

        # record the final hidden size for forward function
        self.surface_final_hidden_size = surface_embedding_layers[-1]
        if ex_feats_embedding_layers is not None:
            self.n_info = self.ex_feats_final_hidden_size = ex_feats_embedding_layers[-1]  
        else:
            self.n_info = self.ex_feats_final_hidden_size = config["ex_feats_dim"]
        
        if config["use_dense_surface"]:
            self.n_surface = surface_embedding_layers[-1]
        else:
            self.n_surface = surface_embedding_layers[-1] * feat_dim[0] * feat_dim[1]

        self.__get_mem(config, latent_dim + ctx_embedding_dim, self.n_surface + self.n_info)

        self.__get_interaction_layers(config, self.n_surface + self.n_info)
        self.surface_decoder_input = nn.Linear(self.n_surface + self.n_info, self.n_surface)
        self.__get_surface_decoder(config)

        if self.n_info > 0:
            self.ex_feats_decoder_input = nn.Linear(self.n_surface + self.n_info, self.n_info)
            self.__get_ex_feats_decoder(config)

    def __get_surface_decoder(self, config):
        surface_embedding_layers = config["surface_hidden"]

        surface_decoder = OrderedDict()
        if config["use_dense_surface"]:
            feat_dim = config["feat_dim"]
            in_feats = surface_embedding_layers[-1]
            for i, out_feats in enumerate(reversed(surface_embedding_layers[:-1])):
                surface_decoder[f"dec_dense_{i}"] = nn.Linear(in_feats, out_feats)
                surface_decoder[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            # transform to the original size
            final_size = feat_dim[0] * feat_dim[1]
            surface_decoder["dec_final"] = nn.Linear(in_feats, final_size)
            surface_decoder["dec_final_activation"] = nn.ReLU()
            surface_decoder["dec_output"] = nn.Linear(final_size, final_size)
        else:
            padding = config["padding"]
            deconv_output_padding = config["deconv_output_padding"]
            in_feats = surface_embedding_layers[-1]
            for i, out_feats in enumerate(reversed(surface_embedding_layers[:-1])):
                surface_decoder[f"dec_deconv_{i}"] = nn.ConvTranspose2d(
                    in_feats, out_feats,
                    kernel_size=3, stride=1, padding=padding, output_padding=deconv_output_padding,
                    )
                surface_decoder[f"dec_activation_{i}"] = nn.ReLU()
                in_feats = out_feats
            
            # transform to the original size
            surface_decoder["dec_final"] = nn.ConvTranspose2d(
                in_feats, in_feats,
                kernel_size=3, stride=1, padding=padding, output_padding=deconv_output_padding,
            )
            surface_decoder["dec_final_activation"] = nn.ReLU()
            surface_decoder["dec_output"] = nn.Conv2d(
                in_feats, 1,
                kernel_size=3, padding="same"
            )
        self.surface_decoder = nn.Sequential(surface_decoder)

    def __get_ex_feats_decoder(self, config):
        ex_feats_dim = config["ex_feats_dim"]
        ex_feats_embedding_layers = config["ex_feats_hidden"]
        if ex_feats_embedding_layers is None:
            # we simply assume that there is no need for embedding the extra features, but we need a linear mapping now
            self.ex_feats_decoder = nn.Linear(ex_feats_dim, ex_feats_dim)
            return
        
        # The following code is not used and not tested
        ex_feats_decoder = OrderedDict()
        in_feats = ex_feats_embedding_layers[-1]
        for i, out_feats in enumerate(reversed(ex_feats_embedding_layers[:-1])):
            ex_feats_decoder[f"ctx_ex_dec_linear_{i}"] = nn.Linear(in_feats, out_feats)
            ex_feats_decoder[f"ctx_ex_dec_activation_{i}"] = nn.ReLU()
            in_feats = out_feats
        ex_feats_decoder["ctx_dec_output"] = nn.Linear(in_feats, ex_feats_dim)
        self.ex_feats_decoder = nn.Sequential(ex_feats_decoder)
    
    def __get_interaction_layers(self, config, input_size):
        if config["interaction_layers"] is not None and config["interaction_layers"] > 0:
            num_layers = config["interaction_layers"]
            interaction = OrderedDict()
            for i in range(num_layers):
                interaction[f"dec_interact_linear_{i}"] = nn.Linear(input_size, input_size)
                interaction[f"dec_interaction_activation_{i}"] = nn.ReLU()
            
            interaction["dec_interact_linear_final"] = nn.Linear(input_size, input_size)
            self.interaction = nn.Sequential(interaction)
        else:
            self.interaction = nn.Identity()

    def __get_mem(self, config, input_size, hidden_size):
        # Memory using LSTM/RNN/GRU
        mem_type = config["mem_type"]
        mem_args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": config["mem_layers"],
            "batch_first": True,
            "dropout": config["mem_dropout"],
        }
        if mem_type == "lstm":
            self.mem = nn.LSTM(**mem_args)
        elif mem_type == "gru":
            self.mem = nn.GRU(**mem_args)
        else:
            self.mem = nn.RNN(**mem_args)
    
    def forward(self, x):
        '''
            Input:
                x should be (B, T, latent_dim + mem_hidden), where x[:,T-C:,latent_dim:] should be 0
        '''
        feat_dim = self.config["feat_dim"]
        ex_feats_dim = self.config["ex_feats_dim"]
        # should be on device already
        x, _ = self.mem(x) # (B, T, n_surface+n_info)
        x = self.interaction(x)
        B, T = x.shape[0], x.shape[1]
        surface_x = self.surface_decoder_input(x) # (B, T, n_surface)
        if self.config["use_dense_surface"]:
            surface_x = surface_x.reshape(-1, self.surface_final_hidden_size) # (BxT, surface_final_hidden_size)
        else:
            surface_x = surface_x.reshape(-1, self.surface_final_hidden_size, feat_dim[0], feat_dim[1]) # (BxT, surface_final_hidden_size, H, W)
        decoded_surface = self.surface_decoder(surface_x)
        decoded_surface = decoded_surface.reshape((B, T, feat_dim[0], feat_dim[1]))

        if ex_feats_dim > 0:
            info_x = self.ex_feats_decoder_input(x) # (B, T, n_info)
            info_x = info_x.reshape(B*T, self.n_info) # (BxT, n_info)
            decoded_ex_feat = self.ex_feats_decoder(info_x) # (BxT, n_info)
            decoded_ex_feat = decoded_ex_feat.reshape((B, T, ex_feats_dim))

            return decoded_surface, decoded_ex_feat
        else:
            return decoded_surface

class CVAEMemRand(BaseVAE):
    def __init__(self, config: dict):
        '''
            Similar to CVAEMem in cvae_with_mem, but we don't assume anything about seq_len (T) or ctx_len (C) here.
            The input size should be (B,T,H,W), sequence length will be used as the color channel, for now, we assume T=seq_len = ctx_len(C)+1
            
            Input:
                config: must contain feat_dim, latent_dim, device, kl_weight, re_feat_weight, surface_hidden, ex_feats_dim, ex_feats_hidden, mem_type, mem_hidden, mem_layers, mem_dropout, ctx_surface_hidden, ctx_ex_feats_hidden
                feat_dim: the feature dimension of each time step, integer if 1D, tuple of integers if 2D
                latent_dim: dimension for latent space
                kl_weight: weight \beta used for loss = RE + \beta * KL, (default: 1.0)
                re_feat_weight: weight \alpha used for RE = RE(surface) + \alpha * RE(ex_feats), (default: 1.0)
                ex_feats_loss_type: loss type for RE(ex_feats), (default: l1)
                surface_hidden: hidden layer sizes for vol surface encoding
                ex_feats_dim: the number of extra features
                ex_feats_hidden: the hidden layer sizes for the extra feature (default: None, if None, identity mapping)
                mem_type: lstm/gru/rnn (default: lstm)
                mem_hidden: hidden size for memory, int
                mem_layers: number of layers for memory, int
                mem_dropout: dropout rate for memory (default: 0)
                ctx_surface_hidden: the hidden layer sizes for the context encoder (for the vol surface)
                ctx_ex_feats_hidden: the hidden layer sizes for the context encoder (for the extra features, default: None)
                interaction_layers: number of nonlinear layers for surface and extra features to interact (default: 2)
                use_dense_surface: whether or not flatten the surface into 1D and use Dense Layers for encoding/decoding (default: False)
                compress_context: whether or not compress the context encoding to the same size as the latent dimension (default: True)
                ex_loss_on_ret_only: whether or not the return should only be optimized on surface & return. Any extra features will not get loss optimized. We assume that ret is always the first ex_feature, index=0. (default: False)
        '''
        super(CVAEMemRand, self).__init__(config)
        self.check_input(config)
        if not config["use_dense_surface"]:
            # we want to keep the dimensions the same, out_dim = (in_dim - kernel_size + 2*padding) / stride + 1
            # so padding = ((out_dim -1) * stride + kernel_size - in_dim) // 2 where in_dim and out_dim are 5
            stride = 1
            feat_dim = config["feat_dim"]
            padding = ((feat_dim[-1] - 1) * stride + 3 - feat_dim[-1]) // 2
            if ((feat_dim[-1] - 1) * stride + 3 - feat_dim[-1]) % 2 == 1:
                padding += 1
                deconv_output_padding = 1
            else:
                deconv_output_padding = 0
            
            config["padding"] = padding
            config["deconv_output_padding"] = deconv_output_padding

        self.encoder = CVAEMemRandEncoder(config)
        self.ctx_encoder = CVAECtxMemRandEncoder(config)
        self.decoder = CVAEMemRandDecoder(config)
        self.to(self.device)

        if config["ex_feats_loss_type"] == "l2":
            self.ex_feats_loss_fn = nn.MSELoss()
        else:
            self.ex_feats_loss_fn = nn.L1Loss()

    def get_surface_given_conditions(self, c: dict[str, torch.Tensor], z: torch.Tensor=None, mu=0, std=1):
        '''
            Input:
                c: context dictionary, "surface" have shape (B,C,H,W) or (C,H,W), "ex_feats" have shape (B, C, ex_feats_dim) or (C, ex_feats_dim)
                z: pre-generated latent samples, must be of shape (T,latent_dim) or (B,T,latent_dim), 
                mu, std: if z is not given, will be sampled from mu and std as normal distribution
        '''

        ctx_surface = c["surface"].to(self.device)
        if len(ctx_surface.shape) == 3:
            ctx_surface = ctx_surface.unsqueeze(0)
        C = ctx_surface.shape[1]
        B = ctx_surface.shape[0]
        T = C + 1
        ctx = {"surface": ctx_surface}

        if "ex_feats" in c:
            ctx_ex_feats = c["ex_feats"].to(self.device)
            if len(ctx_ex_feats.shape) == 2:
                ctx_ex_feats = ctx_ex_feats.unsqueeze(0)
            assert ctx_ex_feats.shape[1] == C, "context length mismatch"
            ctx["ex_feats"] = ctx_ex_feats
        if z is not None:
            if len(z.shape) == 2:
                z = z.unsqueeze(0)
        else:
            z = mu + torch.randn((ctx_surface.shape[0], T, self.config["latent_dim"])) * std
        
        ctx_latent_mean, ctx_latent_log_var, ctx_latent = self.encoder(ctx)
        z[:, :C, ...] = ctx_latent_mean

        ctx_embedding = self.ctx_encoder(ctx) # embedded c
        ctx_embedding_dim = ctx_embedding.shape[2]
        ctx_embedding_padded = torch.zeros((B, T, ctx_embedding_dim)).to(self.device)
        ctx_embedding_padded[:, :C, :] = ctx_embedding
        z = z.to(self.device)
        decoder_input = torch.cat([ctx_embedding_padded, z], dim=-1)
        if "ex_feats" in c:
            decoded_surface, decoded_ex_feat = self.decoder(decoder_input) # P(x|c,z,t)
            return decoded_surface[:, C:, :, :], decoded_ex_feat[:, C:, :]
        else:
            decoded_surface = self.decoder(decoder_input) # P(x|c,z,t)
            return decoded_surface[:, C:, :, :]
    
    def check_input(self, config: dict):
        for req in ["feat_dim", "latent_dim"]:
            if req not in config:
                raise ValueError(f"config doesn't contain: {req}")
        if "device" not in config:
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if "kl_weight" not in config:
            config["kl_weight"] = 1.0
        
        for req in ["surface_hidden", "ex_feats_dim",
                    "mem_hidden", "mem_layers", 
                    "ctx_surface_hidden"]:
            if req not in config:
                raise ValueError(f"config doesn't contain {req}")
        if isinstance(config["surface_hidden"], int):
            config["surface_hidden"] = [config["surface_hidden"]]
        if isinstance(config["ctx_surface_hidden"], int):
            config["ctx_surface_hidden"] = [config["ctx_surface_hidden"]]
        
        for req in ["ex_feats_hidden", "ctx_ex_feats_hidden"]:
            if req not in config:
                config[req] = None
        if config["ex_feats_hidden"] is not None and isinstance(config["ex_feats_hidden"]):
            config["ex_feats_hidden"] = [config["ex_feats_hidden"]]
        if config["ctx_ex_feats_hidden"] is not None and isinstance(config["ctx_ex_feats_hidden"]):
            config["ctx_ex_feats_hidden"] = [config["ctx_ex_feats_hidden"]]
        
        if "mem_type" not in config:
            config["mem_type"] = "lstm"
        if "mem_dropout" not in config:
            config["mem_dropout"] = 0
        
        if "re_feat_weight" not in config:
            config["re_feat_weight"] = 1.0
        if "ex_feats_loss_type" not in config:
            config["ex_feats_loss_type"] = "l1"
        if "ex_loss_on_ret_only" not in config:
            config["ex_loss_on_ret_only"] = False

        if "interaction_layers" not in config:
            config["interaction_layers"] = None
        
        if "use_dense_surface" not in config:
            config["use_dense_surface"] = False
        
        if "compress_context" not in config:
            config["compress_context"] = False
    
    def forward(self, x: dict[str, torch.Tensor]):
        '''
            Input:
                x should be a dictionary with 2 keys:
                - surface: volatility surface of shape (B,T,H,W), 
                    surface[:,:context_len,:,:] are the context surfaces (previous days),
                    surface[:,context_len:,:,:] is the surface to predict
                - ex_feats: extra features of shape (B,T,n), not necessarily needed
            Returns:
                a tuple of reconstruction, z_mean, z_log_var, z, 
                where z is sampled from distribution defined by z_mean and z_log_var
        '''
        surface = x["surface"].to(self.device)
        if len(surface.shape) == 3:
            # unbatched data
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - 1
        ctx_surface = surface[:, :C, :, :] # c
        ctx_encoder_input = {"surface": ctx_surface}

        encoder_input = {"surface": surface}
        if "ex_feats" in x:
            ex_feats = x["ex_feats"].to(self.device)
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ctx_ex_feats = ex_feats[:, :C, :]
            ctx_encoder_input["ex_feats"] = ctx_ex_feats
            encoder_input["ex_feats"] = ex_feats

        ctx_embedding = self.ctx_encoder(ctx_encoder_input) # embedded c (B, C, n)
        ctx_embedding_dim = ctx_embedding.shape[2]
        ctx_embedding_padded = torch.zeros((B, T, ctx_embedding_dim)).to(self.device)
        ctx_embedding_padded[:, :C, :] = ctx_embedding
        
        z_mean, z_log_var, z = self.encoder(encoder_input) # P(z|c,x), (B, T, latent_dim)

        decoder_input = torch.cat([ctx_embedding_padded, z], dim=-1)
        if "ex_feats" in x:
            decoded_surface, decoded_ex_feat = self.decoder(decoder_input) # P(x|c,z,t)

            return decoded_surface[:, C:, :, :], decoded_ex_feat[:, C:, :], z_mean, z_log_var, z
        else:
            decoded_surface = self.decoder(decoder_input) # P(x|c,z,t)

            return decoded_surface[:, C:, :, :], z_mean, z_log_var, z

    def train_step(self, x, optimizer: torch.optim.Optimizer):
        '''
            Input:
                x should be a dictionary with 2 keys:
                - surface: volatility surface of shape (B,T,H,W), 
                    surface[:,:context_len,:,:] are the context surfaces (previous days),
                    surface[:,context_len:,:,:] is the surface to predict
                - ex_feats: extra features of shape (B,T,n)
        '''

        surface = x["surface"]
        if len(surface.shape) == 3:
            # unbatched data
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - 1
        surface_real = surface[:,C:, :, :].to(self.device)

        if "ex_feats" in x:
            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ex_feats_real = ex_feats[:, C:, :,].to(self.device)

        optimizer.zero_grad()
        if "ex_feats" in x:
            surface_reconstruciton, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x)
        else:
            surface_reconstruciton, z_mean, z_log_var, z = self.forward(x)

        # RE = 1/M \sum_{i=1}^M (x_i - y_i)^2
        re_surface = F.mse_loss(surface_reconstruciton, surface_real)
        if "ex_feats" in x:
            if self.config["ex_loss_on_ret_only"]:
                ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                ex_feats_real = ex_feats_real[:, :, :1]
            re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
            reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
        else:
            reconstruction_error = re_surface
        # KL = -1/2 \sum_{i=1}^M (1+log(\sigma_k^2) - \sigma_k^2 - \mu_k^2)
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))
        total_loss = reconstruction_error + self.kl_weight * kl_loss
        total_loss.backward()
        optimizer.step()

        return {
            "loss": total_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex_feats if "ex_feats" in x else torch.zeros(1),
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }
    
    def test_step(self, x):
        surface = x["surface"]
        if len(surface.shape) == 3:
            # unbatched data
            surface = surface.unsqueeze(0)
        B = surface.shape[0]
        T = surface.shape[1]
        C = T - 1
        surface_real = surface[:, C:, :, :].to(self.device)

        if "ex_feats" in x:
            ex_feats = x["ex_feats"]
            if len(ex_feats.shape) == 2:
                ex_feats = ex_feats.unsqueeze(0)
            ex_feats_real = ex_feats[:, C:, :,].to(self.device)
        
        if "ex_feats" in x:
            surface_reconstruciton, ex_feats_reconstruction, z_mean, z_log_var, z = self.forward(x)
        else:
            surface_reconstruciton, z_mean, z_log_var, z = self.forward(x)

        # RE = 1/M \sum_{i=1}^M (x_i - y_i)^2
        re_surface = F.mse_loss(surface_reconstruciton, surface_real)
        if "ex_feats" in x:
            if self.config["ex_loss_on_ret_only"]:
                ex_feats_reconstruction = ex_feats_reconstruction[:, :, :1]
                ex_feats_real = ex_feats_real[:, :, :1]
            re_ex_feats = self.ex_feats_loss_fn(ex_feats_reconstruction, ex_feats_real)
            reconstruction_error = re_surface + self.config["re_feat_weight"] * re_ex_feats
        else:
            reconstruction_error = re_surface
        # KL = -1/2 \sum_{i=1}^M (1+log(\sigma_k^2) - \sigma_k^2 - \mu_k^2)
        kl_loss = -0.5 * (1 + z_log_var - torch.exp(z_log_var) - torch.square(z_mean))
        kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))
        total_loss = reconstruction_error + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "re_surface": re_surface,
            "re_ex_feats": re_ex_feats if "ex_feats" in x else torch.zeros(1),
            "reconstruction_loss": reconstruction_error,
            "kl_loss": kl_loss,
        }