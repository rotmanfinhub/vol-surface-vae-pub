import numpy as np
import torch
from torch.utils.data import Dataset

class VolSurfaceDataSet(Dataset):
    def __init__(self, dataset, seq_len, dtype=torch.float64):
        '''
            Inputs:
                dataset: dataset numpy array (N, feat_dims)
                seq_len: seq_len for each sample
        '''
        self.dataset = torch.from_numpy(dataset)
        if dtype == torch.float32:
            self.dataset = self.dataset.float()
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.dataset) - self.seq_len + 1

    def __getitem__(self, idx):
        ele = self.dataset[idx:idx+self.seq_len]
        return ele
    
class VolSurfaceDataSetDict(Dataset):
    def __init__(self, dataset, seq_len, dtype=torch.float64):
        '''
            Inputs:
                dataset: dataset numpy array (N, feat_dims)
                seq_len: seq_len for each sample
        '''
        self.dataset = torch.from_numpy(dataset)
        if dtype == torch.float32:
            self.dataset = self.dataset.float()
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.dataset) - self.seq_len + 1

    def __getitem__(self, idx):
        ele = self.dataset[idx:idx+self.seq_len]
        return {"surface": ele}

class VolSurfaceExFeatsDataSet(Dataset):
    def __init__(self, dataset, ex_feats, seq_len, dtype=torch.float64):
        '''
            Inputs:
                dataset: dataset numpy array (N, feat_dims)
                ex_feats: numpy array for extra features, 
                should have the same length as dataset (matching timesteps) (N, ex_feat_dims). 
                if it has shape (N,), it will be extended to (N, 1).
                seq_len: seq_len for each sample
        '''
        self.dataset = torch.from_numpy(dataset)
        self.ex_feats = torch.from_numpy(ex_feats)
        if dtype == torch.float32:
            self.dataset = self.dataset.float()
            self.ex_feats = self.ex_feats.float()
        assert len(self.dataset) == len(self.ex_feats), "dataset and ex_feats should have the same length"
        if len(self.ex_feats.shape) == 1:
            self.ex_feats = self.ex_feats.unsqueeze(-1)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.dataset) - self.seq_len + 1

    def __getitem__(self, idx):
        ds_ele = self.dataset[idx:idx+self.seq_len]
        feat_ele = self.ex_feats[idx:idx+self.seq_len]
        return {"surface": ds_ele, "ex_feats": feat_ele}

class SABRDataset(Dataset):
    def __init__(self, dataset, seq_len, dtype=torch.float64):
        '''
            Inputs:
                dataset: dataset numpy array (num_paths, num_period, ttm_grid, moneyness_grid)
                ex_feats: numpy array for extra features, 
                should have the same length as dataset (matching timesteps) (num_paths, num_period, ex_feat_dims). 
                if it has shape (num_paths, num_period, ), it will be extended to (num_paths, num_period, 1).
                seq_len: seq_len for each sample
        '''
        assert seq_len <= dataset.shape[1], "for SABR dataset, seq_len must be <= num_period"
        self.seq_len = seq_len

        # transform to (N, seq_len, ttm_grid, moneyness_grid)
        dataset = np.lib.stride_tricks.sliding_window_view(dataset, seq_len, axis=1).transpose([0, 1, 4, 2, 3]).reshape((-1, seq_len, dataset.shape[2], dataset.shape[3]))

        self.dataset = torch.from_numpy(dataset)
        if dtype == torch.float32:
            self.dataset = self.dataset.float()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return {"surface": self.dataset[idx]}

class SABRExFeatsDataset(Dataset):
    def __init__(self, dataset, ex_feats, seq_len, dtype=torch.float64):
        '''
            Inputs:
                dataset: dataset numpy array (num_paths, num_period, ttm_grid, moneyness_grid)
                ex_feats: numpy array for extra features, 
                should have the same length as dataset (matching timesteps) (num_paths, num_period, ex_feat_dims). 
                if it has shape (num_paths, num_period, ), it will be extended to (num_paths, num_period, 1).
                seq_len: seq_len for each sample
        '''
        assert seq_len <= dataset.shape[1], "for SABR dataset, seq_len must be <= num_period"
        assert dataset.shape[0] == ex_feats.shape[0] and dataset.shape[1] == ex_feats.shape[1], "dataset and ex_feats should have the same length"
        self.seq_len = seq_len

        # transform to (N, seq_len, ttm_grid, moneyness_grid)
        dataset = np.lib.stride_tricks.sliding_window_view(dataset, seq_len, axis=1).transpose([0, 1, 4, 2, 3]).reshape((-1, seq_len, dataset.shape[2], dataset.shape[3]))

        if len(ex_feats.shape) == 2:
            feat_size = 1
            ex_feats = ex_feats.reshape((ex_feats.shape[0], ex_feats.shape[1], 1))
        else:
            feat_size = ex_feats.shape[2]
        ex_feats = np.lib.stride_tricks.sliding_window_view(ex_feats, seq_len, axis=1).transpose([0, 1, 3, 2]).reshape((-1, seq_len, feat_size))
        
        self.dataset = torch.from_numpy(dataset)
        self.ex_feats = torch.from_numpy(ex_feats)
        if dtype == torch.float32:
            self.dataset = self.dataset.float()
            self.ex_feats = self.ex_feats.float()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return {"surface": self.dataset[idx], "ex_feats": self.ex_feats[idx]}
