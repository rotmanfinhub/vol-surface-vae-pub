import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler, Sampler
from typing import Iterator, List
from collections import defaultdict
import random

from torch.utils.data.sampler import Sampler

class GroupedRandomSampler(Sampler):
    def __init__(self, dataset, min_seq_len=4):
        self.dataset = dataset
        self.groups = defaultdict(list)
        for idx in range(len(dataset)):
            seq_len = dataset[idx]["surface"].shape[0]
            if seq_len < min_seq_len:
                continue
            self.groups[seq_len].append(idx)
        # print(self.groups.keys())
    
    def __iter__(self) -> Iterator:
        group_idx = list(self.groups.values())
        random.shuffle(group_idx)
        for group in group_idx:
            random.shuffle(group)
            yield from group
    
    def __len__(self):
        return len(self.dataset)


class CustomBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size: int, min_seq_len=4) -> None:
        super().__init__(GroupedRandomSampler(dataset, min_seq_len), batch_size, drop_last=False)
        self.dataset = dataset
    
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            if not batch:
                batch.append(idx)
            else:
                seq_len = self.dataset[batch[0]]["surface"].shape[0]
                if self.dataset[idx]["surface"].shape[0] == seq_len:
                    batch.append(idx)
                else:
                    yield batch
                    batch = [idx]
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
                
class VolSurfaceDataSetRand(Dataset):
    def __init__(self, dataset, min_seq_len=4, max_seq_len=10, dtype=torch.float64):
        '''
            Inputs:
                dataset: if only surface, then dataset is a numpy array (N, feat_dims), 
                if with ex_feats, then its a tuple (N, feat_dims) and (N, ex_feat_dims)
                should have the same length as dataset (matching timesteps) (N, ex_feat_dims). 
                if it has shape (N,), it will be extended to (N, 1).
                seq_len: seq_len for each sample
        '''
        if isinstance(dataset, tuple):
            self.dataset = torch.from_numpy(dataset[0])
            self.ex_feats = torch.from_numpy(dataset[1])
            if dtype == torch.float32:
                self.dataset = self.dataset.float()
                self.ex_feats = self.ex_feats.float()
            assert len(self.dataset) == len(self.ex_feats), "dataset and ex_feats should have the same length"
            if len(self.ex_feats.shape) == 1:
                self.ex_feats = self.ex_feats.unsqueeze(-1)
        else:
            self.dataset = torch.from_numpy(dataset)
            self.ex_feats = None
            if dtype == torch.float32:
                self.dataset = self.dataset.float()
        self.seq_lens = list(range(min_seq_len, max_seq_len + 1))
    
    def __len__(self):
        return len(self.dataset) * len(self.seq_lens)

    def __getitem__(self, idx):
        seq_len_idx, seq_start_idx = divmod(idx, len(self.dataset))
        seq_len = self.seq_lens[seq_len_idx]
        ds_ele = self.dataset[seq_start_idx:seq_start_idx + seq_len]
        if self.ex_feats is not None:
            feat_ele = self.ex_feats[seq_start_idx:seq_start_idx + seq_len]
            return {"surface": ds_ele, "ex_feats": feat_ele}
        return {"surface": ds_ele}
    
