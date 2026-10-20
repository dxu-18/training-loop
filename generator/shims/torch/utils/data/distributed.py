"""torch-lite DistributedSampler (PyTorch's sharding and padding rules)."""

import math

import torch


class DistributedSampler:
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        import torch.distributed as dist
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset, self.num_replicas, self.rank = dataset, num_replicas, rank
        self.shuffle, self.seed, self.drop_last, self.epoch = shuffle, seed, drop_last, 0
        n = len(dataset)
        if drop_last and n % num_replicas:
            self.num_samples = math.ceil((n - num_replicas) / num_replicas)
        else:
            self.num_samples = math.ceil(n / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))
        if not self.drop_last:
            pad = self.total_size - len(indices)
            if pad <= len(indices):
                indices += indices[:pad]
            else:
                indices += (indices * math.ceil(pad / len(indices)))[:pad]
        else:
            indices = indices[:self.total_size]
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
