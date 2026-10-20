"""torch-lite torch.utils.data: datasets, samplers, default_collate and a DataLoader.

Workers are simulated in-process: batches are dispatched round-robin to `num_workers` workers, and
`get_worker_info()` reports the worker that is producing the current item, as in PyTorch."""

import collections
import copy
import math

import numpy as np

import torch

from . import distributed  # noqa: F401


class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __add__(self, other):
        return ConcatDataset([self, other])


class IterableDataset(Dataset):
    def __iter__(self):
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        if any(tensors[0].shape[0] != t.shape[0] for t in tensors):
            raise AssertionError("Size mismatch between tensors")
        self.tensors = tensors

    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __getitem__(self, i):
        if isinstance(i, list):
            return [self.dataset[self.indices[j]] for j in i]
        return self.dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = list(datasets)
        if not self.datasets:
            raise AssertionError("datasets should not be an empty iterable")
        self.cumulative_sizes = list(np.cumsum([len(d) for d in self.datasets]).tolist())

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, i):
        if i < 0:
            if -i > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            i += len(self)
        from bisect import bisect_right
        d = bisect_right(self.cumulative_sizes, i)
        start = 0 if d == 0 else self.cumulative_sizes[d - 1]
        return self.datasets[d][i - start]


class StackDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self._args, self._kwargs = args, kwargs

    def __getitem__(self, i):
        if self._kwargs:
            return {k: d[i] for k, d in self._kwargs.items()}
        return tuple(d[i] for d in self._args)

    def __len__(self):
        return len((self._args or tuple(self._kwargs.values()))[0])


def random_split(dataset, lengths, generator=None):
    n = len(dataset)
    lengths = list(lengths)
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        sizes = [int(math.floor(n * f)) for f in lengths]
        for i in range(n - sum(sizes)):  # leftover items go round-robin to the splits
            sizes[i % len(sizes)] += 1
        lengths = sizes
    if sum(lengths) != n:
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    perm = torch.randperm(n, generator=generator).tolist()
    out, off = [], 0
    for k in lengths:
        out.append(Subset(dataset, perm[off:off + k]))
        off += k
    return out


# ---------------------------------------------------------------- samplers

class Sampler:
    def __init__(self, data_source=None):
        pass

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        self.data_source, self.replacement, self._num, self.generator = data_source, replacement, num_samples, generator

    @property
    def num_samples(self):
        return len(self.data_source) if self._num is None else self._num

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(0, n, (self.num_samples,), generator=self.generator).tolist())
        out = []
        for _ in range(self.num_samples // n):
            out += torch.randperm(n, generator=self.generator).tolist()
        out += torch.randperm(n, generator=self.generator).tolist()[:self.num_samples % n]
        return iter(out)

    def __len__(self):
        return self.num_samples


class SubsetRandomSampler(Sampler):
    def __init__(self, indices, generator=None):
        self.indices, self.generator = indices, generator

    def __iter__(self):
        return iter([self.indices[i] for i in torch.randperm(len(self.indices), generator=self.generator).tolist()])

    def __len__(self):
        return len(self.indices)


class WeightedRandomSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True, generator=None):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={num_samples}")
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_samples, self.replacement, self.generator = num_samples, replacement, generator

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator).tolist())

    def __len__(self):
        return self.num_samples


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        self.sampler, self.batch_size, self.drop_last = sampler, batch_size, drop_last

    def __iter__(self):
        batch = []
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------- collate

def default_collate(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        shapes = [tuple(t.shape) for t in batch]
        for i, s in enumerate(shapes):
            if s != shapes[0]:
                raise RuntimeError(f"stack expects each tensor to be equal size, but got {list(shapes[0])} at entry 0 "
                                   f"and {list(s)} at entry {i}")
        return torch.stack(batch)
    if isinstance(elem, np.ndarray):
        return default_collate([torch.as_tensor(b) for b in batch])
    if isinstance(elem, np.generic):
        return torch.as_tensor(np.array(batch))
    if isinstance(elem, bool):
        return torch.tensor(batch)
    if isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    if isinstance(elem, int):
        return torch.tensor(batch)
    if isinstance(elem, (str, bytes)):
        return list(batch)
    if isinstance(elem, collections.abc.Mapping):
        return type(elem)({k: default_collate([d[k] for d in batch]) for k in elem}) if type(elem) is not dict else \
            {k: default_collate([d[k] for d in batch]) for k in elem}
    if isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return type(elem)(*(default_collate(list(s)) for s in zip(*batch)))
    if isinstance(elem, collections.abc.Sequence):
        if any(len(e) != len(elem) for e in batch):
            raise RuntimeError("each element in list of batch should be of equal size")
        return [default_collate(list(s)) for s in zip(*batch)]
    raise TypeError(f"default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {type(elem)}")


def default_convert(data):
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data)
    return data


# ---------------------------------------------------------------- workers

WorkerInfo = collections.namedtuple("WorkerInfo", ["id", "num_workers", "seed", "dataset"])
_WORKER = [None]


def get_worker_info():
    return _WORKER[0]


class _Worker:
    def __init__(self, i, n, seed, dataset, init_fn):
        self.info = WorkerInfo(i, n, seed + i, dataset)
        self.init_fn = init_fn
        self.started = False

    def __enter__(self):
        _WORKER[0] = self.info
        self._saved = torch.get_rng_state()
        torch.manual_seed(self.info.seed)
        if self.started:
            torch.set_rng_state(self.rng)
        elif self.init_fn is not None:
            self.init_fn(self.info.id)
        self.started = True
        return self

    def __exit__(self, *exc):
        self.rng = torch.get_rng_state()
        torch.set_rng_state(self._saved)
        _WORKER[0] = None
        return False


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device="", in_order=True):
        if num_workers < 0:
            raise ValueError("num_workers option should be non-negative; use num_workers=0 to disable multiprocessing.")
        if num_workers == 0 and (prefetch_factor is not None or persistent_workers):
            raise ValueError("prefetch_factor and persistent_workers options require num_workers > 0 (multiprocessing)")
        self.dataset, self.num_workers, self.worker_init_fn = dataset, num_workers, worker_init_fn
        self.pin_memory, self.drop_last, self.generator = pin_memory, drop_last, generator
        self.prefetch_factor, self.persistent_workers = prefetch_factor, persistent_workers
        self._iterable = isinstance(dataset, IterableDataset)
        if self._iterable:
            if shuffle:
                raise ValueError(f"DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle={shuffle}")
            if sampler is not None or batch_sampler is not None:
                raise ValueError("DataLoader with IterableDataset: expected unspecified sampler option")
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")
        if batch_sampler is not None and (batch_size != 1 or shuffle or sampler is not None or drop_last):
            raise ValueError("batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last")
        if batch_sampler is not None:
            batch_size, drop_last = None, False
        self.batch_size = batch_size
        if not self._iterable:
            if sampler is None:
                sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
            if batch_size is not None and batch_sampler is None:
                batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.sampler, self.batch_sampler = sampler, batch_sampler
        self.collate_fn = collate_fn if collate_fn is not None else (default_collate if batch_size is not None or
                                                                     batch_sampler is not None else default_convert)

    def _workers(self):
        seed = int(torch._gen(self.generator).integers(0, 2 ** 62)) if self.num_workers else 0
        return [_Worker(i, self.num_workers, seed, self.dataset, self.worker_init_fn) for i in range(self.num_workers)]

    def _batches_of(self, it):
        if self.batch_size is None:
            for item in it:
                yield self.collate_fn(item)
            return
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch and not self.drop_last:
            yield self.collate_fn(batch)

    def __iter__(self):
        if self._iterable:
            if not self.num_workers:
                yield from self._batches_of(iter(self.dataset))
                return
            workers = self._workers()
            streams = []
            for w in workers:  # each worker iterates its own copy of the dataset
                try:
                    ds = copy.deepcopy(self.dataset)
                except Exception:
                    ds = self.dataset
                w.info = w.info._replace(dataset=ds)
                with w:
                    streams.append(self._batches_of(iter(ds)))
            live = list(range(len(workers)))
            while live:  # batches come back round-robin across workers
                for i in list(live):
                    with workers[i]:
                        try:
                            b = next(streams[i])
                        except StopIteration:
                            live.remove(i)
                            continue
                    yield b
            return
        workers = self._workers()
        for k, idx in enumerate(self.batch_sampler if self.batch_sampler is not None else self.sampler):
            if workers:
                with workers[k % len(workers)]:
                    yield self._fetch(idx)
            else:
                yield self._fetch(idx)

    def _fetch(self, idx):
        if self.batch_sampler is None:
            return self.collate_fn(self.dataset[idx])
        return self.collate_fn([self.dataset[i] for i in idx])

    def __len__(self):
        if self._iterable:
            n = len(self.dataset)
            if self.batch_size is None:
                return n
            return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        return len(self.batch_sampler) if self.batch_sampler is not None else len(self.sampler)


def __getattr__(name):
    raise AttributeError(f"'torch.utils.data.{name}' is not available in the browser's torch-lite.")
