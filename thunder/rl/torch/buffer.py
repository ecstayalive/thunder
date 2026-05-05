from __future__ import annotations

from typing import Any, Generator, Iterator, Optional, Tuple

import torch
import torch.utils._pytree as pytree

from thunder.core import Batch, Executor


def gather(data: Batch, rows: torch.Tensor, cols: torch.Tensor) -> Batch:
    """ """

    def _gather(leaf):
        if leaf is None:
            return None
        return leaf[cols, rows]

    return pytree.tree_map(_gather, data)


class Buffer:
    """A generic, high-performance Replay Buffer."""

    def __init__(
        self, capacity: int = 32, device: torch.device = Executor.default_device()
    ):
        """Initializes the Buffer object with a specified capacity and device.
        Args:
            capacity: The maximum number of transitions to store.
            device: The device to place the tensors on.
        """
        self.capacity = capacity

        self.ptr = 0
        self.size = 0
        self.num_envs: Optional[int] = None
        self.storage: Optional[Batch] = None  # [B, L, ...]
        self.device = device

    def data(self):
        """ """
        if self.storage is None:
            return None
        if self.size < self.capacity:
            return pytree.tree_map(
                lambda leaf: leaf[:, : self.size] if leaf is not None else None,
                self.storage,
            )
        if self.ptr == 0:
            return self.storage
        time_indices = (
            torch.arange(self.size, device=self.device, dtype=torch.long) + self.ptr
        ) % self.capacity
        return pytree.tree_map(
            lambda leaf: leaf[:, time_indices] if leaf is not None else None,
            self.storage,
        )

    def _alloc_leaf(self, data: Any) -> Any:
        """"""
        if data is None:
            return None
        if isinstance(data, (tuple, list)):
            return type(data)(self._alloc_leaf(x) for x in data)
        elif isinstance(data, dict):
            return {k: self._alloc_leaf(v) for k, v in data.items()}
        else:
            data = torch.as_tensor(data, device=self.device)
        data: torch.Tensor
        if data.ndim == 0:
            raise ValueError(
                "buffer transition leaves must include an environment dimension"
            )
        return torch.zeros(
            (data.shape[0], self.capacity, *data.shape[1:]),
            dtype=data.dtype,
            device=self.device,
        )

    def _insert(self, storage_leaf: Any, data_leaf: Any, indices: torch.Tensor) -> Any:
        """ """
        if storage_leaf is None:
            if data_leaf is not None:
                storage_leaf = self._alloc_leaf(data_leaf)
            else:
                return None
        if data_leaf is None:
            return storage_leaf
        # Only enter once when the type is changed
        # Usually for hidden states that are tuples/lists/dicts
        if isinstance(data_leaf, (tuple, list)):
            for s, d in zip(storage_leaf, data_leaf):
                self._insert(s, d, indices)
            return storage_leaf
        elif isinstance(data_leaf, dict):
            for k in data_leaf:
                self._insert(storage_leaf[k], data_leaf[k], indices)
            return storage_leaf
        else:
            val = data_leaf.to(self.device)
        storage_leaf[:, indices] = val
        return storage_leaf

    def add_transition(self, t: Batch):
        """Adds a transition to the buffer. Handles lazy init and circular pointer."""
        if self.storage is None:
            self.num_envs = t.rewards.shape[0]
            self.storage = pytree.tree_map(lambda x: None, t)
            self._initialized = True
        self.storage = pytree.tree_map(
            lambda s, d: self._insert(s, d, self.ptr), self.storage, t
        )
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Batch:
        head_ptr = 0 if self.size < self.capacity else self.ptr
        idx = (head_ptr + idx) % self.capacity
        return pytree.tree_map(
            lambda leaf: leaf[:, idx] if leaf is not None else None, self.storage
        )


class BatchSampler:
    def __init__(self, batch_size: int, num_batches: int):
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __call__(self, data: Batch) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError


class MiniBatchLoader:
    """ """

    def __init__(self, sampler: BatchSampler | None = None):
        self.sampler = (
            sampler
            if sampler is not None
            else RandomBatchSampler(batch_size=32, num_batches=10)
        )

    def __call__(self, data: Batch) -> Generator[Batch, None, None]:
        for rows, cols in self.sampler(data):
            yield gather(data, rows, cols)


class RandomBatchSampler(BatchSampler):
    """_summary_

    Args:
        BufferSampler (_type_): _description_
    """

    def __init__(self, batch_size: int, num_batches: int):
        super().__init__(batch_size, num_batches)

    def __call__(self, data: Batch) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rewards: torch.Tensor = data.rewards
        N, L, device = rewards.shape[0], rewards.shape[1], rewards.device
        total_transitions = L * N
        if total_transitions < self.batch_size:
            return data
        num_batches = min(total_transitions // self.batch_size, self.num_batches)
        if num_batches == 0:
            return
        flat_indices = torch.randint(
            0,
            total_transitions,
            (self.batch_size * num_batches,),
            device=device,
        )
        # row (time) = index // num_envs
        # col (env)  = index % num_envs
        rows = torch.div(flat_indices, N, rounding_mode="floor")
        cols = flat_indices % N
        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield rows[start:end], cols[start:end]


class ChunkBatchSampler(BatchSampler):
    """_summary_
    Args:
        BufferSampler (_type_): _description_
    """

    def __init__(self, batch_size: int, chunk_len: int, num_batches: int):
        super().__init__(batch_size, num_batches)
        self.chunk_len = chunk_len

    def __call__(self, data: Batch) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rewards: torch.Tensor = data.rewards
        N, L, _ = rewards.shape[0], rewards.shape[1], rewards.device
        if L < self.chunk_len:
            return
        total_chunks_needed = self.batch_size * self.num_batches
        max_available_slots = (L // self.chunk_len) * N
        if total_chunks_needed > max_available_slots:
            # raise ValueError(f"Not enough data for unique sampling! Needed {total_chunks_needed}, Has {max_available_slots}")
            num_batches = max_available_slots // self.batch_size
            if num_batches == 0:
                return
            total_chunks_needed = self.batch_size * num_batches
        else:
            num_batches = self.num_batches
        time_indices, env_indices = self.sample_chunk_indices(
            data, total_chunks_needed, self.chunk_len
        )
        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield time_indices[start:end], env_indices[start:end]

    def sample_chunk_indices(
        self, data: Batch, batch_size: int, chunk_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate non-overlapping logical chunk indices from a batch-first rollout.
        """
        rewards: torch.Tensor = data.rewards
        N, L, device = rewards.shape[0], rewards.shape[1], rewards.device
        slots_per_env = L // chunk_len
        remainder = L % chunk_len
        total_slots = slots_per_env * N
        if batch_size > total_slots:
            raise ValueError(
                f"Cannot sample {batch_size} mutually independent chunks. "
                f"Max capacity is {total_slots}."
            )
        if remainder > 0:
            env_offsets = torch.randint(0, remainder + 1, (N,), device=device)
        else:
            env_offsets = torch.zeros(N, device=device, dtype=torch.long)
        slot_ids = torch.randperm(total_slots, device=device)[:batch_size]
        env_ids = slot_ids % N
        time_block_ids = torch.div(slot_ids, N, rounding_mode="floor")  # [Batch]
        specific_offsets = env_offsets[env_ids]
        logical_starts = specific_offsets + (time_block_ids * chunk_len)

        seq_offsets = torch.arange(chunk_len, device=device).unsqueeze(0)
        time_indices = logical_starts.unsqueeze(1) + seq_offsets
        env_indices = env_ids.unsqueeze(1).expand(-1, chunk_len)
        return time_indices, env_indices


class SequenceBatchSampler(BatchSampler):
    def __init__(self, batch_size: int):
        """ """
        super().__init__(batch_size, num_batches=0)

    def __call__(self, data: Batch) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        rewards: torch.Tensor = data.rewards
        N, L, device = rewards.shape[0], rewards.shape[1], rewards.device
        if L == 0:
            return
        num_batches = N // self.batch_size
        if num_batches == 0:
            if N > 0:
                num_batches = 1
                actual_batch_size = N
            else:
                return
        else:
            actual_batch_size = self.batch_size
        env_perm = torch.randperm(N, device=device)

        time_seq = torch.arange(L, device=device)
        for i in range(num_batches):
            start = i * actual_batch_size
            end = start + actual_batch_size
            selected_env_ids = env_perm[start:end]
            batch_time_indices = time_seq.unsqueeze(0).expand(len(selected_env_ids), -1)
            batch_env_indices = selected_env_ids.unsqueeze(1).expand(-1, L)
            yield batch_time_indices, batch_env_indices
