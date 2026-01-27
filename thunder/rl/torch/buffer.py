from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.utils._pytree as pytree

from thunder.core import Batch, Executor
from thunder.rl.torch.functional import get_trajectory_lengths


class Buffer:
    """A generic, high-performance Replay Buffer."""

    def __init__(self, capacity: int = 32, device: torch.device = None):
        """Initializes the Buffer object with a specified capacity and device.
        Args:
            capacity: The maximum number of transitions to store.
            device: The device to place the tensors on.
        """
        self.device = Executor.default_device(device)
        self.capacity = capacity

        self.ptr = 0
        self.size = 0
        self.num_envs: Optional[int] = None
        self.storage: Optional[Batch] = None  # [L, B, ...]

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
        return torch.zeros((self.capacity, *data.shape), dtype=data.dtype, device=self.device)

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
        storage_leaf[indices] = val
        return storage_leaf

    def add_transition(self, t: Batch):
        """Adds a transition to the buffer. Handles lazy init and circular pointer."""
        if self.storage is None:
            self.num_envs = t.rewards.shape[0]
            self.storage = pytree.tree_map(lambda x: None, t)
            self._initialized = True
        self.storage = pytree.tree_map(lambda s, d: self._insert(s, d, self.ptr), self.storage, t)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def gather(self, rows: torch.Tensor, cols: torch.Tensor) -> Batch:
        """ """

        def _gather(leaf):
            if leaf is None:
                return None
            return leaf[rows, cols]

        return pytree.tree_map(_gather, self.storage)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Batch:
        head_ptr = 0 if self.size < self.capacity else self.ptr
        idx = (head_ptr + idx) % self.capacity
        return pytree.tree_map(lambda leaf: leaf[idx] if leaf is not None else None, self.storage)


class BufferSampler:
    def __init__(self, buffer: Buffer, batch_size: int, num_batches: int):
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self) -> Iterator[Batch]:
        raise NotImplementedError


class BufferLoader:
    def __init__(self, buffer: Buffer, sampler: BufferSampler | None = None):
        self.buffer = buffer
        self.sampler = (
            sampler
            if sampler is not None
            else RandomBufferSampler(buffer, batch_size=32, num_batches=10)
        )

    def __iter__(self) -> Generator[Batch, None, None]:
        for rows, cols in self.sampler:
            yield self.buffer.gather(rows, cols)


class RandomBufferSampler(BufferSampler):
    """_summary_

    Args:
        BufferSampler (_type_): _description_
    """

    def __init__(self, buffer: Buffer, batch_size: int, num_batches: int):
        super().__init__(buffer, batch_size, num_batches)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        total_transitions = self.buffer.size * self.buffer.num_envs
        if total_transitions < self.batch_size:
            return
        num_batches = min(total_transitions // self.batch_size, self.num_batches)
        if num_batches == 0:
            return
        flat_indices = torch.randint(
            0, total_transitions, (self.batch_size * num_batches,), device=self.buffer.device
        )
        # row (time) = index // num_envs
        # col (env)  = index % num_envs
        rows = torch.div(flat_indices, self.buffer.num_envs, rounding_mode="floor")
        cols = flat_indices % self.buffer.num_envs
        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield rows[start:end], cols[start:end]


class ChunkBufferSampler(BufferSampler):
    """_summary_
    Args:
        BufferSampler (_type_): _description_
    """

    def __init__(self, buffer: Buffer, batch_size: int, chunk_len: int, num_batches: int):
        super().__init__(buffer, batch_size, num_batches)
        self.chunk_len = chunk_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.buffer.size < self.chunk_len:
            return
        total_chunks_needed = self.batch_size * self.num_batches
        max_available_slots = (self.buffer.size // self.chunk_len) * self.buffer.num_envs
        if total_chunks_needed > max_available_slots:
            # raise ValueError(f"Not enough data for unique sampling! Needed {total_chunks_needed}, Has {max_available_slots}")
            num_batches = max_available_slots // self.batch_size
            if num_batches == 0:
                return
            total_chunks_needed = self.batch_size * num_batches
        else:
            num_batches = self.num_batches
        time_indices, env_indices = self.sample_chunk_indices(total_chunks_needed, self.chunk_len)
        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            yield time_indices[start:end], env_indices[start:end]

    def sample_chunk_indices(
        self, batch_size: int, chunk_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates physical indices for chunks using Virtual Logical Indexing.
        Efficiently handles circular buffer wrap-around without data movement.
        """
        slots_per_env = self.buffer.size // chunk_len
        remainder = self.buffer.size % chunk_len
        total_slots = slots_per_env * self.buffer.num_envs
        if batch_size > total_slots:
            raise ValueError(
                f"Cannot sample {batch_size} mutually independent chunks. "
                f"Max capacity is {total_slots}."
            )
        if remainder > 0:
            env_offsets = torch.randint(
                0, remainder + 1, (self.buffer.num_envs,), device=self.buffer.device
            )
        else:
            env_offsets = torch.zeros(
                self.buffer.num_envs, device=self.buffer.device, dtype=torch.long
            )
        slot_ids = torch.randperm(total_slots, device=self.buffer.device)[:batch_size]
        env_ids = slot_ids % self.buffer.num_envs
        time_block_ids = torch.div(slot_ids, self.buffer.num_envs, rounding_mode="floor")  # [Batch]
        specific_offsets = env_offsets[env_ids]
        logical_starts = specific_offsets + (time_block_ids * chunk_len)

        head_ptr = 0 if self.buffer.size < self.buffer.capacity else self.buffer.ptr
        physical_starts = (head_ptr + logical_starts) % self.buffer.capacity
        seq_offsets = torch.arange(chunk_len, device=self.buffer.device).unsqueeze(0)
        time_indices = (physical_starts.unsqueeze(1) + seq_offsets) % self.buffer.capacity
        env_indices = env_ids.unsqueeze(1).expand(-1, chunk_len)
        return time_indices, env_indices


class SequenceSampler(BufferSampler):
    def __init__(self, buffer: Buffer, batch_size: int):
        """ """
        super().__init__(buffer, batch_size, num_batches=0)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        seq_len = self.buffer.size
        num_envs = self.buffer.num_envs

        if seq_len == 0:
            return
        num_batches = num_envs // self.batch_size
        if num_batches == 0:
            if num_envs > 0:
                num_batches = 1
                actual_batch_size = num_envs
            else:
                return
        else:
            actual_batch_size = self.batch_size
        env_perm = torch.randperm(num_envs, device=self.buffer.device)

        head_ptr = 0 if self.buffer.size < self.buffer.capacity else self.buffer.ptr
        time_seq = torch.arange(seq_len, device=self.buffer.device) + head_ptr
        time_seq = time_seq % self.buffer.capacity
        for i in range(num_batches):
            start = i * actual_batch_size
            end = start + actual_batch_size
            selected_env_ids = env_perm[start:end]
            batch_time_indices = time_seq.unsqueeze(0).expand(len(selected_env_ids), -1)
            batch_env_indices = selected_env_ids.unsqueeze(1).expand(-1, seq_len)
            yield batch_time_indices, batch_env_indices
