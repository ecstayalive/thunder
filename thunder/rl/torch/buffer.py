from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from thunder.core.data import Batch
from thunder.rl.torch.functional import get_trajectory_lengths


@dataclass(slots=True, init=False)
class Transition:
    """The Transition class represents a transition in a reinforcement
    learning buffer.
    Args:
        obs: Observations. :shape: `[num_envs, obs_features...]`
        actions: The output of the actor. :shape: `[num_envs, action_dimension]`
        reward: The reward obtained by the agent. :shape: `[num_envs, 1]` for
            scalar reward or `[num_envs, d_{reward}]` for multi-object reward.
        dones: Whether the environment is done. :shape: `[num_envs,]`
        timeouts: Whether the agent has been running in the environment for
            longer than the specified time. :shape: `[num_envs,]`
        next_obs: Observation after agent act in environment. For every
            item in observation :shape: `[num_envs, ...]`
    """

    obs: Dict[str, Any] = None
    actions: Any = None
    rewards: Any = None
    dones: Any = None
    timeouts: Any = None
    next_obs: Dict[str, Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        obs: Dict[str, Any] = None,
        actions: Any = None,
        rewards: Any = None,
        dones: Any = None,
        timeouts: Any = None,
        next_obs: Dict[str, Any] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.timeouts = timeouts
        self.next_obs = next_obs
        if extra is None:
            extra = {}
        extra.update(kwargs)

        self.extra = extra

    def __getattr__(self, name: str) -> Any:
        """ """
        try:
            extra = object.__getattribute__(self, "extra")
            return extra[name]
        except (AttributeError, KeyError):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """ """
        if name in self.__class__.__dataclass_fields__:
            object.__setattr__(self, name, value)
        else:
            self.extra[name] = value

    def __dir__(self):
        """ """
        return list(self.__class__.__dataclass_fields__.keys()) + list(self.extra.keys())


class Buffer:
    """
    A generic, high-performance Replay Buffer.
    Features:
        Recursive Lazy Initialization (supports nested dicts)
        Random Sampling (i.i.d)
        Chunk Sampling (Sequence/Trajectory) with Wrap-around support
        Automatic Mask generation based on 'dones'
    """

    def __init__(self, capacity: int, device: torch.device = None):
        """Initializes the Buffer object with a specified length and device.
        Args:
            length: The maximum number of transitions to store.
            device: The device to place the tensors on.
        """
        self.device = device
        self.capacity = capacity

        self.ptr = 0
        self.size = 0
        self.num_envs: Optional[int] = None
        self._initialized: bool = False

        self.obs: Optional[torch.Tensor] = None
        self.next_obs: Optional[torch.Tensor] = None
        self.rewards: Optional[torch.Tensor] = None
        self.actions: Optional[torch.Tensor] = None
        self.dones: Optional[torch.Tensor] = None
        self.timeouts: Optional[torch.Tensor] = None
        self.extra = {}

    def _recursive_alloc(self, data: Any) -> Any:
        """Recursively allocates tensor buffers matching the structure of 'data'."""
        if isinstance(data, dict):
            return {k: self._recursive_alloc(v) for k, v in data.items()}
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            t_data = torch.as_tensor(data, device=self.device)
            return torch.zeros(
                (self.capacity, *t_data.shape), dtype=t_data.dtype, device=self.device
            )
        else:
            raise ValueError(f"Unsupported data type for buffer allocation: {type(data)}")

    def _recursive_insert(self, storage_node: Any, data_node: Any, idx: int):
        """Recursively inserts data into the buffer at index 'idx'."""
        if isinstance(storage_node, dict):
            for k, v in data_node.items():
                self._recursive_insert(storage_node[k], v, idx)
        elif isinstance(storage_node, (torch.Tensor, np.ndarray)):
            storage_node[idx] = torch.as_tensor(data_node, device=self.device)

    def _recursive_get(self, storage_node: Any, indices: torch.Tensor) -> Any:
        """Retrieves data at specific indices."""
        if isinstance(storage_node, dict):
            return {k: self._recursive_get(v, indices) for k, v in storage_node.items()}
        else:
            return storage_node[indices]

    def _lazy_init(self, t: Transition):
        """Initializes storage based on the first Transition received."""
        self.num_envs = t.rewards.shape[0]
        self.obs = self._recursive_alloc(t.obs)
        self.next_obs = self._recursive_alloc(t.next_obs)
        self.actions = self._recursive_alloc(t.actions)
        self.rewards = self._recursive_alloc(t.rewards)
        self.dones = self._recursive_alloc(t.dones)
        self.timeouts = self._recursive_alloc(t.timeouts)
        self.extra = self._recursive_alloc(t.extra)
        self._initialized = True

    def add_transition(self, t: Transition):
        """Adds a transition to the buffer. Handles lazy init and circular pointer."""
        if not self._initialized:
            self._lazy_init(t)
        self._recursive_insert(self.obs, t.obs, self.ptr)
        self._recursive_insert(self.next_obs, t.next_obs, self.ptr)
        self._recursive_insert(self.actions, t.actions, self.ptr)
        self._recursive_insert(self.rewards, t.rewards, self.ptr)
        self._recursive_insert(self.dones, t.dones, self.ptr)
        self._recursive_insert(self.timeouts, t.timeouts, self.ptr)
        if t.extra:
            self._recursive_insert(self.extra, t.extra, self.ptr)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_random(self, batch_size: int) -> Batch:
        """
        Samples independent transitions from the buffer.
        Output Shape: [Batch, Features]
        """
        if self.size == 0:
            return Batch(obs={})

        total_transitions = self.size * self.num_envs
        flat_indices = torch.randint(0, total_transitions, (batch_size,), device=self.device)

        # row (time) = index // num_envs
        # col (env)  = index % num_envs
        rows = torch.div(flat_indices, self.num_envs, rounding_mode="floor")
        cols = flat_indices % self.num_envs
        indices = (rows, cols)
        return Batch(
            obs=self._recursive_get(self.obs, indices),
            actions=self._recursive_get(self.actions, indices),
            rewards=self._recursive_get(self.rewards, indices),
            dones=self._recursive_get(self.dones, indices),
            timeouts=self._recursive_get(self.timeouts, indices),
            next_obs=self._recursive_get(self.next_obs, indices),
            extra=self._recursive_get(self.extra, indices),
        )

    def sample_chunk_indices(
        self, batch_size: int, chunk_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates physical indices for chunks using Virtual Logical Indexing.
        Efficiently handles circular buffer wrap-around without data movement.
        """
        slots_per_env = self.size // chunk_len
        remainder = self.size % chunk_len
        total_slots = slots_per_env * self.num_envs
        if batch_size > total_slots:
            raise ValueError(
                f"Cannot sample {batch_size} mutually independent chunks. "
                f"Max capacity is {total_slots}."
            )
        if remainder > 0:
            env_offsets = torch.randint(0, remainder + 1, (self.num_envs,), device=self.device)
        else:
            env_offsets = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        slot_ids = torch.randperm(total_slots, device=self.device)[:batch_size]
        env_ids = slot_ids % self.num_envs
        time_block_ids = torch.div(slot_ids, self.num_envs, rounding_mode="floor")  # [Batch]
        specific_offsets = env_offsets[env_ids]
        logical_starts = specific_offsets + (time_block_ids * chunk_len)

        head_ptr = 0 if self.size < self.capacity else self.ptr
        physical_starts = (head_ptr + logical_starts) % self.capacity
        seq_offsets = torch.arange(chunk_len, device=self.device).unsqueeze(0)
        time_indices = (physical_starts.unsqueeze(1) + seq_offsets) % self.capacity
        env_indices = env_ids.unsqueeze(1).expand(-1, chunk_len)

        return time_indices, env_indices

    def sample_chunk(self, batch_size: int, chunk_len: int) -> Batch:
        """
        Samples Trajectories.
            Samples raw chunks (Time < Size - Len).
            Splits chunks by 'done' signals into multiple valid sub-trajectories.
            Selects topK 'batch_size' trajectories according to trajectory or Uniform sample.
            Pads them to 'chunk_len'.
        Args:

        Output Shape: [Batch, ChunkLen, Features]
        """
        if self.size < chunk_len:
            raise ValueError(f"Buffer size ({self.size}) < chunk length ({chunk_len})")
        time_indices, env_indices = self.sample_chunk_indices(batch_size, chunk_len)
        batch = self._recursive_get(
            {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "dones": self.dones,
                "timeouts": self.timeouts,
                "next_obs": self.next_obs,
                "extra": self.extra,
            },
            (time_indices, env_indices),
        )

        return Batch(
            obs=batch["obs"],
            actions=batch["actions"],
            rewards=batch["rewards"],
            dones=batch["dones"],
            timeouts=batch["timeouts"],
            next_obs=batch["next_obs"],
            extra=batch["extra"],
        )

    def sample_chunks(self, num_batches: int, batch_size: int, chunk_len: int) -> Iterator[Batch]:
        """ """
        if self.size < chunk_len:
            raise ValueError(f"Buffer size ({self.size}) < chunk length ({chunk_len})")

        total_chunks_needed = batch_size * num_batches
        time_indices, env_indices = self.sample_chunk_indices(total_chunks_needed, chunk_len)
        global_batch = self._recursive_get(
            {
                "obs": self.obs,
                "actions": self.actions,
                "rewards": self.rewards,
                "dones": self.dones,
                "timeouts": self.timeouts,
                "next_obs": self.next_obs,
                "extra": self.extra,
            },
            (time_indices, env_indices),
        )
        global_indices = torch.randperm(total_chunks_needed, device=self.device)

        def _slice_batch(data_node: Any, idxs: torch.Tensor) -> Any:
            if isinstance(data_node, dict):
                return {k: _slice_batch(v, idxs) for k, v in data_node.items()}
            return data_node[idxs]

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            mini_batch_indices = global_indices[start:end]

            final_data = _slice_batch(global_batch, mini_batch_indices)

            yield Batch(
                obs=final_data["obs"],
                actions=final_data["actions"],
                rewards=final_data["rewards"],
                dones=final_data["dones"],
                timeouts=final_data["timeouts"],
                next_obs=final_data["next_obs"],
                extra=final_data["extra"],
            )

    def as_batch(self):
        return self.sample_chunk(self.num_envs, self.size)

    def as_batches(self, batch_size: int) -> Iterator[Batch]:
        return self.sample_chunks(self.num_envs // batch_size, batch_size, self.size)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def segment_batch(self, batch: Batch, fix_shape: bool = True, top: bool = True) -> Batch:
        """ """
        if batch.dones is None:
            raise ValueError("Batch must contain 'dones' to perform segmentation.")

        traj_lengths: torch.Tensor = get_trajectory_lengths(batch.dones)
        num_trajs = traj_lengths.shape[0]
        max_len = int(traj_lengths.max().item())
        row_indices = torch.arange(max_len, device=self.device).unsqueeze(0)
        scatter_mask = traj_lengths.unsqueeze(1) > row_indices

        def transform_tensor_core(
            tensor: torch.Tensor, target_idxs: torch.Tensor = None
        ) -> torch.Tensor:
            """ """
            flat_tensor = tensor.flatten(0, 1)
            embedding_shape = flat_tensor.shape[1:]
            # Allocate [Num_Trajs, Max_Len, ...]
            pooled = torch.zeros(
                (num_trajs, max_len, *embedding_shape), dtype=flat_tensor.dtype, device=self.device
            )
            pooled[scatter_mask] = flat_tensor
            if target_idxs is not None:
                # [Target_Batch, Max_Len, F]
                return pooled[target_idxs]
            return pooled

        if not fix_shape:

            def map_fn_var(node: Any) -> Any:
                if torch.is_tensor(node):
                    return transform_tensor_core(node, target_idxs=None)
                return node

            segmented_batch = batch.map(map_fn_var)
            return segmented_batch.replace(mask=scatter_mask)
        target_batch_size = batch.dones.shape[0]
        chunk_len = batch.dones.shape[1]
        pool_size = num_trajs
        if pool_size == target_batch_size:
            idxs = torch.randperm(target_batch_size, device=self.device)
        else:
            if top:
                _, idxs = torch.topk(traj_lengths, k=target_batch_size, largest=True)
                idxs = idxs[torch.randperm(target_batch_size, device=self.device)]
            else:
                idxs = torch.randperm(pool_size, device=self.device)[:target_batch_size]

        def transform_fixed(tensor: torch.Tensor) -> torch.Tensor:
            selected = transform_tensor_core(tensor, target_idxs=idxs)
            curr_len = selected.shape[1]  # Time dim is 1
            pad_size = chunk_len - curr_len
            if pad_size > 0:
                shape = selected.shape
                shape[1] = pad_size
                padding = torch.zeros(shape, dtype=selected.dtype, device=self.device)
                selected = torch.cat([selected, padding], dim=1)
            return selected

        def map_fn_fixed(node: Any) -> Any:
            if torch.is_tensor(node):
                return transform_fixed(node)
            return node

        segmented_batch = batch.map(map_fn_fixed)
        mask_subset = scatter_mask[idxs]
        curr_mask_len = mask_subset.shape[1]
        mask_pad = chunk_len - curr_mask_len
        if mask_pad > 0:
            padding = torch.zeros(
                (target_batch_size, mask_pad), dtype=torch.bool, device=self.device
            )
            mask_subset = torch.cat([mask_subset, padding], dim=1)

        return segmented_batch.replace(mask=mask_subset)

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Transition:
        head_ptr = 0 if self.size < self.capacity else self.ptr
        idx = (head_ptr + idx) % self.capacity
        return Transition(
            obs=self._recursive_get(self.obs, idx),
            actions=self._recursive_get(self.actions, idx),
            rewards=self._recursive_get(self.rewards, idx),
            dones=self._recursive_get(self.dones, idx),
            timeouts=self._recursive_get(self.timeouts, idx),
            next_obs=self._recursive_get(self.next_obs, idx),
            extra=self._recursive_get(self.extra, idx),
        )
