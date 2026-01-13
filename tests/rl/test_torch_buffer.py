import importlib
import os
import sys

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def setup_torch_env():
    os.environ["THUNDER_BACKEND"] = "torch"
    modules_to_reload = [
        "thunder.core.context",
        "thunder.core.data",
        "thunder.core.module",
        "thunder.core.executor",
        "thunder.core.algorithm",
        "thunder.core.operation",
    ]
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)
    yield


import torch
import torch.nn as nn

import thunder.core.algorithm as algo_mod
import thunder.core.context as ctx_mod
import thunder.core.data as data_mod
import thunder.core.executor as exec_mod
import thunder.core.module as module_mod
import thunder.core.operation as op_mod
from thunder.rl.buffer.torch import Buffer, Transition


class TestBuffer:
    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def buffer(self, device):
        """Creates a small buffer initialized with data."""
        capacity = 20
        buf = Buffer(capacity=capacity, device=device)

        # Populate with deterministic pattern
        # 2 Environments
        # Env 0: [0, 1, 2, ... 19]
        # Env 1: [100, 101, ... 119]
        for i in range(capacity):
            obs = {"policy": torch.tensor([[float(i)], [float(i + 100)]], device=device)}
            actions = torch.zeros((2, 1), device=device)
            rewards = torch.ones((2, 1), device=device)
            dones = torch.zeros((2, 1), device=device)
            timeouts = torch.zeros((2, 1), device=device)
            next_obs = {"policy": torch.tensor([[float(i + 1)], [float(i + 101)]], device=device)}
            # Add explicit done at index 5 and 15 for Env 0 for segmentation tests
            if i == 5 or i == 15:
                dones[0] = 1.0
            t = Transition(
                obs=obs,
                next_obs=next_obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                timeouts=timeouts,
            )
            buf.add_transition(t)
        return buf

    def test_initialization_and_add(self, buffer: Buffer):
        assert buffer.size == 20
        assert buffer.ptr == 0
        assert buffer.num_envs == 2
        assert buffer.obs["policy"].shape == (20, 2, 1)

    def test_sample_random(self, buffer: Buffer):
        batch_size = 10
        batch = buffer.sample_random(batch_size)
        assert batch.obs["policy"].shape == (10, 1)  # [Batch, Feature]
        assert batch.rewards.shape == (10, 1)
        valid = (batch.obs["policy"] >= 0) & (batch.obs["policy"] < 20) | (
            batch.obs["policy"] >= 100
        ) & (batch.obs["policy"] < 120)
        assert torch.all(valid)

    def test_sample_chunk_indices_independence(self, buffer):
        """
        Critical Test: Ensure chunks do not overlap and cover unique slots.
        """
        batch_size = 4
        chunk_len = 5
        time_idxs, env_idxs = buffer.sample_chunk_indices(batch_size, chunk_len)
        time_idxs: torch.Tensor
        env_idxs: torch.Tensor
        assert time_idxs.shape == (batch_size, chunk_len)
        assert env_idxs.shape == (batch_size, chunk_len)
        start_times = time_idxs[:, 0]
        start_envs = env_idxs[:, 0]
        hashes = start_envs * 1000 + start_times
        assert len(torch.unique(hashes)) == batch_size

    def test_sample_chunks(self, buffer):
        """Test sampling raw chunks (sliding window)."""
        batch_size = 5
        chunk_len = 4
        for batch in buffer.sample_chunks(2, batch_size, chunk_len):
            batch: data_mod.Batch
            assert batch.obs["policy"].shape == (batch_size, chunk_len, 1)
            traj = batch.obs["policy"].squeeze()
            diffs = traj[1:] - traj[:-1]
            is_seq = diffs == 1.0
            assert batch.dones.shape == (batch_size, chunk_len, 1)

    def test_to_batch(self, buffer):
        """
        Test retrieving the full buffer for PPO.
        Uses sample_chunk under the hood (make sure sample_chunk logic is correct).
        """
        full_batch: data_mod.Batch = buffer.to_batch()
        assert full_batch.obs["policy"].shape == (2, 20, 1)
        flat_obs = full_batch.obs["policy"].view(-1)
        assert torch.sum(flat_obs < 20) == 20  # 20 items from Env 0
        assert torch.sum(flat_obs >= 100) == 20  # 20 items from Env 1

    def test_to_batches_generator(self, buffer):
        """Test generator yields correct number of mini-batches."""
        num_batches = 2
        batches = list(buffer.to_batches(num_batches))
        assert len(batches) == 2
        b0: data_mod.Batch = batches[0]
        assert b0.obs["policy"].shape == (1, 20, 1)  # [Batch=1, Chunk=Size=20, F=1]

    def test_segment_batch_variable_shape(self, buffer: Buffer):
        """Test fix_shape=False (Return all segments)."""
        obs = buffer.obs["policy"].permute(1, 0, 2)  # [2, 20, 1]
        dones = buffer.dones.permute(1, 0, 2)
        batch = data_mod.Batch(obs=obs, dones=dones)
        segmented = buffer.segment_batch(batch, fix_shape=False)
        assert segmented.obs.shape[0] == 4
        assert segmented.obs.shape[1] == 20
        mask_sums = segmented.mask.sum(dim=1)
        assert 6 in mask_sums
        assert 10 in mask_sums  # 6-15
        assert 4 in mask_sums  # 16-19
        assert 20 in mask_sums  # Env 1

    def test_segment_batch_fixed_shape(self, buffer: Buffer):
        """Test fix_shape=True (Padding and Selection)."""
        obs = buffer.obs["policy"].permute(1, 0, 2)  # [2, 20, 1]
        dones = buffer.dones.permute(1, 0, 2)
        batch = data_mod.Batch(obs=obs, dones=dones)
        segmented = buffer.segment_batch(batch, fix_shape=True, top=True)
        assert segmented.obs.shape == (2, 20, 1)

        # Verify we picked the longest ones
        mask_sums = segmented.mask.sum(dim=1)
        assert 20 in mask_sums
        assert 10 in mask_sums
        assert 6 not in mask_sums

    def test_segment_batch_no_dones_optimization(self, buffer, device):
        """Test the fast path when pool_size == target_batch_size."""
        # Create dummy batch with NO dones
        B, T, F = 4, 10, 1
        obs = {"policy": torch.randn(B, T, F, device=device)}
        dones = torch.zeros(B, T, 1, device=device)  # No splits
        batch = data_mod.Batch(obs=obs, dones=dones)
        segmented: data_mod.Batch = buffer.segment_batch(batch, fix_shape=True)
        assert segmented.obs["policy"].shape == (B, T, F)
        assert torch.all(segmented.mask)  # All True

    def test_clear(self, buffer: Buffer):
        buffer.clear()
        assert buffer.ptr == 0
        assert buffer.size == 0
        assert buffer.obs is not None


if __name__ == "__main__":
    pytest.main([__file__])
