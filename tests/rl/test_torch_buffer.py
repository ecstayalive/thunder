import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def setup_torch_env():
    os.environ["THUNDER_BACKEND"] = "torch"
    yield


_THUNDER_PATH = Path(__file__).resolve().parents[2] / "thunder"

_DATA_SPEC = importlib.util.spec_from_file_location(
    "thunder_core_data_buffer_test", _THUNDER_PATH / "core" / "data.py"
)
data_mod = importlib.util.module_from_spec(_DATA_SPEC)
assert _DATA_SPEC.loader is not None
sys.modules[_DATA_SPEC.name] = data_mod
_DATA_SPEC.loader.exec_module(data_mod)


class _ExecutorStub:
    @staticmethod
    def default_device(device=None):
        return torch.device("cpu" if device is None else device)


_thunder_mod = types.ModuleType("thunder")
_core_mod = types.ModuleType("thunder.core")
_core_mod.Batch = data_mod.Batch
_core_mod.Executor = _ExecutorStub
sys.modules.setdefault("thunder", _thunder_mod)
sys.modules["thunder.core"] = _core_mod

_BUFFER_SPEC = importlib.util.spec_from_file_location(
    "thunder_rl_torch_buffer_test", _THUNDER_PATH / "rl" / "torch" / "buffer.py"
)
buffer_mod = importlib.util.module_from_spec(_BUFFER_SPEC)
assert _BUFFER_SPEC.loader is not None
sys.modules[_BUFFER_SPEC.name] = buffer_mod
_BUFFER_SPEC.loader.exec_module(buffer_mod)

Buffer = buffer_mod.Buffer
BufferLoader = buffer_mod.BufferLoader
ChunkBufferSampler = buffer_mod.ChunkBufferSampler
RandomBufferSampler = buffer_mod.RandomBufferSampler
SequenceSampler = buffer_mod.SequenceSampler


class TestBuffer:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

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
            terminated = torch.zeros((2, 1), device=device)
            timeouts = torch.zeros((2, 1), device=device)
            next_obs = {"policy": torch.tensor([[float(i + 1)], [float(i + 101)]], device=device)}
            # Add explicit done at index 5 and 15 for Env 0 for segmentation tests
            if i == 5 or i == 15:
                terminated[0] = 1.0
            t = data_mod.Batch(
                obs=obs,
                next_obs=next_obs,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                timeouts=timeouts,
            )
            buf.add_transition(t)
        return buf

    def test_storage_is_batch_first(self, buffer: Buffer):
        assert buffer.size == 20
        assert buffer.ptr == 0
        assert buffer.num_envs == 2
        assert buffer.storage.obs["policy"].shape == (2, 20, 1)
        assert torch.equal(
            buffer.storage.obs["policy"][0, :, 0],
            torch.arange(20, dtype=torch.float32),
        )
        assert torch.equal(
            buffer.storage.obs["policy"][1, :, 0],
            torch.arange(100, 120, dtype=torch.float32),
        )

    def test_random_sampler_returns_flat_transitions(self, buffer: Buffer):
        batch_size = 10
        batch = next(iter(BufferLoader(RandomBufferSampler(batch_size, 1))(buffer.data())))
        assert batch.obs["policy"].shape == (10, 1)  # [Batch, Feature]
        assert batch.rewards.shape == (10, 1)
        valid = (batch.obs["policy"] >= 0) & (batch.obs["policy"] < 20) | (
            batch.obs["policy"] >= 100
        ) & (batch.obs["policy"] < 120)
        assert torch.all(valid)

    def test_chunk_sampler_returns_batch_first_sequences(self, buffer):
        """
        Critical Test: Ensure chunks do not overlap and cover unique slots.
        """
        batch_size = 4
        chunk_len = 5
        sampler = ChunkBufferSampler(batch_size, chunk_len, num_batches=1)
        time_idxs, env_idxs = sampler.sample_chunk_indices(buffer.data(), batch_size, chunk_len)
        time_idxs: torch.Tensor
        env_idxs: torch.Tensor
        assert time_idxs.shape == (batch_size, chunk_len)
        assert env_idxs.shape == (batch_size, chunk_len)
        start_times = time_idxs[:, 0]
        start_envs = env_idxs[:, 0]
        hashes = start_envs * 1000 + start_times
        assert len(torch.unique(hashes)) == batch_size

        batch = next(iter(BufferLoader(sampler)(buffer.data())))
        assert batch.obs["policy"].shape == (batch_size, chunk_len, 1)
        assert batch.terminated.shape == (batch_size, chunk_len, 1)
        diffs = batch.obs["policy"][:, 1:, 0] - batch.obs["policy"][:, :-1, 0]
        assert torch.all(diffs == 1.0)

    def test_sequence_sampler_returns_full_env_sequences(self, buffer):
        batch = next(iter(BufferLoader(SequenceSampler(batch_size=2))(buffer.data())))
        assert batch.obs["policy"].shape == (2, 20, 1)
        assert batch.terminated.shape == (2, 20, 1)
        sorted_first_values = torch.sort(batch.obs["policy"][:, 0, 0]).values
        assert torch.equal(sorted_first_values, torch.tensor([0.0, 100.0]))

    def test_loader_samples_enriched_context_batch(self, buffer):
        data = buffer.data()
        enriched = data.replace(
            advantages=torch.full_like(data.rewards, 7.0),
            returns=torch.full_like(data.rewards, 11.0),
        )

        batch = next(iter(BufferLoader(SequenceSampler(batch_size=2))(enriched)))

        assert batch.advantages.shape == (2, 20, 1)
        assert batch.returns.shape == (2, 20, 1)
        assert torch.all(batch.advantages == 7.0)
        assert torch.all(batch.returns == 11.0)

    def test_circular_storage_keeps_batch_first_logical_order(self, device):
        buffer = Buffer(capacity=4, device=device)
        for i in range(6):
            t = data_mod.Batch(
                obs={"policy": torch.tensor([[float(i)], [float(i + 100)]], device=device)},
                next_obs={"policy": torch.tensor([[float(i + 1)], [float(i + 101)]], device=device)},
                actions=torch.zeros((2, 1), device=device),
                rewards=torch.ones((2, 1), device=device),
                terminated=torch.zeros((2, 1), device=device),
                timeouts=torch.zeros((2, 1), device=device),
            )
            buffer.add_transition(t)

        batch = next(iter(BufferLoader(SequenceSampler(batch_size=2))(buffer.data())))
        sorted_rows = batch.obs["policy"][torch.argsort(batch.obs["policy"][:, 0, 0]), :, 0]
        expected = torch.tensor([[2.0, 3.0, 4.0, 5.0], [102.0, 103.0, 104.0, 105.0]])
        assert torch.equal(sorted_rows, expected)

    def test_clear(self, buffer: Buffer):
        buffer.clear()
        assert buffer.ptr == 0
        assert buffer.size == 0
        assert buffer.storage.obs is not None

    @pytest.mark.skip(reason="segment_batch helpers are not part of the current Buffer API")
    def test_sample_chunks(self, buffer):
        batch_size = 5
        chunk_len = 4
        for batch in buffer.sample_chunks(2, batch_size, chunk_len):
            batch: data_mod.Batch
            assert batch.obs["policy"].shape == (batch_size, chunk_len, 1)
            traj = batch.obs["policy"].squeeze()
            diffs = traj[1:] - traj[:-1]
            is_seq = diffs == 1.0
            assert batch.terminated.shape == (batch_size, chunk_len, 1)

    @pytest.mark.skip(reason="to_batch helper is not part of the current Buffer API")
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

    @pytest.mark.skip(reason="to_batches helper is not part of the current Buffer API")
    def test_to_batches_generator(self, buffer):
        """Test generator yields correct number of mini-batches."""
        num_batches = 2
        batches = list(buffer.to_batches(num_batches))
        assert len(batches) == 2
        b0: data_mod.Batch = batches[0]
        assert b0.obs["policy"].shape == (1, 20, 1)  # [Batch=1, Chunk=Size=20, F=1]

    @pytest.mark.skip(reason="segment_batch helper is not part of the current Buffer API")
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

    @pytest.mark.skip(reason="segment_batch helper is not part of the current Buffer API")
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

    @pytest.mark.skip(reason="segment_batch helper is not part of the current Buffer API")
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


if __name__ == "__main__":
    pytest.main([__file__])
