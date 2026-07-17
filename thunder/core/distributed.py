from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterator

_BACKEND = os.getenv("THUNDER_BACKEND", "torch").lower()

__all__ = ["DistributedSpec", "DistributedContextManager", "quiet_unless_main"]


@dataclass(frozen=True)
class DistributedSpec:
    """Resolved topology of the current process (backend-neutral).

    The shared return type of every backend's :class:`DistributedContextManager`.
    Common facts live as plain fields; the backend-specific parallelism handle is
    carried in :attr:`device` (torch DDP) or :attr:`mesh` (jax GSPMD), mirroring the
    ``device``/``mesh`` split already present on
    :class:`~thunder.core.context.ExecutionContextManager`.

    Attributes:
        enabled: ``True`` when launched as a multi-process run.
        rank: Global process rank (``jax.process_index()`` for jax).
        local_rank: Node-local rank, used to select the local accelerator.
        world_size: Total number of processes (``jax.process_count()`` for jax).
        device: Primary device this process binds (``cuda:{local_rank}`` for torch).
        mesh: Device mesh for sharding (jax GSPMD); ``None`` for torch DDP.
    """

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: str
    mesh: Any = None

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@contextlib.contextmanager
def quiet_unless_main(spec: DistributedSpec) -> Iterator[None]:
    """Silence stdout on every non-main rank for the duration of the block."""
    if spec.is_main or not spec.enabled:
        yield
        return
    sys.stdout.flush()
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


if _BACKEND == "torch":
    import torch
    import torch.distributed as dist

    class DistributedContextManager:
        """torch DDP process-group lifecycle -- the outermost scope of a run.

        On enter it resolves topology from the ``torchrun`` environment, binds the
        local GPU, and initializes the default process group; on exit it
        synchronizes and tears the group down.

        Outside ``torchrun`` (``WORLD_SIZE`` unset or ``1``) it is a no-op yielding a
        single-process :class:`DistributedSpec`, so single-GPU runs are byte-for-byte
        unchanged. Whether training is distributed is then a single fact -- the
        existence of the process group -- which downstream code (e.g. the executor)
        reads via ``torch.distributed.is_initialized()`` rather than a threaded flag.
        """

        def __init__(self, backend: str = "nccl"):
            self.backend = backend
            self._initialized = False

        def __enter__(self) -> DistributedSpec:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if world_size <= 1 or not dist.is_available():
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                return DistributedSpec(False, 0, 0, 1, device)

            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device, backend = f"cuda:{local_rank}", self.backend
            else:
                device, backend = "cpu", "gloo"
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
            self._initialized = True
            return DistributedSpec(True, rank, local_rank, world_size, device)

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._initialized:
                if exc_type is None:
                    dist.barrier()
                dist.destroy_process_group()
                self._initialized = False
            return False

elif _BACKEND == "jax":

    class DistributedContextManager:
        """jax ``distributed`` lifecycle + GSPMD device mesh.

        Mirrors the torch contract: a context manager yielding a
        :class:`DistributedSpec`. The planned implementation initializes the jax
        distributed runtime (auto-detected under the launcher), builds a ``Mesh``
        over ``jax.devices()`` along :paramref:`axis_name`, and returns a spec with
        ``rank = jax.process_index()``, ``world_size = jax.process_count()`` and
        ``mesh`` populated; the executor then auto-detects sharding from the live
        mesh, just as the torch executor auto-detects the live process group.
        """

        def __init__(self, axis_name: str = "data"):
            self.axis_name = axis_name

        def __enter__(self) -> DistributedSpec:
            raise NotImplementedError(
                "jax distributed backend is not implemented yet; the GSPMD seam is "
                "DistributedSpec.mesh / ExecutionContextManager.mesh."
            )

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

else:
    raise ValueError(f"Unknown THUNDER_BACKEND: {_BACKEND}. Supported: torch, jax.")
