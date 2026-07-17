from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any, Dict, Iterable, Tuple, TYPE_CHECKING

from .context import Ref

if TYPE_CHECKING:
    from .operation import Operation


RefSpec = Any


def normalize_refs(refs: Iterable[RefSpec] | None) -> frozenset[Ref]:
    """ """
    if refs is None:
        return frozenset()
    return frozenset(ref if isinstance(ref, Ref) else Ref(ref) for ref in refs)


class PipelineValidationError(ValueError):
    pass


class PipelineRuntimeError(RuntimeError):
    pass


@dataclass(slots=True)
class _RefNode:
    terminal: bool = False
    children: Dict[Any, "_RefNode"] = field(default_factory=dict)


class RefIndex:
    __slots__ = ("_root",)

    def __init__(self, refs: Iterable[RefSpec] = ()):
        self._root = _RefNode()
        self.update(refs)

    def add(self, ref: RefSpec) -> None:
        ref = ref if isinstance(ref, Ref) else Ref(ref)
        node = self._root
        for step in ref.path:
            node = node.children.setdefault(step, _RefNode())
        node.terminal = True

    def update(self, refs: Iterable[RefSpec]) -> None:
        for ref in refs:
            self.add(ref)

    def covers(self, ref: RefSpec) -> bool:
        ref = ref if isinstance(ref, Ref) else Ref(ref)
        node = self._root
        if node.terminal:
            return True
        for step in ref.path:
            node = node.children.get(step)
            if node is None:
                return False
            if node.terminal:
                return True
        return False


class PipelineValidator:
    """Validate the local data flow between operations.

    Analysis reports external inputs. Validation is stricter: a requirement must
    be available at the point where the operation is reached.
    """

    def __init__(self, pipeline: Iterable[Operation | Tuple[frozenset[Ref], frozenset[Ref]]], name: str = ""):
        self.pipeline = tuple(pipeline)
        self.name = name

    def analyze(
        self, initial_refs: Iterable[RefSpec] = ()
    ) -> Tuple[frozenset[Ref], frozenset[Ref]]:
        available = RefIndex(initial_refs)
        external_requires: list[Ref] = []
        provided_refs: list[Ref] = []

        for op in self.pipeline:
            if isinstance(op, tuple):
                sub_requires, sub_provides = op
            else:
                sub_requires = op.requires
                sub_provides = op.provides
            for ref in sub_requires:
                if available.covers(ref):
                    continue
                external_requires.append(ref)
                available.add(ref)
            available.update(sub_provides)
            provided_refs.extend(sub_provides)

        return frozenset(external_requires), frozenset(provided_refs)

    def validate(self, initial_refs: Iterable[RefSpec] = (), mode: str = "error"):
        issue = self._first_issue(initial_refs)
        if issue is None:
            return

        message = self._format_issue(*issue)
        if mode == "warn":
            print(f"[Thunder][Pipeline Warning] {message}")
            return
        raise PipelineValidationError(message)

    def _first_issue(
        self, initial_refs: Iterable[RefSpec]
    ) -> tuple[int, Operation, tuple[Ref, ...]] | None:
        available = RefIndex(initial_refs)

        for idx, op in enumerate(self.pipeline):
            if isinstance(op, tuple):
                sub_requires, sub_provides = op
            else:
                sub_requires = op.requires
                sub_provides = op.provides
            missing = tuple(ref for ref in sub_requires if not available.covers(ref))
            if missing:
                return idx, op, missing
            available.update(sub_provides)

        return None

    def _format_issue(self, idx: int, op, missing: tuple[Ref, ...]) -> str:
        name = self.name or "<unnamed>"
        missing_refs = ", ".join(map(repr, missing))
        op_label = f"sub-pipeline" if isinstance(op, tuple) else f"{type(op).__name__}('{op.name}')"
        return (
            f"Pipeline '{name}' validation failed at op[{idx}] "
            f"{op_label}. Missing requirements: {missing_refs}"
        )
