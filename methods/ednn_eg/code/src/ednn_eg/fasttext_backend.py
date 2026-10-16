"""Supervised fastText representation of coding sequences for eDNN-EG."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .contracts import FORMAL_FASTTEXT_SPEC, FastTextSpec, normalize_cds
from .errors import ContractViolation, MissingRequiredDependency


class EmbeddingBackend(Protocol):
    metadata: dict[str, object]

    def fit(self, ids: Sequence[str], sequences: Sequence[str], labels: np.ndarray) -> None: ...

    def transform(self, ids: Sequence[str], sequences: Sequence[str]) -> np.ndarray: ...


class OfficialFastTextBackend:

    def __init__(
        self,
        *,
        seed: int,
        work_dir: str | Path,
        thread: int = 1,
        spec: FastTextSpec = FORMAL_FASTTEXT_SPEC,
    ) -> None:
        spec.assert_formal()
        if thread != 1:
            raise ContractViolation(
                "formal fastText training requires thread=1 for reproducible updates"
            )
        self.seed = int(seed)
        self.work_dir = Path(work_dir)
        self.thread = int(thread)
        self.spec = spec
        self.model = None
        self.metadata: dict[str, object] = {}

    @staticmethod
    def _import_fasttext():
        try:
            import fasttext
        except Exception as exc:
            raise MissingRequiredDependency(
                "fasttext==0.9.3 is required; no embedding fallback is permitted"
            ) from exc
        return fasttext

    def fit(self, ids: Sequence[str], sequences: Sequence[str], labels: np.ndarray) -> None:
        if len(ids) != len(sequences) or len(ids) != len(labels):
            raise ContractViolation("fastText training IDs, sequences and labels have different lengths")
        if len(set(ids)) != len(ids):
            raise ContractViolation("duplicate training identifiers passed to fastText")
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if set(np.unique(y).tolist()) != {0, 1}:
            raise ContractViolation("every fastText training fold must contain both classes")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.work_dir / "supervised_train.txt"
        digest = hashlib.sha256()
        with train_path.open("w", encoding="utf-8", newline="\n") as handle:
            for sequence, label in zip(sequences, y, strict=True):
                normalized = normalize_cds(sequence)
                class_token = "__label__pos" if int(label) == 1 else "__label__neg"
                line = f"{class_token} {' '.join(normalized)}\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))

        fasttext = self._import_fasttext()
        kwargs = self.spec.as_fasttext_kwargs(seed=self.seed, thread=self.thread)
        try:
            self.model = fasttext.train_supervised(input=str(train_path), **kwargs)
        except Exception as exc:
            raise MissingRequiredDependency(
                "official supervised fastText training failed; no fallback is permitted"
            ) from exc

        labels_seen = set(self.model.get_labels())
        if labels_seen != {"__label__neg", "__label__pos"}:
            raise ContractViolation(f"unexpected fastText label set: {sorted(labels_seen)}")
        self.metadata = {
            "backend": "fasttext",
            "supervised": True,
            "thread": self.thread,
            "train_rows": len(ids),
            "parameters": {
                name: value for name, value in kwargs.items() if name != "seed"
            },
            "labels": sorted(labels_seen),
        }

    def save_model(self, path: str | Path) -> None:

        if self.model is None:
            raise ContractViolation("fastText save called before fit")
        self.model.save_model(str(path))

    def load_model(self, path: str | Path) -> None:

        source = Path(path)
        if not source.is_file():
            raise ContractViolation(f"fastText model is absent: {source}")
        fasttext = self._import_fasttext()
        self.model = fasttext.load_model(str(source))
        labels_seen = set(self.model.get_labels())
        if labels_seen != {"__label__neg", "__label__pos"}:
            raise ContractViolation(
                f"stored fastText model has label set {sorted(labels_seen)}"
            )

    def transform(self, ids: Sequence[str], sequences: Sequence[str]) -> np.ndarray:
        if self.model is None:
            raise ContractViolation("fastText transform called before fit")
        if len(ids) != len(sequences):
            raise ContractViolation("fastText transform IDs and sequences have different lengths")
        if not ids:
            raise ContractViolation("fastText transform received no sequences")
        vectors = []
        for sequence in sequences:
            normalized = normalize_cds(sequence)
            vector = np.asarray(self.model.get_sentence_vector(" ".join(normalized)), dtype=np.float32)
            if vector.shape != (self.spec.dim,):
                raise ContractViolation(
                    f"fastText returned shape {vector.shape}; expected ({self.spec.dim},)"
                )
            if not np.isfinite(vector).all():
                raise ContractViolation("fastText returned non-finite values")
            vectors.append(vector)
        return np.vstack(vectors)
