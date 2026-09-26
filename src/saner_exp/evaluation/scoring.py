"""File-ranking metrics matching the frozen LinuxFLBench implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence


class SubmissionFormatError(ValueError):
    """A submitted ranking is outside the experiment output contract."""


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    reciprocal_rank: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    first_relevant_rank: int | None
    submitted_count: int
    relevant_count: int


def _canonical_prediction(path: str) -> str:
    if type(path) is not str or not path:
        raise SubmissionFormatError("every submitted path must be a nonempty string")
    value = path.removeprefix("./")
    parsed = PurePosixPath(value)
    if not value or value.startswith("/") or "\\" in value or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        raise SubmissionFormatError("submitted paths must be canonical relative POSIX paths")
    return parsed.as_posix()


def score_submission(
    correct_paths: Sequence[str], submitted_paths: Sequence[str] | None,
) -> EpisodeScore:
    """Score one ranked submission; a missing legal submission receives zeros."""
    if type(correct_paths) not in (list, tuple) or not correct_paths:
        raise ValueError("correct_paths must be a nonempty sequence")
    if any(type(path) is not str or not path for path in correct_paths):
        raise ValueError("correct_paths must contain nonempty strings")
    correct = set(correct_paths)
    if submitted_paths is None:
        predictions: list[str] = []
    else:
        if type(submitted_paths) not in (list, tuple) or len(submitted_paths) > 10:
            raise SubmissionFormatError("submitted ranking must contain at most ten paths")
        predictions = []
        seen = set()
        for raw_path in submitted_paths:
            path = _canonical_prediction(raw_path)
            if path not in seen:
                seen.add(path)
                predictions.append(path)

    first_rank = next(
        (rank for rank, path in enumerate(predictions, start=1) if path in correct), None,
    )

    def recall_at(k: int) -> float:
        return len(correct.intersection(predictions[:k])) / len(correct)

    return EpisodeScore(
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        recall_at_1=recall_at(1),
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        first_relevant_rank=first_rank,
        submitted_count=len(predictions),
        relevant_count=len(correct),
    )
