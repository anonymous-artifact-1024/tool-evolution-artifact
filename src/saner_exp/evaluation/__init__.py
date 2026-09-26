"""Ground-truth evaluation kept outside the agent execution environment."""

from .factorial import (
    factorial_contrast_vectors,
    joint_max_t_intervals,
    practical_effect_class,
)
from .scoring import EpisodeScore, SubmissionFormatError, score_submission
from .main_analysis import analyze_primary, contrast_bounds, grouped_joint_max_t_intervals

__all__ = [
    "EpisodeScore",
    "SubmissionFormatError",
    "factorial_contrast_vectors",
    "joint_max_t_intervals",
    "practical_effect_class",
    "score_submission",
    "analyze_primary",
    "contrast_bounds",
    "grouped_joint_max_t_intervals",
]
