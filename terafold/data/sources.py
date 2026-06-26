"""Curated registry of public cloth / folding data sources.

This is *knowledge*, not data: a small, hand-maintained table describing public
datasets relevant to TeraFold, how they relate to our robot, and what they are
actually good for. It is pure-python (no network, no heavy deps) so it always
works — even fully offline — and drives ``terafold data-sources`` plus the
offline portion of ``terafold inspect-hf-dataset``.

Guiding principle (repeated everywhere): data collected on OTHER embodiments
(SO-101, Aloha, Unitree H1, humans) is for **perception / scoring / visual
pretraining / reference**, NOT for directly driving our LeArm. Only an
action/state space that matches our robot is a policy candidate, and even then
frames/units must be verified before any execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "DataSource",
    "DATA_SOURCES",
    "list_data_sources",
    "get_data_source",
    "format_data_sources",
    "POLICY_COMPAT_VALUES",
    "RECOMMENDED_USE_VOCAB",
]

# Allowed values for the "direct policy compatibility with our robot" field.
POLICY_COMPAT_VALUES = ("yes", "no", "maybe")

# Controlled vocabulary for recommended uses.
RECOMMENDED_USE_VOCAB = (
    "perception",
    "fold_success_scoring",
    "visual_pretraining",
    "ACT_policy_testing",
    "bimanual_reference",
    "segmentation",
    "keypoint_pretraining",
    "fold_stage_recognition",
    "not_direct_rollout",
)


@dataclass(frozen=True)
class DataSource:
    """One public dataset entry."""

    repo_id: str
    name: str
    modality: Tuple[str, ...]  # e.g. image, video, robot_actions, human_demos, depth, skeleton
    embodiment: str  # SO-101 | Aloha (bimanual) | Unitree H1 | human | image-only | synthetic | unknown
    policy_compatible: str  # one of POLICY_COMPAT_VALUES — w.r.t. OUR robot (LeArm now)
    recommended_use: Tuple[str, ...]
    est_size: str = "unknown"
    fmt: str = "unknown"  # lerobot | hdf5+mp4 | coco | images | synthetic | unknown
    large: bool = False  # if True, full download is gated behind --allow-large-download
    streamable: bool = True  # whether HF streaming (no full download) is expected to work
    verified_repo_id: bool = True  # is the HF repo id confirmed exact?
    notes: str = ""

    def __post_init__(self) -> None:
        if self.policy_compatible not in POLICY_COMPAT_VALUES:
            raise ValueError(
                f"{self.repo_id}: policy_compatible must be one of {POLICY_COMPAT_VALUES}"
            )

    def to_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "name": self.name,
            "modality": list(self.modality),
            "embodiment": self.embodiment,
            "policy_compatible": self.policy_compatible,
            "recommended_use": list(self.recommended_use),
            "est_size": self.est_size,
            "format": self.fmt,
            "large": self.large,
            "streamable": self.streamable,
            "verified_repo_id": self.verified_repo_id,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# The registry. Keep entries honest: mark verified_repo_id=False when the exact
# Hugging Face id is not confirmed, and never claim direct LeArm compatibility.
# --------------------------------------------------------------------------

DATA_SOURCES: List[DataSource] = [
    DataSource(
        repo_id="observabot/so101_cloth_folding1",
        name="SO-101 cloth folding (LeRobot)",
        modality=("image", "robot_actions"),
        embodiment="SO-101",
        policy_compatible="maybe",
        recommended_use=(
            "ACT_policy_testing",
            "perception",
            "fold_success_scoring",
            "visual_pretraining",
        ),
        est_size="small–medium (LeRobot episodes)",
        fmt="lerobot",
        large=False,
        streamable=True,
        notes=(
            "Most useful public POLICY dataset if you adopt an SO-101-like "
            "action/state space. MAYBE policy-compatible for SO-101; NOT directly "
            "compatible with the LeArm (different DoF / encoder / kinematics) — "
            "re-map and validate before any execution."
        ),
    ),
    DataSource(
        repo_id="lerobot/xvla-soft-fold",
        name="XVLA soft-fold (LeRobot)",
        modality=("video", "robot_actions"),
        embodiment="unknown (LeRobot)",
        policy_compatible="maybe",
        recommended_use=(
            "visual_pretraining",
            "fold_success_scoring",
            "ACT_policy_testing",
            "not_direct_rollout",
        ),
        est_size="large (inspect/stream metadata first)",
        fmt="lerobot",
        large=True,
        streamable=True,
        notes=(
            "Large LeRobot-format cloth-folding dataset. INSPECT/STREAM metadata "
            "first; NEVER full-download by default. Compatibility depends on the "
            "underlying embodiment — verify the action space before policy use."
        ),
    ),
    DataSource(
        repo_id="Facebear/XVLA-Soft-Fold",
        name="XVLA Soft-Fold (Aloha bimanual)",
        modality=("video", "robot_actions"),
        embodiment="Aloha (bimanual)",
        policy_compatible="no",
        recommended_use=(
            "visual_pretraining",
            "bimanual_reference",
            "fold_success_scoring",
            "perception",
            "not_direct_rollout",
        ),
        est_size="large (HDF5 + MP4)",
        fmt="hdf5+mp4",
        large=True,
        streamable=False,
        notes=(
            "Aloha bimanual cloth folding (HDF5/MP4). Use for visual "
            "representation learning, BIMANUAL reference (TeraFold V4), and "
            "success/perception learning. NOT directly compatible with the "
            "single-arm LeArm. Streaming may be limited (raw HDF5/MP4)."
        ),
    ),
    DataSource(
        repo_id="AIRO/folding-demonstrations",
        name="AIRO/Ghent folding demonstrations (human)",
        modality=("video", "image", "depth", "skeleton", "human_demos"),
        embodiment="human",
        policy_compatible="no",
        recommended_use=(
            "fold_stage_recognition",
            "perception",
            "visual_pretraining",
            "not_direct_rollout",
        ),
        est_size="unknown",
        fmt="images",
        large=True,
        streamable=False,
        verified_repo_id=False,
        notes=(
            "Human folding demonstrations (RGB / depth / skeleton / subtask labels "
            "where available) from KU Leuven/Ghent AIRO. Use for fold-STAGE "
            "recognition and perception — NOT robot action rollout (human "
            "embodiment). NOTE: exact HF repo id is unverified; it may live under "
            "the AIRO 'airo-mono' / cloth-competition releases rather than a single "
            "HF dataset — confirm before fetching."
        ),
    ),
    DataSource(
        repo_id="tlpss/aRTF-Clothes-dataset",
        name="aRTF real clothes (images)",
        modality=("image",),
        embodiment="image-only",
        policy_compatible="no",
        recommended_use=(
            "perception",
            "segmentation",
            "keypoint_pretraining",
            "fold_success_scoring",
            "not_direct_rollout",
        ),
        est_size="medium (image dataset)",
        fmt="coco",
        large=False,
        streamable=True,
        notes=(
            "Real clothing images with annotations. Load via the COCO/image "
            "importer for cloth detection + segmentation + keypoint pretraining. "
            "Image-only: no robot actions."
        ),
    ),
    DataSource(
        repo_id="tlpss/synthetic-cloth-data",
        name="Synthetic cloth (keypoints/segmentation)",
        modality=("image", "synthetic"),
        embodiment="synthetic",
        policy_compatible="no",
        recommended_use=(
            "keypoint_pretraining",
            "segmentation",
            "perception",
            "not_direct_rollout",
        ),
        est_size="medium–large (synthetic renders)",
        fmt="coco",
        large=True,
        streamable=True,
        notes=(
            "Synthetic keypoint/segmentation reference renders. Useful to augment "
            "TeraFold's own synthetic generator for perception pretraining. "
            "Image-only / synthetic; no robot actions."
        ),
    ),
    # --- Additional well-known references (founder previously inspected these) ---
    DataSource(
        repo_id="lerobot/unitreeh1_fold_clothes",
        name="Unitree H1 fold clothes (LeRobot)",
        modality=("video", "robot_actions"),
        embodiment="Unitree H1",
        policy_compatible="no",
        recommended_use=(
            "visual_pretraining",
            "bimanual_reference",
            "fold_success_scoring",
            "not_direct_rollout",
        ),
        est_size="large",
        fmt="lerobot",
        large=True,
        streamable=True,
        notes=(
            "Humanoid (Unitree H1) cloth folding. Very different embodiment — use "
            "for visual pretraining and bimanual reference only, never rollout."
        ),
    ),
    DataSource(
        repo_id="xenorobotics/towel-fold-trimmed-v21",
        name="Xeno towel fold (trimmed)",
        modality=("image", "robot_actions"),
        embodiment="unknown (LeRobot)",
        policy_compatible="maybe",
        recommended_use=(
            "perception",
            "fold_success_scoring",
            "visual_pretraining",
            "ACT_policy_testing",
        ),
        est_size="small–medium",
        fmt="lerobot",
        large=False,
        streamable=True,
        notes=(
            "Towel-folding LeRobot dataset. Inspect the action space; treat as a "
            "policy candidate ONLY if the state/action space matches our arm."
        ),
    ),
]

_BY_ID: Dict[str, DataSource] = {s.repo_id.lower(): s for s in DATA_SOURCES}


def list_data_sources() -> List[DataSource]:
    """Return all registered data sources."""
    return list(DATA_SOURCES)


def get_data_source(repo_id: str) -> Optional[DataSource]:
    """Look up a source by repo id (case-insensitive). ``None`` if unknown."""
    return _BY_ID.get(str(repo_id).lower())


def format_data_sources(sources: Optional[List[DataSource]] = None) -> str:
    """Render the registry as a readable, block-per-source string."""
    sources = sources if sources is not None else DATA_SOURCES
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("TeraFold — known public cloth / folding data sources")
    lines.append(
        "Reminder: foreign-embodiment data is for perception / scoring / "
        "pretraining / reference,"
    )
    lines.append("          NOT for directly driving our LeArm. Verify before any rollout.")
    lines.append("=" * 78)
    for s in sources:
        compat = {
            "yes": "yes",
            "no": "no  (foreign embodiment — not for rollout)",
            "maybe": "maybe (verify action/state space first)",
        }[s.policy_compatible]
        flags = []
        if s.large:
            flags.append("LARGE→needs --allow-large-download for full pull")
        flags.append("streamable" if s.streamable else "not streamable")
        if not s.verified_repo_id:
            flags.append("repo-id UNVERIFIED")
        lines.append("")
        lines.append(f"● {s.repo_id}")
        lines.append(f"    name        : {s.name}")
        lines.append(f"    modality    : {', '.join(s.modality)}")
        lines.append(f"    embodiment  : {s.embodiment}")
        lines.append(f"    est. size   : {s.est_size}")
        lines.append(f"    format      : {s.fmt}")
        lines.append(f"    policy-compat (our robot): {compat}")
        lines.append(f"    recommended : {', '.join(s.recommended_use)}")
        lines.append(f"    flags       : {'; '.join(flags)}")
        if s.notes:
            lines.append(f"    notes       : {s.notes}")
    lines.append("")
    lines.append("-" * 78)
    lines.append(
        "Use:  terafold inspect-hf-dataset --repo-id <id> --streaming   (metadata, no download)"
    )
    lines.append(
        "      terafold sample-hf-dataset  --repo-id <id> --max-samples 500 --out data/public_samples/<name>"
    )
    lines.append(
        "      terafold cache-hf-subset    --repo-id <id> --max-episodes 20 --out data/cache/<name>"
    )
    lines.append(
        "      terafold import-hf-lerobot  --repo-id <id> --max-episodes 20 --out data/public/<name>"
    )
    return "\n".join(lines)
