"""Real-image-first towel perception/planning dataset pipeline.

This package builds the *towel-only* learning layer on top of the safety-gated
robot platform. The default data path is **real images** (phone / webcam /
datasets); synthetic generation is optional augmentation/pretraining only.

Scope (intentionally narrow):

* generic white/off-white hotel towels, one at a time;
* states: flat / wrinkled / partially folded / folded, plus reject buckets;
* perception + planning only — NO action policy, NO ACT/Diffusion, NO contact
  unlock, NO change to the hardware safety gates, NO physical robot required.

Sub-modules:

* :mod:`terafold.towel.schema`   — label schema, states, image hashing/sizing.
* :mod:`terafold.towel.dataset`  — create a real-image dataset + manifest.
* :mod:`terafold.towel.overlay`  — draw corner/bbox/fold-axis previews.
* :mod:`terafold.towel.labeling` — local 4-corner labeling helpers.
* :mod:`terafold.towel.web_import` — legal/controlled external importers.
* :mod:`terafold.towel.synthetic` — OPTIONAL synthetic augmentation.
* :mod:`terafold.towel.merge`    — merge datasets (dedup + train/val/test split).
* :mod:`terafold.towel.yolo_export` — YOLO pose dataset export (4 corner kpts).
* :mod:`terafold.towel.yolo_runtime` — train command / inference / evaluation.
* :mod:`terafold.towel.plan_critic` — rule-based fold-plan critic (+scaffold).

External-image + Claude pseudo-labeling pipeline:

* :mod:`terafold.towel.raw_ingest` — Kaggle / Open Images RAW image ingest + manifest.
* :mod:`terafold.towel.filtering`  — keyword/VLM candidate filtering.
* :mod:`terafold.towel.claude_pseudolabel` — Claude Vision pseudo-labeling (geometry only).
* :mod:`terafold.towel.openai_pseudolabel` — OpenAI Vision pseudo-labeling (second opinion).
* :mod:`terafold.towel.openai_multipass` — multi-stage high-precision OpenAI labeling.
* :mod:`terafold.towel.crop_candidates` — crop-first localization (box the target towel, then crop).
* :mod:`terafold.towel.review`     — human review + high-confidence auto-approval.

OpenAI-generated "realism bridge" layer (Layer B):

* :mod:`terafold.towel.prompt_bank` — towel prompt curriculum (categories + buckets).
* :mod:`terafold.towel.costing`     — approximate image-generation cost + budget caps.
* :mod:`terafold.towel.generated_dataset` — generated dataset manifest + metadata.
* :mod:`terafold.towel.openai_generation` — cost-aware OpenAI image generation.
* :mod:`terafold.towel.qc`          — generated-image quality control / dedup.
* :mod:`terafold.towel.training_plan` — recommended staged training strategy.
* :mod:`terafold.towel.critic_dataset` — scene-usability critic data from review.
"""

from __future__ import annotations

__all__ = [
    "schema", "dataset", "overlay", "merge", "yolo_export",
    "labeling", "web_import", "synthetic", "yolo_runtime", "plan_critic",
    "raw_ingest", "filtering", "claude_pseudolabel", "openai_pseudolabel",
    "openai_multipass", "crop_candidates", "review",
    "prompt_bank", "costing", "generated_dataset", "openai_generation", "qc",
    "training_plan", "critic_dataset",
]
