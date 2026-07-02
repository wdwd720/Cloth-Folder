"""Prompt curriculum for OpenAI-generated hotel-towel images (Layer B).

The curriculum is organized into two explicit **training tracks**:

* ``pose_positive`` — single, flat/mildly-wrinkled, fully-visible towels with all
  four outer corners showing. These are the ONLY images useful for the YOLO
  4-corner pose model. Their prompts carry a hard constraint block.
* ``critic_negative`` — folded stacks, rolled, multiple, hanging, occluded, or
  not-a-towel scenes. Useful later for a rejection/scene critic; NOT pose data.

A **task** is a named mix of category proportions (e.g. ``pose_positive_flat_only``
generates ONLY pose positives). Each prompt record cleanly separates
``category_target``, ``target_state``, ``usable_hint``, and ``training_track``.

Nothing here is ground truth — these fields are the *intent* of a prompt; the
actual label always comes later from Claude + human review / auto-triage.

Stdlib only (uses :mod:`random` with an explicit seed for reproducibility).
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

__all__ = [
    "CATEGORIES", "BUCKETS", "TASKS", "TASK_BUCKETS", "DEFAULT_TASK", "QUALITY_SUFFIX",
    "POSE_POSITIVE_SUFFIX", "TRAINING_TRACKS", "list_tasks", "list_categories",
    "task_proportions", "allocate_counts", "build_prompt_curriculum",
    "category_target_state", "category_training_track", "task_training_track",
]

DEFAULT_TASK = "mix_flat_wrinkled_negatives"
TRAINING_TRACKS = ("pose_positive", "critic_negative")

# Appended to NON-strict prompts: realism + "no junk" guidance.
QUALITY_SUFFIX = (
    "Photorealistic, high detail, realistic fabric folds and soft shadows, "
    "natural colour. No text, no watermark, no logo, no brand, no people, "
    "no hands. Single coherent scene."
)

# Hard constraint block for STRICT pose-positive prompts. The earlier block still
# allowed some "compact folded hotel towel" outputs, so this version is much more
# explicit that we want a SINGLE-LAYER towel FULLY SPREAD OPEN as a flat sheet —
# the way a towel is placed *before* a robot folds it, not after housekeeping.
POSE_POSITIVE_SUFFIX = (
    "Generate one single-layer towel fully spread open like a flat rectangular "
    "sheet. The towel should be unfolded to its full size, not folded into a "
    "smaller rectangle. Do not show a folded towel stack, layered towel edges, "
    "rolled towel, compact hotel towel, towel pile, or decorative folded towel. "
    "The towel should lie flat on the surface with a thin visible edge, all four "
    "outer corners visible, and the full perimeter visible. Make it look like a "
    "towel placed open for a robot to fold, not like a towel already folded by "
    "housekeeping. The whole towel must be fully visible inside the image frame and "
    "occupy roughly 55-80% of the frame (not a tight close-up). All four outer "
    "corners must be separated and clearly visible, with no thick stacked edges and "
    "no multiple layers. The towel must not be folded, not rolled, not stacked, not "
    "hanging, not a towel pile, and not multiple towels. Photorealistic, natural "
    "colour. No text, no watermark, no logo, no people, no hands, no severe occlusion."
)

# Randomizable detail slots (global defaults; overridable per-category and per-task).
BUCKETS: Dict[str, List[str]] = {
    "viewpoint": [
        "viewed directly top-down", "from a high angle (~60° from horizontal)",
        "from a shallow 45° angle", "from a slight oblique angle",
        "from near eye-level",
    ],
    "environment": [
        "in a clean hotel bathroom", "in a luxury hotel room", "in a spa",
        "on a hotel housekeeping cart", "on a bathroom vanity",
        "on a freshly made hotel bed", "in a bright laundry area",
    ],
    "lighting": [
        "soft natural daylight", "warm tungsten lighting",
        "bright even studio lighting", "soft diffused morning light",
        "cool overcast light", "dramatic side lighting with soft shadows",
    ],
    "surface": [
        "a polished marble countertop", "a light wooden table",
        "a crisp white bed", "a stainless steel tray", "a glass shelf",
        "a beige tiled floor", "a folded-down white duvet",
    ],
    "texture": [
        "plush thick terry cloth", "thin flat-weave cotton", "waffle-weave cotton",
        "ribbed cotton", "soft velour",
    ],
    "towel_color": ["white", "off-white", "cream", "ivory", "pale grey"],
    "clutter": [
        "with no other objects around it",
        "with a few minimal toiletries nearby",
        "with a small potted plant in the soft-focus background",
        "with a neatly arranged tray of amenities beside it",
        "with light household clutter at the frame edges",
    ],
}

# Re-usable viewpoint subsets for the strict pose categories.
_TOPDOWN = ["viewed directly top-down", "photographed from straight overhead"]
_NEAR_TOPDOWN = ["from a near-top-down angle just off vertical",
                 "from a slightly high angle close to overhead"]
_OBLIQUE = ["from a shallow 45° angle", "from a high angle (~60° from horizontal)",
            "from a slight oblique angle"]
# Clean, uncluttered surroundings for pose positives (no "neatly arranged" framing
# that can nudge the model toward a folded-towel hotel presentation).
_CLEAN_CLUTTER = ["on an otherwise empty surface", "with a plain uncluttered background",
                  "with minimal tidy surroundings"]

# ----------------------------------------------------------------------
# Categories. ``target_state`` / ``usable_hint`` / ``training_track`` are INTENT
# metadata (never trusted). ``strict_pose`` ⇒ use POSE_POSITIVE_SUFFIX.
# ``bucket_overrides`` constrain specific slots for that category.
# ----------------------------------------------------------------------

CATEGORIES: List[Dict[str, Any]] = [
    # --- existing broad positives (kept for the legacy mix task) ---
    {
        "key": "positive_flat", "letter": "A", "target_state": "flat_unfolded",
        "usable_hint": True, "training_track": "pose_positive", "strict_pose": False,
        "templates": [
            "A single {towel_color} hotel towel made of {texture}, lying completely "
            "flat and fully unfolded on {surface} {environment}, {viewpoint}, lit by "
            "{lighting}, {clutter}. The whole towel and all four corners are clearly "
            "visible and uncovered.",
            "One {towel_color} bath towel ({texture}) spread out flat and smooth on "
            "{surface}, {viewpoint}, {lighting}, {clutter}. Entire rectangular towel "
            "in frame with crisp visible edges.",
        ],
    },
    {
        "key": "positive_wrinkled", "letter": "B", "target_state": "wrinkled_unfolded",
        "usable_hint": True, "training_track": "pose_positive", "strict_pose": False,
        "templates": [
            "A single {towel_color} hotel towel of {texture}, unfolded but mildly "
            "wrinkled and slightly rumpled, on {surface} {environment}, {viewpoint}, "
            "{lighting}, {clutter}. Still mostly flat and fully visible, all corners "
            "in frame.",
            "One {towel_color} towel ({texture}) laid out with gentle wrinkles and a "
            "few soft folds, on {surface}, {viewpoint}, {lighting}, {clutter}. The "
            "towel remains foldable and its outline is clear.",
        ],
    },
    {
        "key": "positive_challenging", "letter": "C", "target_state": "flat_unfolded",
        "usable_hint": True, "training_track": "pose_positive", "strict_pose": False,
        "templates": [
            "A single {towel_color} hotel towel ({texture}) lying flat on {surface} "
            "{environment}, {viewpoint}, under {lighting} casting noticeable shadows, "
            "{clutter}. Harder lighting and perspective but the towel is still fully "
            "visible and usable for fold planning.",
            "One {towel_color} towel of {texture} on {surface}, {viewpoint}, with "
            "{lighting}, mild clutter and strong shadows, {clutter}. Challenging but "
            "the entire towel and its corners can still be seen.",
        ],
    },
    # --- STRICT pose-positive categories (single-layer fully-spread-open sheets) ---
    {
        "key": "pose_flat_topdown", "letter": "P1", "target_state": "flat_unfolded",
        "usable_hint": True, "training_track": "pose_positive", "strict_pose": True,
        "bucket_overrides": {"viewpoint": _TOPDOWN, "clutter": _CLEAN_CLUTTER},
        "templates": [
            "A single {towel_color} bath towel of {texture}, fully spread open as a "
            "flat single-layer rectangular sheet at its full size on {surface} "
            "{environment}, {viewpoint}, lit by {lighting}, {clutter}.",
        ],
    },
    {
        "key": "pose_flat_near_topdown", "letter": "P2", "target_state": "flat_unfolded",
        "usable_hint": True, "training_track": "pose_positive", "strict_pose": True,
        "bucket_overrides": {"viewpoint": _NEAR_TOPDOWN, "clutter": _CLEAN_CLUTTER},
        "templates": [
            "A single {towel_color} bath towel ({texture}) lying fully spread open and "
            "flat, a single-layer full-size rectangle on {surface} {environment}, "
            "{viewpoint}, {lighting}, {clutter}.",
        ],
    },
    {
        "key": "pose_flat_oblique", "letter": "P3", "target_state": "flat_unfolded",
        "usable_hint": True, "training_track": "pose_positive", "strict_pose": True,
        "bucket_overrides": {"viewpoint": _OBLIQUE, "clutter": _CLEAN_CLUTTER},
        "templates": [
            "A single {towel_color} bath towel ({texture}) fully spread open and flat "
            "as a single-layer rectangular sheet at full size on {surface} "
            "{environment}, {viewpoint}, {lighting}, {clutter}.",
        ],
    },
    {
        "key": "pose_wrinkled", "letter": "P4", "target_state": "wrinkled_unfolded",
        "usable_hint": True, "training_track": "pose_positive", "strict_pose": True,
        "bucket_overrides": {"viewpoint": _TOPDOWN + _NEAR_TOPDOWN + _OBLIQUE,
                             "clutter": _CLEAN_CLUTTER},
        "templates": [
            "A single {towel_color} bath towel ({texture}) fully spread open and lying "
            "mostly flat with only mild wrinkles, a single-layer full-size rectangle "
            "on {surface} {environment}, {viewpoint}, {lighting}, {clutter}.",
        ],
    },
    # --- critic negatives (NOT pose data) ---
    {
        "key": "negative_multiple_towels", "letter": "D", "target_state": "multiple_towels",
        "usable_hint": False, "training_track": "critic_negative", "strict_pose": False,
        "templates": [
            "Several {towel_color} hotel towels of {texture} piled and overlapping on "
            "{surface} {environment}, {viewpoint}, {lighting}, {clutter}. Multiple "
            "towels are visible at once — not a single isolated towel.",
            "A stack of three or more {towel_color} towels ({texture}) jumbled "
            "together on {surface}, {viewpoint}, {lighting}. Many towels overlapping.",
        ],
    },
    {
        "key": "negative_bad_view", "letter": "E", "target_state": "bad_view",
        "usable_hint": False, "training_track": "critic_negative", "strict_pose": False,
        "templates": [
            "A single {towel_color} hotel towel ({texture}) hanging from a towel rack "
            "{environment}, {viewpoint}, {lighting}, {clutter}. The towel is draped "
            "vertically and partly out of frame — not lying flat, not foldable.",
            "A {towel_color} towel of {texture} mostly off-frame and severely "
            "occluded behind objects on {surface}, {viewpoint}, {lighting}. Only part "
            "of the towel is visible; useless for fold planning.",
            "A {towel_color} towel ({texture}) bunched, crumpled and draped over the "
            "edge of {surface} {environment}, {viewpoint}, {lighting}, {clutter}. "
            "Heavily distorted, corners not visible.",
        ],
    },
    {
        "key": "negative_not_towel", "letter": "F", "target_state": "not_towel",
        "usable_hint": False, "training_track": "critic_negative", "strict_pose": False,
        "templates": [
            "A {towel_color} bed sheet (NOT a towel) spread on {surface} "
            "{environment}, {viewpoint}, {lighting}, {clutter}.",
            "A folded {towel_color} bathrobe (NOT a towel) on {surface}, {viewpoint}, "
            "{lighting}, {clutter}.",
            "A {towel_color} blanket / cloth napkin / bath mat (NOT a bath towel) on "
            "{surface} {environment}, {viewpoint}, {lighting}, {clutter}. Easy to "
            "confuse with a towel but it is not one.",
        ],
    },
    {
        "key": "maybe_folded", "letter": "G", "target_state": "partially_folded",
        "usable_hint": False, "training_track": "critic_negative", "strict_pose": False,
        "templates": [
            "A single {towel_color} hotel towel ({texture}) neatly folded into a tidy "
            "rectangle on {surface} {environment}, {viewpoint}, {lighting}, {clutter}. "
            "Already folded, not flat.",
            "A {towel_color} towel of {texture} half-folded over itself on {surface}, "
            "{viewpoint}, {lighting}, {clutter}. Partially folded, decorative "
            "presentation.",
        ],
    },
    {
        "key": "critic_folded_stack", "letter": "H", "target_state": "folded_success",
        "usable_hint": False, "training_track": "critic_negative", "strict_pose": False,
        "templates": [
            "A neat stack of several folded {towel_color} hotel towels ({texture}) on "
            "{surface} {environment}, {viewpoint}, {lighting}, {clutter}. A folded "
            "towel stack, not a single flat towel.",
        ],
    },
    {
        "key": "critic_rolled", "letter": "I", "target_state": "bad_view",
        "usable_hint": False, "training_track": "critic_negative", "strict_pose": False,
        "templates": [
            "A {towel_color} hotel towel ({texture}) rolled up tightly into a cylinder "
            "on {surface} {environment}, {viewpoint}, {lighting}, {clutter}. A rolled "
            "towel, not lying flat.",
        ],
    },
]

_CATEGORY_BY_KEY = {c["key"]: c for c in CATEGORIES}

# Named task mixes: category_key -> relative weight (need not sum to 1).
TASKS: Dict[str, Dict[str, float]] = {
    "flat_only": {"positive_flat": 1.0},
    "positives_only": {
        "positive_flat": 0.5, "positive_wrinkled": 0.3, "positive_challenging": 0.2,
    },
    "mix_flat_wrinkled_negatives": {
        "positive_flat": 0.35, "positive_wrinkled": 0.20, "positive_challenging": 0.15,
        "negative_multiple_towels": 0.10, "negative_bad_view": 0.10,
        "negative_not_towel": 0.05, "maybe_folded": 0.05,
    },
    "hard_negatives": {
        "negative_multiple_towels": 0.25, "negative_bad_view": 0.30,
        "negative_not_towel": 0.25, "maybe_folded": 0.20,
    },
    "balanced": {c["key"]: 1.0 for c in CATEGORIES},
    # --- new strict tracks (Deliverable 1) ---
    "pose_positive_flat_only": {
        "pose_flat_topdown": 0.50, "pose_flat_oblique": 0.30, "pose_wrinkled": 0.20,
    },
    "pose_positive_diverse_v1": {
        "pose_flat_topdown": 0.40, "pose_flat_oblique": 0.35, "pose_wrinkled": 0.25,
    },
    # Strictest pose task: single-layer, fully-spread-open sheets only.
    # 70% top-down, 20% near-top-down, 10% mild-wrinkle (still full-size unfolded).
    "pose_positive_unfolded_sheet_v1": {
        "pose_flat_topdown": 0.70, "pose_flat_near_topdown": 0.20, "pose_wrinkled": 0.10,
    },
    "critic_negatives_v1": {
        "critic_folded_stack": 0.20, "critic_rolled": 0.15,
        "negative_multiple_towels": 0.20, "negative_bad_view": 0.20,
        "negative_not_towel": 0.15, "maybe_folded": 0.10,
    },
}

# Optional per-task slot overrides (applied unless a category overrides the slot).
TASK_BUCKETS: Dict[str, Dict[str, List[str]]] = {
    # flat_only: keep it simple + clean (minimal clutter, plain surfaces).
    "pose_positive_flat_only": {
        "surface": ["a light wooden table", "a crisp white bed", "a bathroom counter",
                    "a clean flat tabletop"],
        "environment": ["in a clean hotel room", "on a hotel bathroom counter",
                        "in a tidy hotel setting"],
        "lighting": ["bright even lighting", "soft natural daylight"],
        "towel_color": ["white", "off-white", "cream"],
        "clutter": ["with no other objects around it", "with minimal tidy surroundings"],
    },
    # diverse_v1: the explicit diversity sets from the spec.
    "pose_positive_diverse_v1": {
        "surface": ["a crisp white bed", "a light wooden table", "a bathroom counter",
                    "a polished marble countertop", "a beige tiled floor",
                    "a housekeeping cart tray"],
        "lighting": ["bright even lighting", "soft window light", "warm bathroom light"],
        "texture": ["plush terry cloth", "waffle-weave cotton", "ribbed cotton"],
        "towel_color": ["white", "off-white", "cream"],
        "clutter": ["with no other objects around it", "with minimal tidy surroundings"],
    },
    # unfolded_sheet_v1: clean, simple surfaces so the spread-open sheet dominates.
    "pose_positive_unfolded_sheet_v1": {
        "surface": ["a large light wooden table", "a wide flat tabletop", "a clean floor",
                    "a broad bathroom counter", "a smooth desk surface"],
        "environment": ["in a clean hotel room", "in a tidy room", "in a plain setting"],
        "lighting": ["bright even lighting", "soft natural daylight"],
        "towel_color": ["white", "off-white", "cream"],
    },
}


def list_tasks() -> List[str]:
    return sorted(TASKS)


def list_categories() -> List[str]:
    return [c["key"] for c in CATEGORIES]


def category_target_state(key: str) -> str:
    return _CATEGORY_BY_KEY[key]["target_state"]


def category_training_track(key: str) -> str:
    return _CATEGORY_BY_KEY[key]["training_track"]


def task_training_track(task: str) -> str:
    """The single track of a task, or 'mixed' if it spans both."""
    tracks = {category_training_track(k) for k in TASKS[task]}
    return next(iter(tracks)) if len(tracks) == 1 else "mixed"


def task_proportions(task: str) -> Dict[str, float]:
    """Return normalized category proportions for a task (sums to 1.0)."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; known: {list_tasks()}")
    weights = TASKS[task]
    total = float(sum(weights.values())) or 1.0
    return {k: v / total for k, v in weights.items()}


def allocate_counts(task: str, num: int) -> Dict[str, int]:
    """Split ``num`` images across a task's categories (largest-remainder, exact)."""
    if num <= 0:
        return {}
    props = task_proportions(task)
    raw = {k: p * num for k, p in props.items()}
    counts = {k: int(v) for k, v in raw.items()}
    remainder = num - sum(counts.values())
    frac_order = sorted(props, key=lambda k: (raw[k] - counts[k], k), reverse=True)
    for i in range(remainder):
        counts[frac_order[i % len(frac_order)]] += 1
    return {k: c for k, c in counts.items() if c > 0}


def _effective_buckets(category: Dict[str, Any], task: str) -> Dict[str, List[str]]:
    """Per-slot options: category override > task override > global default."""
    cat_over = category.get("bucket_overrides", {})
    task_over = TASK_BUCKETS.get(task, {})
    eff = {}
    for name in BUCKETS:
        if name in cat_over:
            eff[name] = cat_over[name]
        elif name in task_over:
            eff[name] = task_over[name]
        else:
            eff[name] = BUCKETS[name]
    return eff


def _fill(template: str, rng: random.Random, eff: Dict[str, List[str]]):
    slots = {name: rng.choice(opts) for name, opts in eff.items()}
    return template.format(**slots), slots


def build_prompt_curriculum(task: str, num: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Build a reproducible list of ``num`` prompt records for ``task``.

    Each record: ``{category, category_letter, target_state, usable_hint,
    training_track, template_name, prompt, slots}``. Deterministic given
    ``(task, num, seed)``.
    """
    counts = allocate_counts(task, num)
    records: List[Dict[str, Any]] = []
    for ci, (key, n) in enumerate(sorted(counts.items())):
        cat = _CATEGORY_BY_KEY[key]
        rng = random.Random((seed << 8) ^ (ci * 1000003 + 7))
        templates = cat["templates"]
        eff = _effective_buckets(cat, task)
        suffix = POSE_POSITIVE_SUFFIX if cat.get("strict_pose") else QUALITY_SUFFIX
        for j in range(n):
            template = templates[j % len(templates)]
            prompt, slots = _fill(template, rng, eff)
            records.append({
                "category": key,
                "category_letter": cat["letter"],
                "target_state": cat["target_state"],
                "usable_hint": cat["usable_hint"],
                "training_track": cat["training_track"],
                "template_name": f"{key}#{j % len(templates)}",
                "prompt": prompt + " " + suffix,
                "slots": slots,
            })
    return records
