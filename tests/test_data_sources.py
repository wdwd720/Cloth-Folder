"""Public data-source registry + streaming guards (no network / no hf libs)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from terafold.data.sources import (
    DATA_SOURCES,
    POLICY_COMPAT_VALUES,
    RECOMMENDED_USE_VOCAB,
    format_data_sources,
    get_data_source,
    list_data_sources,
)

REQUIRED = [
    "observabot/so101_cloth_folding1",
    "lerobot/xvla-soft-fold",
    "Facebear/XVLA-Soft-Fold",
    "AIRO/folding-demonstrations",
    "tlpss/aRTF-Clothes-dataset",
    "tlpss/synthetic-cloth-data",
]


def test_all_required_sources_present():
    ids = {s.repo_id for s in DATA_SOURCES}
    for repo in REQUIRED:
        assert repo in ids, f"missing required source {repo}"


def test_sources_have_valid_fields():
    for s in list_data_sources():
        assert s.policy_compatible in POLICY_COMPAT_VALUES
        assert s.modality, f"{s.repo_id} has no modality"
        assert s.recommended_use, f"{s.repo_id} has no recommended use"
        for use in s.recommended_use:
            assert use in RECOMMENDED_USE_VOCAB, f"{s.repo_id}: bad use {use}"


def test_lookup_case_insensitive():
    assert get_data_source("OBSERVABOT/SO101_CLOTH_FOLDING1") is not None
    assert get_data_source("does/not-exist") is None


def test_so101_is_maybe_and_warns_about_learm():
    s = get_data_source("observabot/so101_cloth_folding1")
    assert s.policy_compatible == "maybe"
    assert "LeArm" in s.notes


def test_large_sources_flagged():
    assert get_data_source("lerobot/xvla-soft-fold").large is True
    assert get_data_source("Facebear/XVLA-Soft-Fold").large is True
    assert get_data_source("Facebear/XVLA-Soft-Fold").policy_compatible == "no"


def test_airo_repo_id_unverified():
    assert get_data_source("AIRO/folding-demonstrations").verified_repo_id is False


def test_format_lists_every_source():
    text = format_data_sources()
    for repo in REQUIRED:
        assert repo in text
    assert "NOT for directly driving our LeArm" in text


# --------------------------------------------------------------------------
# Streaming guards — must never download huge data by default.
# --------------------------------------------------------------------------

from terafold.data.hf_streaming import (  # noqa: E402
    _extract_images,
    cache_hf_subset,
    import_hf_lerobot,
    requires_large_download_flag,
    sample_hf_dataset,
)


def test_large_download_flag_logic():
    big = "lerobot/xvla-soft-fold"
    # Unbounded request on a large source needs the flag...
    assert requires_large_download_flag(big, bounded=False, allow_large_download=False) is True
    # ...but a bounded request never does...
    assert requires_large_download_flag(big, bounded=True, allow_large_download=False) is False
    # ...nor does an explicitly authorized one.
    assert requires_large_download_flag(big, bounded=False, allow_large_download=True) is False
    # Unknown repo is treated as large (cautious default).
    assert requires_large_download_flag("who/knows", bounded=False, allow_large_download=False) is True


def test_unbounded_large_source_blocked_before_download(tmp_path):
    # max_episodes=0 means "all" -> unbounded -> must be gated, and must NOT
    # touch the network/datasets lib or create output.
    out = str(tmp_path / "cache")
    res = cache_hf_subset("Facebear/XVLA-Soft-Fold", max_episodes=0, out=out)
    assert res["status"] == "requires_allow_large_download"
    assert not os.path.exists(out)


def test_missing_dependency_is_graceful_and_no_download(tmp_path):
    # datasets/huggingface_hub are not installed in this env: bounded requests
    # should report missing_dependency with an install command, write nothing.
    out = str(tmp_path / "samples")
    res = sample_hf_dataset("lerobot/xvla-soft-fold", max_samples=10, out=out)
    assert res["status"] == "missing_dependency"
    assert "pip install" in res["install_command"]
    assert not os.path.exists(out)  # nothing downloaded / written


def test_bounded_request_not_gated_for_large_source(tmp_path):
    # A bounded sample from a large source must NOT be blocked by the gate; it
    # proceeds to the (missing) dependency check instead.
    res = sample_hf_dataset("lerobot/xvla-soft-fold", max_samples=5, out=str(tmp_path / "s"))
    assert res["status"] == "missing_dependency"  # gate passed, dep missing


def test_import_unbounded_gated(tmp_path):
    res = import_hf_lerobot("lerobot/xvla-soft-fold", max_episodes=0, out=str(tmp_path / "imp"))
    assert res["status"] == "requires_allow_large_download"


def test_extract_images_from_numpy():
    example = {
        "observation.images.top": np.zeros((8, 8, 3), np.uint8),
        "observation.state": [0.1, 0.2],
        "action": [0.0],
    }
    imgs = _extract_images(example)
    assert "observation.images.top" in imgs
    assert imgs["observation.images.top"].shape == (8, 8, 3)
