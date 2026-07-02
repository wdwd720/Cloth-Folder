"""Tests for the two-track refinement: strict pose-positive tasks, auto-triage,
track-aware export, and the critic-dataset builder.

No network / no SDK: generation via ``image_responder`` seam, Claude via
``responder`` seam, small PNGs from the pure codec.
"""

from __future__ import annotations

import json
import os

import numpy as np

from terafold.towel import critic_dataset as cd
from terafold.towel import openai_generation as og
from terafold.towel import claude_pseudolabel as cl
from terafold.towel import prompt_bank as pb
from terafold.towel import review as rv
from terafold.towel import yolo_export as yx
from terafold.towel.dataset import load_label, load_manifest
from terafold.vision.imageio import _png_encode


def _silent(*_a) -> None:
    pass


def _fake_image(prompt, size_str):
    w, h = (min(int(x), 48) for x in size_str.split("x"))
    rng = np.random.RandomState(abs(hash(prompt)) % (2 ** 31))
    return _png_encode((rng.rand(h, w, 3) * 255).astype(np.uint8))


def _label_dict(state, conf, risk=None, visible=None):
    return {
        "state": state, "usable_for_training": True, "bbox": [6, 6, 42, 42],
        "corners": {"tl": [6, 6], "tr": [42, 6], "br": [42, 42], "bl": [6, 42]},
        "visible": visible or {"tl": True, "tr": True, "br": True, "bl": True},
        "fold_axis": "vertical", "grasp_edge": "right", "place_edge": "left",
        "confidence": conf, "risk_reasons": risk or [], "notes": "x",
    }


# ============================ Deliverable 1/2: tasks + prompts ============================


def test_new_pose_tasks_exist():
    for t in ("pose_positive_flat_only", "pose_positive_diverse_v1", "critic_negatives_v1"):
        assert t in pb.list_tasks()


def test_positive_only_task_has_no_negative_categories():
    for task in ("pose_positive_flat_only", "pose_positive_diverse_v1"):
        recs = pb.build_prompt_curriculum(task, 30, seed=0)
        assert {r["training_track"] for r in recs} == {"pose_positive"}
        assert all(r["usable_hint"] for r in recs)
        assert all(r["target_state"] in ("flat_unfolded", "wrinkled_unfolded") for r in recs)


def test_flat_only_proportions():
    c = pb.allocate_counts("pose_positive_flat_only", 100)
    assert c == {"pose_flat_topdown": 50, "pose_flat_oblique": 30, "pose_wrinkled": 20}


def test_positive_prompts_include_hard_constraints():
    recs = pb.build_prompt_curriculum("pose_positive_flat_only", 12, seed=1)
    for r in recs:
        p = r["prompt"]
        assert "All four outer corners" in p
        assert "fully visible" in p
        for neg in ("not folded", "not rolled", "not stacked", "not hanging",
                    "no people", "no hands"):
            assert neg in p, neg


def test_unfolded_sheet_task_exists_and_mix():
    assert "pose_positive_unfolded_sheet_v1" in pb.list_tasks()
    assert pb.allocate_counts("pose_positive_unfolded_sheet_v1", 100) == {
        "pose_flat_topdown": 70, "pose_flat_near_topdown": 20, "pose_wrinkled": 10}
    recs = pb.build_prompt_curriculum("pose_positive_unfolded_sheet_v1", 30, seed=3)
    assert {r["training_track"] for r in recs} == {"pose_positive"}
    assert all(r["target_state"] in ("flat_unfolded", "wrinkled_unfolded") for r in recs)


def test_stronger_pose_prompts_single_layer_constraints():
    for task in ("pose_positive_flat_only", "pose_positive_diverse_v1",
                 "pose_positive_unfolded_sheet_v1"):
        for r in pb.build_prompt_curriculum(task, 18, seed=1):
            p = r["prompt"]
            assert "single-layer" in p
            assert "fully spread open" in p
            assert "not folded into a smaller rectangle" in p


def test_pose_prompts_have_no_ambiguous_folded_stack_language():
    # Positive pose prompts must not POSITIVELY describe a folded/stacked towel.
    forbidden = ("neatly folded", "half-folded", "folded into a tidy",
                 "neatly arranged hotel towel")
    for task in ("pose_positive_flat_only", "pose_positive_diverse_v1",
                 "pose_positive_unfolded_sheet_v1"):
        for r in pb.build_prompt_curriculum(task, 18, seed=2):
            for bad in forbidden:
                assert bad not in r["prompt"], (task, bad)


def test_critic_task_has_only_negative_categories():
    recs = pb.build_prompt_curriculum("critic_negatives_v1", 24, seed=2)
    assert {r["training_track"] for r in recs} == {"critic_negative"}
    assert not any(r["usable_hint"] for r in recs)


def test_legacy_mix_task_unchanged_track_metadata():
    recs = pb.build_prompt_curriculum("mix_flat_wrinkled_negatives", 20, seed=1)
    assert all("Photorealistic" in r["prompt"] for r in recs)  # legacy suffix preserved
    assert {r["training_track"] for r in recs} == {"pose_positive", "critic_negative"}


# ============================ track propagation ============================


def _labeled(tmp_path, scenarios, task="pose_positive_flat_only"):
    """Generate n=len(scenarios) images then Claude-label per id from `scenarios`."""
    gen = str(tmp_path / "gen")
    og.generate_openai_towel_dataset(gen, len(scenarios), task=task, dry_run=False,
                                     image_responder=_fake_image, confirm=True,
                                     max_cost_usd=100, log=_silent)
    lab = str(tmp_path / "lab")

    def responder(path, w, h):
        sid = os.path.splitext(os.path.basename(path))[0]
        return json.dumps(scenarios[sid])

    cl.claude_label_towel_folder(gen, lab, responder=responder, log=_silent)
    return lab


def test_training_track_propagates_to_label(tmp_path):
    sc = {f"{i:06d}": _label_dict("flat_unfolded", 0.95) for i in range(1, 4)}
    lab = _labeled(tmp_path, sc, task="pose_positive_flat_only")
    for s in load_manifest(lab)["samples"]:
        label = load_label(lab, s)
        assert s.get("training_track") == "pose_positive"
        assert label.get("training_track") == "pose_positive"


# ============================ Deliverable 3: auto-triage ============================

# 6 scenarios exercising approve / reject(state) / reject(low-conf) / review(uncertain)
_SCN = {
    "000001": _label_dict("flat_unfolded", 0.95),         # approve
    "000002": _label_dict("folded_success", 0.95),        # reject state
    "000003": _label_dict("partially_folded", 0.95),      # reject state
    "000004": _label_dict("multiple_towels", 0.95),       # reject state
    "000005": _label_dict("bad_view", 0.95),              # reject state
    "000006": _label_dict("flat_unfolded", 0.60),         # reject low-conf
    "000007": _label_dict("flat_unfolded", 0.80),         # review (uncertain band)
    "000008": _label_dict("wrinkled_unfolded", 0.95),     # approve
}


def _triage_dataset(tmp_path):
    return _labeled(tmp_path, _SCN, task="pose_positive_flat_only")


def test_auto_triage_dry_run_does_not_modify(tmp_path):
    lab = _triage_dataset(tmp_path)
    before = {s["id"]: load_label(lab, s)["approval_status"] for s in load_manifest(lab)["samples"]}
    res = rv.auto_triage_pseudolabels(lab, apply=False, log=_silent)
    after = {s["id"]: load_label(lab, s)["approval_status"] for s in load_manifest(lab)["samples"]}
    assert before == after                      # labels untouched in dry-run
    assert os.path.exists(res["report"])
    assert res["auto_approved"] == 2 and res["auto_rejected"] == 5 and res["review_needed"] == 1


def test_auto_triage_apply_decisions(tmp_path):
    lab = _triage_dataset(tmp_path)
    rv.auto_triage_pseudolabels(lab, apply=True, log=_silent)
    st = {s["id"]: load_label(lab, s) for s in load_manifest(lab)["samples"]}
    assert st["000001"]["approval_status"] == "auto_approved" and st["000001"]["usable_for_training"]
    assert st["000008"]["approval_status"] == "auto_approved"
    for rid in ("000002", "000003", "000004", "000005", "000006"):
        assert st[rid]["approval_status"] == "rejected"
        assert st[rid]["usable_for_training"] is False
    assert st["000007"]["approval_status"] == "review_needed"


# --- outer_visible_corners_any_state policy ---

# states: flat / partially_folded / folded_success / wrinkled approve; the rest reject/review
_SCN_ANY = {
    "000001": _label_dict("flat_unfolded", 0.95),                      # approve
    "000002": _label_dict("partially_folded", 0.95),                  # approve (NEW)
    "000003": _label_dict("folded_success", 0.95),                    # approve (NEW)
    "000004": _label_dict("wrinkled_unfolded", 0.95),                 # approve
    "000005": _label_dict("multiple_towels", 0.95),                   # reject state
    "000006": _label_dict("bad_view", 0.95),                          # reject state
    "000007": _label_dict("partially_folded", 0.60),                  # reject low-conf
    "000008": _label_dict("partially_folded", 0.80),                  # review (uncertain)
    "000009": _label_dict("folded_success", 0.95, risk=["hands cover a corner"]),  # reject risk
    "000010": _label_dict("partially_folded", 0.95,
                          visible={"tl": True, "tr": False, "br": True, "bl": True}),  # hidden
}


def test_new_policy_registered():
    assert "outer_visible_corners_any_state" in rv.TRIAGE_POLICIES


def test_any_state_policy_accepts_folded_and_partial(tmp_path):
    lab = _labeled(tmp_path, _SCN_ANY)
    res = rv.auto_triage_pseudolabels(lab, policy="outer_visible_corners_any_state",
                                      apply=True, log=_silent)
    st = {s["id"]: load_label(lab, s) for s in load_manifest(lab)["samples"]}
    # all four trainable states with clean visible corners get approved
    for ok in ("000001", "000002", "000003", "000004"):
        assert st[ok]["approval_status"] == "auto_approved", ok
        assert st[ok]["usable_for_training"] is True
    # rejects
    for bad in ("000005", "000006", "000007", "000009", "000010"):
        assert st[bad]["approval_status"] == "rejected", bad
        assert st[bad]["usable_for_training"] is False
    # uncertain confidence stays for human review (never auto-approved low-conf)
    assert st["000008"]["approval_status"] == "review_needed"
    assert res["auto_approved"] == 4 and res["review_needed"] == 1


def test_pose_positive_strict_still_rejects_folded(tmp_path):
    # Requirement 5: the strict policy is unchanged — folded/partial still rejected.
    lab = _labeled(tmp_path, _SCN_ANY)
    rv.auto_triage_pseudolabels(lab, policy="pose_positive_strict", apply=True, log=_silent)
    st = {s["id"]: load_label(lab, s)["approval_status"] for s in load_manifest(lab)["samples"]}
    assert st["000002"] == "rejected"   # partially_folded rejected by strict
    assert st["000003"] == "rejected"   # folded_success rejected by strict
    assert st["000001"] == "auto_approved"  # flat still approved


def test_any_state_dry_run_does_not_modify(tmp_path):
    lab = _labeled(tmp_path, _SCN_ANY)
    before = {s["id"]: load_label(lab, s)["approval_status"] for s in load_manifest(lab)["samples"]}
    res = rv.auto_triage_pseudolabels(lab, policy="outer_visible_corners_any_state",
                                      apply=False, log=_silent)
    after = {s["id"]: load_label(lab, s)["approval_status"] for s in load_manifest(lab)["samples"]}
    assert before == after and os.path.exists(res["report"])


def test_any_state_never_approves_low_confidence(tmp_path):
    lab = _labeled(tmp_path, {"000001": _label_dict("partially_folded", 0.50)})
    res = rv.auto_triage_pseudolabels(lab, policy="outer_visible_corners_any_state",
                                      apply=True, log=_silent)
    assert res["auto_approved"] == 0


def test_auto_triage_unknown_policy_errors(tmp_path):
    lab = _labeled(tmp_path, {"000001": _label_dict("flat_unfolded", 0.95)})
    assert rv.auto_triage_pseudolabels(lab, policy="nope", log=_silent)["status"] == "error"


def test_auto_triage_never_approves_everything(tmp_path):
    lab = _triage_dataset(tmp_path)
    res = rv.auto_triage_pseudolabels(lab, apply=True, log=_silent)
    assert res["auto_approved"] < res["considered"]  # not a blanket approve


# ============================ Deliverable 4: track-aware export ============================


def test_export_excludes_critic_negatives_by_default(tmp_path):
    # A critic-track dataset, Claude-labeled (stub flat) and force-approved:
    sc = {f"{i:06d}": _label_dict("flat_unfolded", 0.95) for i in range(1, 5)}
    lab = _labeled(tmp_path, sc, task="critic_negatives_v1")
    # Force-approve so only the track exclusion can stop them.
    for s in load_manifest(lab)["samples"]:
        lb = load_label(lab, s)
        lb["approval_status"] = "approved"
        lb["usable_for_training"] = True
        from terafold.data.episode_schema import write_json
        write_json(os.path.join(lab, s["label"]), lb)
    res = yx.export_yolo_towel_pose(lab, str(tmp_path / "y"),
                                    include_pseudolabels="approved_only", log=_silent)
    assert res["exported"] == 0
    assert res["skipped_critic_negatives"] == 4
    # ...unless explicitly requested:
    res2 = yx.export_yolo_towel_pose(lab, str(tmp_path / "y2"),
                                     include_pseudolabels="approved_only",
                                     include_critic_negatives=True, log=_silent)
    assert res2["exported"] == 4


def test_export_reports_track_and_approval_breakdown(tmp_path):
    lab = _triage_dataset(tmp_path)
    rv.auto_triage_pseudolabels(lab, apply=True, log=_silent)
    res = yx.export_yolo_towel_pose(lab, str(tmp_path / "y"),
                                    include_pseudolabels="approved_only", log=_silent)
    assert res["exported"] == 2  # the two approved flat/wrinkled
    assert res["by_source"]["openai_generated"] == 2
    assert res["by_training_track"].get("pose_positive") == 2
    assert res["pseudolabel_approval_status"]["rejected"] == 5


# ============================ Deliverable 5: critic builder ============================


def test_build_critic_from_review(tmp_path):
    lab = _triage_dataset(tmp_path)
    rv.auto_triage_pseudolabels(lab, apply=True, log=_silent)
    out = str(tmp_path / "critic")
    res = cd.build_towel_critic_from_review(lab, out, log=_silent)
    assert res["status"] == "ok"
    # 2 approved flat/wrinkled -> usable; 5 rejected -> not_usable; 1 review -> skipped
    assert res["by_class"]["usable"] == 2
    assert res["by_class"]["not_usable"] == 5
    assert res["skipped_undecided"] == 1
    man = load_manifest(out)
    e0 = next(s for s in man["samples"] if s["target_class"] == "not_usable")
    assert "rejection_reason" in e0 and "state" in e0 and "source" in e0


# ============================ CLI registration ============================


def test_cli_registers_track_commands():
    from terafold.cli import app
    names = {c.name for c in app.registered_commands}
    assert "auto-triage-pseudolabels" in names
    assert "build-towel-critic-from-review" in names
