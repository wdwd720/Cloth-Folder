"""Tests for the OpenAI-generated towel image pipeline (Layer B).

NO network and NO OpenAI SDK: generation is driven through the ``image_responder``
injection seam (returns PNG bytes) and Claude labeling through its ``responder``
seam. Images are small PNGs from the pure-Python codec.
"""

from __future__ import annotations

import json
import os

import numpy as np

from terafold.towel import costing as co
from terafold.towel import generated_dataset as gd
from terafold.towel import openai_generation as og
from terafold.towel import prompt_bank as pb
from terafold.towel import qc
from terafold.towel import claude_pseudolabel as cl
from terafold.towel import review as rv
from terafold.towel import yolo_export as yx
from terafold.towel import merge as mg
from terafold.towel import training_plan as tp
from terafold.towel.dataset import load_label, load_manifest
from terafold.vision.imageio import _png_encode, imwrite


def _silent(*_a) -> None:
    pass


def _fake_image(prompt, size_str):
    """Deterministic-but-distinct small PNG per prompt (structurally varied)."""
    w, h = (min(int(x), 48) for x in size_str.split("x"))
    rng = np.random.RandomState(abs(hash(prompt)) % (2 ** 31))
    arr = (rng.rand(h, w, 3) * 255).astype(np.uint8)
    return _png_encode(arr)


# ============================ prompt bank ============================


def test_tasks_and_categories_exist():
    assert "mix_flat_wrinkled_negatives" in pb.list_tasks()
    keys = pb.list_categories()
    for need in ("positive_flat", "positive_wrinkled", "positive_challenging",
                 "negative_multiple_towels", "negative_bad_view", "negative_not_towel",
                 "maybe_folded"):
        assert need in keys


def test_allocate_counts_sum_exact():
    for task in pb.list_tasks():
        for n in (10, 37, 100, 500):
            counts = pb.allocate_counts(task, n)
            assert sum(counts.values()) == n


def test_curriculum_deterministic_and_seed_varies():
    a = pb.build_prompt_curriculum("mix_flat_wrinkled_negatives", 20, seed=1)
    b = pb.build_prompt_curriculum("mix_flat_wrinkled_negatives", 20, seed=1)
    c = pb.build_prompt_curriculum("mix_flat_wrinkled_negatives", 20, seed=2)
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]
    assert [r["prompt"] for r in a] != [r["prompt"] for r in c]
    assert all("Photorealistic" in r["prompt"] for r in a)


def test_balanced_task_covers_all_categories():
    recs = pb.build_prompt_curriculum("balanced", 70, seed=0)
    cats = {r["category"] for r in recs}
    assert cats == set(pb.list_categories())


# ============================ costing ============================


def test_cost_estimate_and_budget():
    e = co.estimate_cost("gpt-image-1", 1024, 500, quality="medium")
    assert e["total_usd"] > 0 and e["known_pricing"] is True
    assert co.within_budget(e, 100.0) is True
    assert co.within_budget(e, 1.0) is False
    assert co.within_budget(e, None) is True


def test_cost_unknown_model_flagged():
    e = co.estimate_cost("totally-made-up", 1024, 10)
    assert e["known_pricing"] is False and e["total_usd"] > 0


def test_parse_size_variants():
    assert co.parse_size(1024) == (1024, 1024)
    assert co.parse_size("1024x1536") == (1024, 1536)
    assert co.parse_size("auto") == (1024, 1024)


# ============================ generation: gates ============================


def test_missing_openai_key_graceful(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = og.generate_openai_towel_dataset(str(tmp_path / "g"), 5, dry_run=False, log=_silent)
    assert res["status"] == "no_api_key"
    assert any("OPENAI_API_KEY" in line for line in res["instructions"])


def test_dry_run_preview_no_calls(tmp_path):
    out = str(tmp_path / "g")
    res = og.generate_openai_towel_dataset(out, 50, task="mix_flat_wrinkled_negatives",
                                           dry_run=True, log=_silent)
    assert res["status"] == "dry_run"
    assert res["num_requested"] == 50 and len(res["preview"]) > 0
    assert res["cost_estimate"]["total_usd"] > 0
    # No images were generated (dry run).
    assert not os.path.isdir(os.path.join(out, "images")) or not os.listdir(os.path.join(out, "images"))


def test_needs_confirmation_before_paid(tmp_path):
    res = og.generate_openai_towel_dataset(str(tmp_path / "g"), 5, dry_run=False,
                                           image_responder=_fake_image, log=_silent)
    assert res["status"] == "needs_confirmation"
    assert og.CONFIRM_FLAG in res["confirm_flag"]


def test_max_cost_refusal(tmp_path):
    res = og.generate_openai_towel_dataset(str(tmp_path / "g"), 50, dry_run=False,
                                           image_responder=_fake_image, confirm=True,
                                           max_cost_usd=0.0001, log=_silent)
    assert res["status"] == "cost_exceeded"


def test_first_image_failure_aborts(tmp_path):
    def boom(_p, _s):
        raise RuntimeError("invalid model")

    res = og.generate_openai_towel_dataset(str(tmp_path / "g"), 5, dry_run=False,
                                           image_responder=boom, confirm=True,
                                           max_cost_usd=100, log=_silent)
    assert res["status"] == "generation_failed"


# ============================ generation: real (responder) ============================


def _generate(tmp_path, n=8, task="pose_positive_flat_only"):
    out = str(tmp_path / "gen")
    res = og.generate_openai_towel_dataset(out, n, task=task, dry_run=False,
                                           image_responder=_fake_image, confirm=True,
                                           max_cost_usd=100, log=_silent)
    assert res["status"] == "ok"
    return out, res


def test_generation_manifest_schema_validates(tmp_path):
    out, res = _generate(tmp_path, n=8)
    man = gd.load_generated_manifest(out)
    assert gd.validate_generated_manifest(man) == []
    assert man["num_images"] == 8 and res["num_images"] == 8
    s0 = man["images"][0]
    assert s0["source"] == "openai_generated" and s0["usable_for_training"] is False
    assert s0["label_status"] == "unlabeled" and s0["approval_status"] == "pending"


def test_resume_does_not_duplicate(tmp_path):
    out, _ = _generate(tmp_path, n=6)
    res2 = og.generate_openai_towel_dataset(out, 6, dry_run=False, image_responder=_fake_image,
                                            confirm=True, max_cost_usd=100, resume=True, log=_silent)
    assert res2["status"] == "ok"
    assert res2["resumed"] == 6 and res2["num_images"] == 0
    assert len(os.listdir(os.path.join(out, "images"))) == 6


# ============================ QC / dedup ============================


def _gen_manifest_with(tmp_path, entries):
    inp = tmp_path / "raw"
    (inp / "images").mkdir(parents=True)
    images = []
    for e in entries:
        imwrite(str(inp / e["image"]), e.pop("_arr"))
        images.append({"source": "openai_generated", "generation_model": "gpt-image-1",
                       "generation_prompt": "a towel", "prompt_template_name": "t",
                       "category_target": "positive_flat", "hash": "", "width": None,
                       "height": None, **e})
    from terafold.data.episode_schema import write_json
    write_json(str(inp / "manifest.json"),
               {"kind": "openai_generated", "source": "openai_generated",
                "generation_model": "gpt-image-1", "images": images})
    return str(inp)


def test_qc_detects_exact_and_near_duplicates(tmp_path):
    def block(val, shift=0):
        a = np.full((80, 80, 3), 20, np.uint8)
        a[10 + shift:40 + shift, 10:40] = val   # structurally distinct positions
        return a

    inp = _gen_manifest_with(tmp_path, [
        {"image": "images/a.png", "_arr": block(200, 0)},
        {"image": "images/a_dup.png", "_arr": block(200, 0)},   # exact duplicate of a
        {"image": "images/b.png", "_arr": block(150, 35)},      # structurally different
    ])
    res = qc.qc_filter_generated_images(inp, str(tmp_path / "qc"), min_size=64, log=_silent)
    assert res["status"] == "ok"
    assert res["reasons"].get("exact_duplicate", 0) >= 1
    assert res["kept"] == 2  # a + b survive; a_dup dropped


def test_qc_drops_corrupt_and_tiny(tmp_path):
    inp = tmp_path / "raw"
    (inp / "images").mkdir(parents=True)
    imwrite(str(inp / "images/ok.png"), np.full((80, 80, 3), 90, np.uint8))
    imwrite(str(inp / "images/tiny.png"), np.full((20, 20, 3), 90, np.uint8))
    with open(inp / "images/bad.png", "wb") as f:
        f.write(b"garbage-not-an-image")
    from terafold.data.episode_schema import write_json
    write_json(str(inp / "manifest.json"), {"kind": "openai_generated", "source": "openai_generated",
               "generation_model": "m", "images": [
                   {"image": "images/ok.png", "source": "openai_generated", "hash": "",
                    "generation_model": "m", "generation_prompt": "x", "prompt_template_name": "t",
                    "category_target": "positive_flat"},
                   {"image": "images/tiny.png", "source": "openai_generated", "hash": ""},
                   {"image": "images/bad.png", "source": "openai_generated", "hash": ""}]})
    res = qc.qc_filter_generated_images(str(inp), str(tmp_path / "qc"), min_size=64, log=_silent)
    assert res["kept"] == 1
    assert res["reasons"].get("too_small", 0) == 1
    assert res["reasons"].get("corrupt_or_unreadable", 0) == 1


# ============================ full flow: gen -> label -> review -> export ============================

W = H = 48
CLEAN = {
    "state": "flat_unfolded", "usable_for_training": True, "bbox": [6, 6, 42, 42],
    "corners": {"tl": [6, 6], "tr": [42, 6], "br": [42, 42], "bl": [6, 42]},
    "visible": {"tl": True, "tr": True, "br": True, "bl": True},
    "fold_axis": "vertical", "grasp_edge": "right", "place_edge": "left",
    "confidence": 0.95, "risk_reasons": [], "notes": "clean flat towel",
}


def _labeled_generated(tmp_path, n=6):
    gen_out, _ = _generate(tmp_path, n=n)
    lab_out = str(tmp_path / "labeled")
    res = cl.claude_label_towel_folder(gen_out, lab_out,
                                       responder=lambda p, w, h: json.dumps(CLEAN), log=_silent)
    assert res["status"] == "ok"
    return lab_out


def test_generated_flows_through_label_keeps_provenance(tmp_path):
    lab_out = _labeled_generated(tmp_path, n=4)
    man = load_manifest(lab_out)
    assert len(man["samples"]) == 4
    for s in man["samples"]:
        lab = load_label(lab_out, s)
        assert s["source"] == "openai_generated"          # provenance preserved
        assert lab["pseudolabel"] is True                 # but it is a machine label
        assert lab["usable_for_training"] is False         # untrusted until approved
        assert s.get("generation_model") == "gpt-image-1"  # lineage carried through


def test_export_excludes_generated_pseudolabels_by_default(tmp_path):
    lab_out = _labeled_generated(tmp_path, n=4)
    res = yx.export_yolo_towel_pose(lab_out, str(tmp_path / "y"), log=_silent)  # default none
    assert res["exported"] == 0 and res["skipped_pseudolabels"] == 4


def test_approved_generated_exports_as_openai_generated(tmp_path):
    lab_out = _labeled_generated(tmp_path, n=4)
    a = rv.approve_high_confidence_pseudolabels(lab_out, 0.90, 0.10, log=_silent)
    assert a["auto_approved"] == 4
    res = yx.export_yolo_towel_pose(lab_out, str(tmp_path / "y"),
                                    include_pseudolabels="approved_only", log=_silent)
    assert res["exported"] == 4
    assert res["by_source"]["openai_generated"] == 4


def test_review_source_filter(tmp_path):
    lab_out = _labeled_generated(tmp_path, n=3)
    # Filter to a non-matching source -> nothing reviewed.
    res = rv.review_pseudolabels(lab_out, input_fn=lambda _p: "approve",
                                 source_filter="kaggle", log=_silent)
    assert res["approved"] == 0
    # Filter to the real source -> all approved.
    res2 = rv.review_pseudolabels(lab_out, input_fn=lambda _p: "approve",
                                  source_filter="openai_generated", log=_silent)
    assert res2["approved"] == 3


def test_merge_preserves_openai_generated_source(tmp_path):
    lab_out = _labeled_generated(tmp_path, n=4)
    rv.approve_high_confidence_pseudolabels(lab_out, 0.90, 0.10, log=_silent)
    merged = str(tmp_path / "master")
    res = mg.merge_towel_datasets([lab_out], merged, log=_silent)
    assert res["num_images"] == 4
    assert res["by_source"]["openai_generated"] == 4
    man = load_manifest(merged)
    assert all(s["source"] == "openai_generated" for s in man["samples"])
    assert all(load_label(merged, s).get("pseudolabel") for s in man["samples"])


# ============================ training plan / command ============================


def test_recommend_plan_on_merged_dataset(tmp_path):
    lab_out = _labeled_generated(tmp_path, n=4)
    rv.approve_high_confidence_pseudolabels(lab_out, 0.90, 0.10, log=_silent)
    merged = str(tmp_path / "master")
    mg.merge_towel_datasets([lab_out], merged, log=_silent)
    res = tp.recommend_towel_training_plan(merged, out=str(tmp_path / "plan.md"), log=_silent)
    assert res["status"] == "ok"
    assert res["composition"]["by_source"]["openai_generated"] == 4
    assert os.path.exists(res["report"])
    assert res["stages"]  # staged strategy present


def test_recommend_plan_on_yolo_dataset(tmp_path):
    lab_out = _labeled_generated(tmp_path, n=4)
    rv.approve_high_confidence_pseudolabels(lab_out, 0.90, 0.10, log=_silent)
    yolo = str(tmp_path / "y")
    yx.export_yolo_towel_pose(lab_out, yolo, include_pseudolabels="approved_only", log=_silent)
    res = tp.recommend_towel_training_plan(yolo, log=_silent)
    assert res["status"] == "ok" and res["composition"]["kind"] == "yolo_pose"


def test_print_training_command_on_exported(tmp_path):
    lab_out = _labeled_generated(tmp_path, n=4)
    rv.approve_high_confidence_pseudolabels(lab_out, 0.90, 0.10, log=_silent)
    yolo = str(tmp_path / "y")
    yx.export_yolo_towel_pose(lab_out, yolo, include_pseudolabels="approved_only", log=_silent)
    from terafold.towel.yolo_runtime import print_towel_training_command
    res = print_towel_training_command(yolo, model="yolo11n-pose.pt", epochs=100, imgsz=640)
    assert "yolo" in res["command"].lower()


# ============================ summarize + CLI registration ============================


def test_summarize_generated_dataset(tmp_path):
    out, _ = _generate(tmp_path, n=6)
    res = gd.summarize_generated_dataset(out, out=str(tmp_path / "s.md"), log=_silent)
    assert res["num_images"] == 6
    assert sum(res["by_category"].values()) == 6
    assert os.path.exists(res["report"])


def test_cli_registers_new_commands():
    from terafold.cli import app
    names = {c.name for c in app.registered_commands}
    for cmd in ("generate-openai-towel-dataset", "preview-openai-towel-prompts",
                "summarize-openai-towel-dataset", "filter-generated-towel-images",
                "recommend-towel-training-plan"):
        assert cmd in names, cmd


def test_generated_images_never_auto_usable(tmp_path):
    """Safety-relevant invariant: generated images are never trainable without approval."""
    out, _ = _generate(tmp_path, n=4)
    man = gd.load_generated_manifest(out)
    assert all(s["usable_for_training"] is False for s in man["images"])
    # And after pseudo-labeling, still not usable until approved.
    lab_out = _labeled_generated(tmp_path, n=4)
    for s in load_manifest(lab_out)["samples"]:
        assert load_label(lab_out, s)["usable_for_training"] is False
