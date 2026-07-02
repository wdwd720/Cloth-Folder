"""Tests for the multi-stage HIGH-PRECISION OpenAI towel labeler (``--multipass``).

NO network / NO OpenAI SDK. The three vision stages are driven through a
``stage_responder(stage, image_path, w, h)`` seam and, separately, through a FAKE
Responses ``client`` (a mocked OpenAI response that branches on the system prompt) —
so both the high-level pipeline and the real Responses-API request shape are covered.
Images are PNGs from the pure-Python codec.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from terafold.towel import claude_pseudolabel as cl
from terafold.towel import openai_multipass as mp
from terafold.towel import openai_pseudolabel as op
from terafold.towel.dataset import load_label, load_manifest
from terafold.data.episode_schema import read_json
from terafold.vision.imageio import imwrite

POLICY = "striped_towel_visible_outer_corners"

# A 200x150 scene with the towel rectangle at original [60..140, 40..110].
# Localize bbox [60,40,140,110] + 0.08 pad -> crop offset (x_off=53, y_off=34), 94x82.
W, H = 200, 150
LOC_BBOX = [60, 40, 140, 110]
X_OFF, Y_OFF = 53, 34


def _silent(*_a) -> None:
    pass


def _img_folder(tmp_path, n=3):
    d = tmp_path / "in"
    d.mkdir()
    for i in range(n):
        img = np.full((H, W, 3), 30, np.uint8)
        img[40:110, 60:140] = 200
        imwrite(str(d / f"IMG_{i}.png"), img)
    return str(d)


def _loc(bbox=LOC_BBOX, found=True, is_target=True, conf=0.93):
    return json.dumps({"found": found, "is_target": is_target, "bbox": list(bbox),
                       "confidence": conf, "rejected_regions": ["bed", "pillow"]})


# a rotated, tight quad in CROP coordinates (NOT axis-aligned, well inside the crop)
ROTATED_CROP = {"tl": [14, 9], "tr": [84, 16], "br": [80, 72], "bl": [10, 66]}


def _cor(corners=None, state="flat_unfolded", is_target=True, conf=0.92):
    return json.dumps({"is_target_towel_visible": is_target, "reject_reason": None,
                       "state": state, "corners": corners or ROTATED_CROP,
                       "visible": {"tl": True, "tr": True, "br": True, "bl": True},
                       "confidence": conf, "risks": []})


def _ver(verdict="accept", all_on=True, on_bg=False, loose=False, tight=True,
         corrected=None, reason="good"):
    return json.dumps({"all_on_towel": all_on, "any_on_background": on_bg,
                       "is_loose_bbox": loose, "tight_enough_for_yolo": tight,
                       "verdict": verdict, "corrected_corners": corrected, "reason": reason})


def _responder(loc=None, cor=None, ver=None):
    loc, cor, ver = loc or _loc(), cor or _cor(), ver or _ver()

    def sr(stage, image_path, w, h):
        return {"localize": loc, "corners": cor, "verify": ver}[stage]
    return sr


# ============================ command + delegation ============================


def test_multipass_flag_registered():
    from terafold.cli import app

    import inspect

    cmd = next(c for c in app.registered_commands
               if (c.name or "") == "openai-label-towel-folder")
    # the --multipass / --crop-padding / --verify params exist on the command
    sig = inspect.signature(cmd.callback)
    for opt in ("multipass", "crop_padding", "verify"):
        assert opt in sig.parameters


def test_openai_label_delegates_to_multipass(tmp_path):
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(inp, out, multipass=True, force=True,
                                       stage_responder=_responder(), label_policy=POLICY,
                                       log=_silent)
    assert res["status"] == "ok"
    assert load_manifest(out)["pipeline"] == "multipass"


# ============================ verify prompt + parse ============================


def test_verify_prompt_asks_the_four_questions():
    p = mp.build_verify_prompt(W, H, [[1, 2], [3, 4], [5, 6], [7, 8]])
    for q in ("all_on_towel", "any_on_background", "is_loose_bbox", "tight_enough_for_yolo"):
        assert q in p
    assert "corrected_corners" in p and "accept" in p and "reject" in p


@pytest.mark.parametrize("raw,verdict", [
    (_ver(verdict="accept"), "accept"),
    (_ver(verdict="correct", corrected={"tl": [1, 1], "tr": [2, 1], "br": [2, 2], "bl": [1, 2]}), "correct"),
    (_ver(verdict="reject"), "reject"),
    (json.dumps({"verdict": "weird"}), "reject"),   # unknown -> conservative reject
])
def test_parse_verification_verdict(raw, verdict):
    assert mp.parse_verification(raw)["verdict"] == verdict


def test_parse_verification_robust_bools_and_partial_corrected():
    v = mp.parse_verification(json.dumps({
        "all_on_towel": "false", "any_on_background": "true", "verdict": "correct",
        "corrected_corners": {"tl": [1, 1], "tr": [2, 1]}}))  # only 2 corners -> ignored
    assert v["all_on_towel"] is False and v["any_on_background"] is True
    assert v["corrected_corners"] is None


# ============================ geometry checks (#6) ============================


def test_geometry_clean_rotated_quad_passes():
    corners = [[64, 44], [136, 50], [132, 104], [60, 100]]  # rotated, inside towel region
    bbox = [60, 44, 136, 104]
    res = mp.multipass_geometry_checks(corners, bbox, W, H, LOC_BBOX, corners, 94, 82)
    assert res["reject"] is False and res["risks"] == []


def test_geometry_axis_aligned_is_flagged_not_rejected():
    corners = [[62, 42], [138, 42], [138, 108], [62, 108]]  # axis-aligned but tight
    bbox = [62, 42, 138, 108]
    res = mp.multipass_geometry_checks(corners, bbox, W, H, LOC_BBOX, None, None, None)
    assert "corners_axis_aligned_bbox" in res["risks"]
    assert res["reject"] is False   # a top-down towel can be axis-aligned -> review, not reject


def test_geometry_crop_border_fill_rejects():
    crop_corners = [[0, 0], [93, 0], [93, 81], [0, 81]]   # filled the crop frame
    full = [[53, 34], [146, 34], [146, 115], [53, 115]]
    res = mp.multipass_geometry_checks(full, [53, 34, 146, 115], W, H, LOC_BBOX,
                                       crop_corners, 94, 82)
    assert "corners_fill_crop_border" in res["risks"] and res["reject"] is True


def test_geometry_image_border_fill_rejects():
    full = [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
    res = mp.multipass_geometry_checks(full, [0, 0, W - 1, H - 1], W, H, LOC_BBOX)
    assert "corners_fill_image_border" in res["risks"] and res["reject"] is True


def test_geometry_corner_on_image_edge_rejects():
    full = [[1, 1], [136, 50], [132, 104], [60, 100]]  # tl ~ on the image edge
    res = mp.multipass_geometry_checks(full, [1, 1, 136, 104], W, H, LOC_BBOX)
    assert any(r.startswith("corner_on_image_edge") for r in res["risks"])
    assert res["reject"] is True


def test_geometry_impossible_quad_rejects():
    bowtie = [[60, 40], [140, 110], [140, 40], [60, 110]]  # self-crossing order
    res = mp.multipass_geometry_checks(bowtie, [60, 40, 140, 110], W, H, LOC_BBOX)
    assert "geometry_impossible" in res["risks"] and res["reject"] is True


def test_geometry_area_exceeds_towel_region_rejects():
    # quad bbox far larger than the detected towel bbox (area ratio > 1.8)
    full = [[20, 20], [180, 25], [178, 130], [22, 128]]
    res = mp.multipass_geometry_checks(full, [20, 20, 180, 130], W, H, LOC_BBOX)
    assert "quad_area_exceeds_towel_region" in res["risks"] and res["reject"] is True


def test_fills_frame_helper():
    assert mp._fills_frame([[0, 0], [99, 0], [99, 99], [0, 99]], 100, 100) is True
    assert mp._fills_frame([[20, 20], [80, 20], [80, 80], [20, 80]], 100, 100) is False


def test_crop_border_tolerance_scales_with_padding():
    # A true crop-FRAME label (corners ~0 inset) must be flagged; a CORRECT tight label
    # that fills a small-padding crop (~3% inset) must NOT be flagged -> no false reject.
    cw = ch = 100
    full = [[10, 10], [90, 11], [91, 90], [9, 89]]  # rotated full corners (other checks pass)
    fbb = [9, 10, 91, 90]
    frame = [[0, 0], [99, 0], [99, 99], [0, 99]]
    tight = [[3, 3], [96, 3], [96, 96], [3, 96]]    # correct towel filling a pad=0.03 crop
    r_frame = mp.multipass_geometry_checks(full, fbb, W, H, LOC_BBOX, frame, cw, ch, crop_pad_frac=0.03)
    r_tight = mp.multipass_geometry_checks(full, fbb, W, H, LOC_BBOX, tight, cw, ch, crop_pad_frac=0.03)
    assert "corners_fill_crop_border" in r_frame["risks"] and r_frame["reject"] is True
    assert "corners_fill_crop_border" not in r_tight["risks"]


def test_axis_aligned_tolerance_scales_with_towel_not_image():
    # A 600x450 towel rotated ~30px off-axis in a 4000x3000 photo is NOT axis-aligned;
    # tolerance must track the towel (loc_bbox), not the full-image diagonal.
    loc = [1000, 1000, 1600, 1450]
    corners = [[1030, 1000], [1600, 1030], [1570, 1450], [1000, 1420]]  # ~30px rotation
    bbox = [1000, 1000, 1600, 1450]
    res = mp.multipass_geometry_checks(corners, bbox, 4000, 3000, loc)
    assert "corners_axis_aligned_bbox" not in res["risks"] and res["risks"] == []


def test_multipass_tight_crop_does_not_false_reject(tmp_path):
    # End-to-end: a clean rotated label at small --crop-padding must not be rejected
    # by the crop-border gate (regression for the fixed-tolerance bug).
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    op.openai_label_towel_folder(inp, out, multipass=True, force=True, crop_padding=0.03,
                                 stage_responder=_responder(), label_policy=POLICY,
                                 max_crop_size=None, log=_silent)
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["approval_status"] != "rejected"
    assert "corners_fill_crop_border" not in lab["risk_reasons"]


# ============================ transport (mocked OpenAI client) ============================


class _FakeUsage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _FakeResponse:
    def __init__(self, text):
        self.output_text = text
        self.usage = _FakeUsage(10, 5)


class _FakeResponses:
    def __init__(self, loc, cor, ver):
        self._loc, self._cor, self._ver = loc, cor, ver
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        instr = kwargs["instructions"]
        if "QA checker" in instr:
            return _FakeResponse(self._ver)
        if "bounding box" in instr.lower():
            return _FakeResponse(self._loc)
        return _FakeResponse(self._cor)


class _FakeClient:
    def __init__(self, loc=None, cor=None, ver=None):
        self.responses = _FakeResponses(loc or _loc(), cor or _cor(), ver or _ver())


def test_transport_routes_three_stages_via_client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    fake = _FakeClient()
    res = op.openai_label_towel_folder(inp, out, multipass=True, force=True, client=fake,
                                       label_policy=POLICY, verify=True, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 1
    # exactly 3 Responses calls (localize, corners, verify), each with an image input
    assert len(fake.responses.calls) == 3
    for kw in fake.responses.calls:
        types = [c["type"] for c in kw["input"][0]["content"]]
        assert "input_image" in types and "input_text" in types
        assert kw["input"][0]["content"][1]["image_url"].startswith("data:image/")
    assert res["usage"]["input_tokens"] == 30  # 3 calls x 10


def test_no_verify_skips_third_call(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    fake = _FakeClient()
    op.openai_label_towel_folder(inp, out, multipass=True, verify=False, force=True,
                                 client=fake, label_policy=POLICY, log=_silent)
    assert len(fake.responses.calls) == 2  # localize + corners only
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["verification"]["skipped"] is True


# ============================ end-to-end pipeline ============================


def test_multipass_end_to_end_artifacts_and_remap(tmp_path):
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    op.openai_label_towel_folder(inp, out, multipass=True, force=True,
                                 stage_responder=_responder(), label_policy=POLICY,
                                 max_crop_size=None, log=_silent)
    man = load_manifest(out)
    assert man["pipeline"] == "multipass" and man["backend"] == "openai"
    s = man["samples"][0]
    lab = load_label(out, s)
    # remap exact: crop corner [14,9] -> original [14+53, 9+34] = [67, 43]
    assert lab["corners"]["tl"] == [float(14 + X_OFF), float(9 + Y_OFF)]
    assert lab["crop_corners"]["tl"] == [14, 9]
    assert lab["pipeline"] == "multipass" and lab["usable_for_training"] is False
    # all seven artifact kinds exist on disk
    for key in ("image", "crop", "crop_overlay", "localize_overlay", "overlay", "verify"):
        assert s[key] and os.path.exists(os.path.join(out, s[key])), key
    assert read_json(os.path.join(out, s["verify"]))["verdict"] == "accept"


def test_localization_reject_drops_image(tmp_path):
    inp = _img_folder(tmp_path, n=2)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(
        inp, out, multipass=True, force=True, label_policy=POLICY, log=_silent,
        stage_responder=_responder(loc=_loc(is_target=False)))
    assert res["num_images"] == 0 and res["rejected_localization"] == 2


def test_verify_reject_marks_label_rejected(tmp_path):
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(
        inp, out, multipass=True, force=True, label_policy=POLICY, log=_silent,
        stage_responder=_responder(ver=_ver(verdict="reject", all_on=False, on_bg=True, tight=False)))
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["approval_status"] == "rejected"
    assert "verify_on_background" in lab["risk_reasons"]
    assert res["label_rejected"] == 1


def test_verify_correct_adopts_corrected_corners(tmp_path):
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    corrected = {"tl": [64, 44], "tr": [136, 50], "br": [132, 104], "bl": [60, 100]}
    op.openai_label_towel_folder(
        inp, out, multipass=True, force=True, label_policy=POLICY, log=_silent,
        stage_responder=_responder(ver=_ver(verdict="correct", loose=True, tight=False,
                                            corrected=corrected)))
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["corners"]["tl"] == [64.0, 44.0]   # adopted the corrected corners
    assert "verify_corrected" in lab["risk_reasons"]


def test_clean_rotated_is_pending(tmp_path):
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    op.openai_label_towel_folder(inp, out, multipass=True, force=True,
                                 stage_responder=_responder(), label_policy=POLICY,
                                 max_crop_size=None, log=_silent)
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["approval_status"] == "pending" and lab["risk_reasons"] == []


def test_verify_correct_without_corners_routes_to_review(tmp_path):
    # verdict="correct" but corrected_corners unparseable (only 2 of 4) -> the model said
    # "needs fixing" with no usable fix, so it must NOT land in clean 'pending'.
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    partial = json.dumps({"all_on_towel": True, "any_on_background": False,
                          "is_loose_bbox": False, "tight_enough_for_yolo": True,
                          "verdict": "correct", "corrected_corners": {"tl": [1, 1], "tr": [2, 1]},
                          "reason": "nudge"})
    op.openai_label_towel_folder(inp, out, multipass=True, force=True, label_policy=POLICY,
                                 max_crop_size=None,
                                 stage_responder=_responder(ver=partial), log=_silent)
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["approval_status"] == "review_needed"
    assert "verify_correct_no_corners" in lab["risk_reasons"]


def test_sample_approval_matches_label_after_schema_downgrade(tmp_path, monkeypatch):
    # If validate_label downgrades the final label to review_needed, the manifest sample
    # and the run counts must agree with the on-disk label (no stale 'pending').
    monkeypatch.setattr(mp, "validate_label", lambda lab: ["forced schema problem"])
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(inp, out, multipass=True, force=True, label_policy=POLICY,
                                       max_crop_size=None, stage_responder=_responder(), log=_silent)
    man = load_manifest(out)
    s = man["samples"][0]
    lab = load_label(out, s)
    assert lab["approval_status"] == "review_needed"
    assert s["approval_status"] == "review_needed"            # not the stale 'pending'
    assert man["num_review_needed"] == 1 and man["num_pending"] == 0
    assert "schema_invalid" in s["risk_reasons"]


def test_max_images_caps_multipass(tmp_path):
    inp = _img_folder(tmp_path, n=5)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(inp, out, multipass=True, force=True, max_images=3,
                                       stage_responder=_responder(), label_policy=POLICY,
                                       log=_silent)
    assert res["num_images"] == 3


def test_multipass_no_api_key_graceful(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    inp = _img_folder(tmp_path, n=1)
    res = op.openai_label_towel_folder(inp, str(tmp_path / "o"), multipass=True, log=_silent)
    assert res["status"] == "no_api_key" and res["instructions"]


def test_multipass_rejects_non_striped_policy(tmp_path):
    inp = _img_folder(tmp_path, n=1)
    res = op.openai_label_towel_folder(inp, str(tmp_path / "o"), multipass=True,
                                       label_policy="default",
                                       stage_responder=_responder(), log=_silent)
    assert res["status"] == "error" and "label_policies" in res


# ============================ provider isolation + robust transport (goal) ============================


class _StageResponses:
    """A fake Responses client that returns canned text per stage (branches on the
    system prompt), records calls, and can be made to return empty / non-JSON."""

    def __init__(self, text_for):
        self._text_for = text_for   # callable(stage) -> str
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        instr = kwargs["instructions"]
        stage = "verify" if "QA checker" in instr else (
            "localize" if "bounding box" in instr.lower() else "corners")
        return _FakeResponse(self._text_for(stage))


class _StageClient:
    def __init__(self, text_for):
        self.responses = _StageResponses(text_for)


def test_multipass_never_instantiates_claude_labeler(tmp_path, monkeypatch):
    # Hard guarantee: the OpenAI multipass path must not construct a Claude labeler.
    import terafold.towel.claude_pseudolabel as clmod

    def boom(*_a, **_k):
        raise AssertionError("ClaudeTowelLabeler must not be used in OpenAI multipass")
    monkeypatch.setattr(clmod.ClaudeTowelLabeler, "__init__", boom)
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(inp, out, multipass=True, force=True, label_policy=POLICY,
                                       max_crop_size=None, stage_responder=_responder(), log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 1


def test_empty_response_is_openai_branded_not_claude(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    logs = []
    client = _StageClient(lambda stage: "")   # always empty
    op.openai_label_towel_folder(inp, out, multipass=True, force=True, client=client,
                                 label_policy=POLICY, max_crop_size=None,
                                 log=lambda m: logs.append(m))
    warns = [m for m in logs if "[warn]" in m]
    assert warns and "OpenAI returned an empty response" in warns[-1]
    assert "provider=openai" in warns[-1]
    assert not any("Claude" in m for m in logs)        # never Claude-branded
    # retried: 1 initial + 2 retries on the (first, localize) stage
    assert len(client.responses.calls) == 3


def test_debug_logging_and_artifact_on_parse_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    logs = []
    # localize returns prose (non-JSON, non-empty) -> parse fails -> debug artifact saved
    client = _StageClient(lambda stage: "I can't help with that." if stage == "localize" else "{}")
    op.openai_label_towel_folder(inp, out, multipass=True, force=True, client=client, debug=True,
                                 label_policy=POLICY, max_crop_size=None,
                                 log=lambda m: logs.append(m))
    # --debug prints stage/model/provider/image/extracted
    dbg_lines = [m for m in logs if "[debug] provider=openai stage=" in m]
    assert dbg_lines and "model=" in dbg_lines[0] and "text_extracted=" in dbg_lines[0]
    # a non-JSON response (not empty) is NOT retried -> exactly one localize call
    assert len(client.responses.calls) == 1
    # raw-response snippet saved to <out>/debug/, OpenAI-branded, and no key stored
    dbg_dir = os.path.join(out, "debug")
    files = os.listdir(dbg_dir)
    assert files and files[0].endswith("_localize.json")
    art = read_json(os.path.join(dbg_dir, files[0]))
    assert art["provider"] == "openai" and art["stage"] == "localize"
    assert "I can't help" in art["response_snippet"]
    assert "api_key" not in art and "OPENAI_API_KEY" not in json.dumps(art)
    assert not any("Claude" in m for m in logs)


def test_output_text_and_content_block_extraction():
    # output_text accessor
    assert op._openai_output_text({"output_text": '{"a": 1}'}) == '{"a": 1}'
    # content-block walk (Responses API output[].content[] with output_text blocks)
    blocks = {"output": [{"type": "message",
                          "content": [{"type": "output_text", "text": '{"b": 2}'}]}]}
    assert op._openai_output_text(blocks) == '{"b": 2}'
    # a refusal block is surfaced (so it lands in debug, not a silent empty)
    refusal = {"output": [{"content": [{"type": "refusal", "refusal": "no"}]}]}
    assert op._openai_output_text(refusal) == "no"
    # JSON embedded in prose / code fences is still recovered by _parse_json
    fenced = op._openai_output_text({"output_text": 'Here:\n```json\n{"c": 3}\n```'})
    assert cl._parse_json(fenced) == {"c": 3}


def test_json_mode_requested_then_falls_back(tmp_path, monkeypatch):
    # Structured JSON output is requested; if the SDK/model rejects it, fall back cleanly.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _PickyResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if "text" in kwargs:                      # json mode -> reject once
                raise TypeError("unexpected keyword argument 'text'")
            instr = kwargs["instructions"]
            stage = "verify" if "QA checker" in instr else (
                "localize" if "bounding box" in instr.lower() else "corners")
            return _FakeResponse(_responder()(stage, "x", 1, 1))

    class _PickyClient:
        def __init__(self):
            self.responses = _PickyResponses()

    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    client = _PickyClient()
    res = op.openai_label_towel_folder(inp, out, multipass=True, force=True, client=client,
                                       label_policy=POLICY, max_crop_size=None, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 1
    # the first attempt asked for json mode (text=...) and then a fallback without it
    assert any("text" in c for c in client.responses.calls)
    assert any("text" not in c for c in client.responses.calls)


def test_is_unsupported_param_classification():
    class _E(Exception):
        def __init__(self, status=None):
            self.status_code = status
    assert mp._is_unsupported_param(_E(400)) is True
    assert mp._is_unsupported_param(_E(422)) is True
    assert mp._is_unsupported_param(_E(429)) is False   # rate limit -> propagate
    assert mp._is_unsupported_param(_E(500)) is False   # server error -> propagate
    assert mp._is_unsupported_param(_E(None)) is False   # connection error -> propagate

    class BadRequestError(Exception):
        pass
    assert mp._is_unsupported_param(BadRequestError()) is True  # by class name


def test_json_mode_transient_error_propagates_not_swallowed(tmp_path, monkeypatch):
    # A genuine transient error on the json-mode call must NOT be misread as
    # "json unsupported": it propagates (no duplicate call, json_mode not disabled).
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _RateLimit(Exception):
        status_code = 429

    class _Resp:
        def __init__(self):
            self.calls = []

        def create(self, **k):
            self.calls.append(k)
            if "text" in k:
                raise _RateLimit("rate limited")     # transient error on the json-mode attempt
            return _FakeResponse(_responder()("localize", "x", 1, 1))

    class _C:
        def __init__(self):
            self.responses = _Resp()

    c = _C()
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(inp, out, multipass=True, force=True, client=c,
                                       label_policy=POLICY, max_crop_size=None, log=_silent)
    # the transient error propagates -> image errors out, NOT silently retried without json
    assert res["num_images"] == 0 and res["errors"] == 1
    assert len(c.responses.calls) == 1 and "text" in c.responses.calls[0]


def test_json_mode_bad_request_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _BadRequest(Exception):
        status_code = 400

    class _Resp:
        def __init__(self):
            self.calls = []

        def create(self, **k):
            self.calls.append(k)
            if "text" in k:
                raise _BadRequest("model does not support json mode")
            instr = k["instructions"]
            stage = "verify" if "QA checker" in instr else (
                "localize" if "bounding box" in instr.lower() else "corners")
            return _FakeResponse(_responder()(stage, "x", 1, 1))

    class _C:
        def __init__(self):
            self.responses = _Resp()

    c = _C()
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(inp, out, multipass=True, force=True, client=c,
                                       label_policy=POLICY, max_crop_size=None, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 1
    # first stage: a json-mode attempt (400) immediately followed by a fallback without it
    assert "text" in c.responses.calls[0] and "text" not in c.responses.calls[1]
