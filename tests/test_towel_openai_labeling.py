"""Tests for the OpenAI Vision towel labeler (second opinion alongside Claude).

NO network and NO OpenAI SDK: the Responses API is exercised through two seams —
a ``responder`` (canned JSON text) and a FAKE ``client`` that mimics
``client.responses.create(...) -> response.output_text`` (a mocked OpenAI response).
Images are PNGs from the pure-Python codec, so no OpenCV/Pillow is needed.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from terafold.towel import claude_pseudolabel as cl
from terafold.towel import openai_pseudolabel as op
from terafold.towel.dataset import load_label, load_manifest
from terafold.vision.imageio import imwrite

POLICY = "striped_towel_visible_outer_corners"


def _silent(*_a) -> None:
    pass


def _img_folder(tmp_path, n=3, W=100, H=100):
    d = tmp_path / "full"
    d.mkdir()
    for i in range(n):
        img = np.full((H, W, 3), 40, np.uint8)
        img[12:88, 10:90] = 200
        imwrite(str(d / f"IMG_{i}.png"), img)
    return str(d)


def _label_json(corners=None, state="flat_unfolded", is_target=True, conf=0.9,
                reject_reason=None, risks=None, visible=None):
    return json.dumps({
        "is_target_towel_visible": is_target,
        "reject_reason": reject_reason,
        "state": state,
        "corners": corners or {"tl": [10, 12], "tr": [88, 10], "br": [90, 86], "bl": [12, 88]},
        "visible": visible or {"tl": True, "tr": True, "br": True, "bl": True},
        "confidence": conf,
        "risks": risks or [],
    })


# ---- a minimal fake Responses client (the "mocked OpenAI response") ----


class _FakeUsage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class _FakeResponse:
    def __init__(self, text, usage=(7, 9)):
        self.output_text = text
        self.usage = _FakeUsage(*usage)


class _FakeResponses:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._text)


class _FakeClient:
    """Stands in for openai.OpenAI(): exposes .responses.create()."""

    def __init__(self, text):
        self.responses = _FakeResponses(text)


# ============================ command + prompt ============================


def test_command_registered():
    from terafold.cli import app

    names = {c.name or c.callback.__name__.replace("_", "-") for c in app.registered_commands}
    assert "openai-label-towel-folder" in names


def test_defaults():
    assert op.DEFAULT_OPENAI_MODEL == "gpt-5.5"
    assert op.OPENAI_LABEL_POLICY == POLICY


def test_prompt_targets_striped_towel_and_ignores_distractors():
    p = op.OPENAI_USER_PROMPT
    assert "beige/white striped towel" in p
    for token in ("bed sheet", "mattress cover", "pillow", "blanket", "headboard",
                  "floor", "body part"):
        assert token in p


def test_prompt_has_corner_rule_and_json_fields():
    p = op.OPENAI_USER_PROMPT
    assert "Label the 4 visible outer corners of the towel's CURRENT visible outer shape." in p
    assert "Do NOT label the image border, crop border, bed rectangle, or bounding box." in p
    assert "If folded, label the 4 outer corners of the visible folded rectangle." in p
    assert "If partially folded, label the 4 outer corners of the visible towel silhouette." in p
    assert "visual intersection of the two outer towel edges" in p
    for field in ("is_target_towel_visible", "reject_reason", "state", "corners",
                  "visible", "confidence", "risks"):
        assert field in p


def test_policy_registered_in_gates():
    assert POLICY in cl.STRIPED_TOWEL_POLICIES
    assert POLICY in cl.STRICT_OUTER_POLICIES


def test_system_prompt_geometry_only():
    assert "STRICT JSON" in op.OPENAI_SYSTEM_PROMPT
    assert "NEVER output robot commands" in op.OPENAI_SYSTEM_PROMPT


# ============================ response helpers ============================


def test_output_text_from_object_and_dict_and_walk():
    assert op._openai_output_text(_FakeResponse("hello")) == "hello"
    assert op._openai_output_text({"output_text": "hi"}) == "hi"
    walk = {"output": [{"content": [{"type": "output_text", "text": "deep"}]}]}
    assert op._openai_output_text(walk) == "deep"


def test_usage_extraction():
    assert op._openai_usage(_FakeResponse("x", usage=(5, 6))) == {"input_tokens": 5, "output_tokens": 6}
    assert op._openai_usage({"usage": {"input_tokens": 3}}) == {"input_tokens": 3, "output_tokens": 0}
    assert op._openai_usage({}) == {"input_tokens": 0, "output_tokens": 0}


# ============================ transport ============================


def test_have_transport_logic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert op.OpenAITowelLabeler().have_transport() is False
    assert op.OpenAITowelLabeler(responder=lambda p, w, h: "{}").have_transport() is True
    assert op.OpenAITowelLabeler(client=object()).have_transport() is True
    assert op.OpenAITowelLabeler(api_key="sk-test").have_transport() is True


def test_call_api_builds_responses_request(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    img = str(tmp_path / "a.png")
    imwrite(img, np.full((40, 40, 3), 150, np.uint8))
    fake = _FakeClient(_label_json())
    labeler = op.OpenAITowelLabeler(client=fake, model="gpt-5.5")
    out = labeler.raw_response(img, 40, 40)
    assert json.loads(out)["state"] == "flat_unfolded"
    kw = fake.responses.calls[0]
    assert kw["model"] == "gpt-5.5"
    assert kw["instructions"] == op.OPENAI_SYSTEM_PROMPT
    content = kw["input"][0]["content"]
    assert content[0]["type"] == "input_text" and "{w}" not in content[0]["text"]  # formatted
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert labeler.last_usage == {"input_tokens": 7, "output_tokens": 9}


# ============================ folder labeling (end to end) ============================


def test_label_folder_via_responder(tmp_path):
    inp = _img_folder(tmp_path, n=3)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(inp, out, responder=lambda p, w, h: _label_json(),
                                       label_policy=POLICY, force=True, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 3
    man = load_manifest(out)
    assert man["backend"] == "openai" and man["source"] == "openai_pseudolabel"
    assert man["model"] == op.DEFAULT_OPENAI_MODEL
    s0 = man["samples"][0]
    lab = load_label(out, s0)
    assert lab["source"] == "openai_pseudolabel" and lab["label_method"] == "openai_vision"
    assert lab["label_backend"] == "openai" and lab["pseudolabel"] is True
    # overlay rendered in the existing style
    assert s0["overlay"] and os.path.exists(os.path.join(out, s0["overlay"]))


def test_label_folder_via_mocked_client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    inp = _img_folder(tmp_path, n=2)
    out = str(tmp_path / "out")
    fake = _FakeClient(_label_json())
    res = op.openai_label_towel_folder(inp, out, client=fake, label_policy=POLICY,
                                       force=True, log=_silent)
    assert res["status"] == "ok" and res["num_images"] == 2
    # the Responses API was actually invoked once per image
    assert len(fake.responses.calls) == 2
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["corners"]["tl"] == [10.0, 12.0]


def test_max_images_caps_the_run(tmp_path):
    inp = _img_folder(tmp_path, n=5)
    out = str(tmp_path / "out")
    res = op.openai_label_towel_folder(inp, out, responder=lambda p, w, h: _label_json(),
                                       max_images=3, force=True, log=_silent)
    assert res["num_images"] == 3


def test_reject_rule_when_not_target(tmp_path):
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    op.openai_label_towel_folder(
        inp, out, responder=lambda p, w, h: _label_json(is_target=False, state="flat_unfolded",
                                                        reject_reason="only the bed sheet is visible"),
        label_policy=POLICY, force=True, log=_silent)
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert "not_striped_towel" in lab["risk_reasons"]
    assert lab["state"] == "not_towel"
    assert lab["is_target"] is False
    assert any(r.startswith("reject_reason:") for r in lab["risk_reasons"])
    assert lab["usable_for_training"] is False
    assert lab["approval_status"] == "review_needed"


def test_risks_alias_becomes_risk_reasons(tmp_path):
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    op.openai_label_towel_folder(
        inp, out, responder=lambda p, w, h: _label_json(risks=["blurry", "shadow"]),
        label_policy=POLICY, force=True, log=_silent)
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert "blurry" in lab["risk_reasons"] and "shadow" in lab["risk_reasons"]


def test_no_api_key_graceful(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    inp = _img_folder(tmp_path, n=1)
    res = op.openai_label_towel_folder(inp, str(tmp_path / "o"), log=_silent)
    assert res["status"] == "no_api_key" and res["instructions"]
    # the OpenAI key hint is shown (never a key value)
    assert any("OPENAI_API_KEY" in line for line in res["instructions"])


def test_unknown_label_policy_errors(tmp_path):
    # A non-striped policy must HARD-FAIL (else the OpenAI prompt asks is_target but the
    # reject gate is silently disabled). The Claude command validates similarly.
    inp = _img_folder(tmp_path, n=1)
    res = op.openai_label_towel_folder(inp, str(tmp_path / "o"),
                                       responder=lambda p, w, h: _label_json(),
                                       label_policy="default", log=_silent)
    assert res["status"] == "error" and "label_policies" in res
    assert POLICY in res["label_policies"]


def test_stringified_false_triggers_reject(tmp_path):
    # An off-spec stringified boolean ("false") must NOT be coerced to True and bypass
    # the striped-towel reject rule.
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "out")
    raw = json.dumps({
        "is_target_towel_visible": "false", "reject_reason": None, "state": "flat_unfolded",
        "corners": {"tl": [10, 12], "tr": [88, 10], "br": [90, 86], "bl": [12, 88]},
        "visible": {"tl": True, "tr": True, "br": True, "bl": True},
        "confidence": 0.95, "risks": [],
    })
    op.openai_label_towel_folder(inp, out, responder=lambda p, w, h: raw,
                                 label_policy=POLICY, force=True, log=_silent)
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["is_target"] is False
    assert "not_striped_towel" in lab["risk_reasons"]
    assert lab["state"] == "not_towel" and lab["approval_status"] == "review_needed"


def test_as_bool_robust_coercion():
    assert cl._as_bool(None, default=True) is True
    assert cl._as_bool(None, default=False) is False
    assert cl._as_bool(True) is True and cl._as_bool(False) is False
    for falsey in ("false", "False", "no", "0", "off", "", "null"):
        assert cl._as_bool(falsey) is False, falsey
    for truthy in ("true", "yes", "1"):
        assert cl._as_bool(truthy) is True, truthy


def test_claude_label_source_unchanged(tmp_path):
    # Regression: the Claude backend must still stamp source=claude_pseudolabel.
    inp = _img_folder(tmp_path, n=1)
    out = str(tmp_path / "c")
    cl.claude_label_towel_folder(
        inp, out, responder=lambda p, w, h: _label_json(),
        label_policy="outer_visible_corners_any_state", force=True, log=_silent)
    lab = load_label(out, load_manifest(out)["samples"][0])
    assert lab["source"] == "claude_pseudolabel" and lab["label_backend"] == "claude"
    assert lab["claude_model"] == cl.DEFAULT_MODEL
