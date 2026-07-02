"""Guard: every `terafold <command>` referenced in the docs is a real command.

Keeps the operator documentation honest — a renamed/removed CLI command can't
silently leave a wrong example in the README or docs/.
"""

from __future__ import annotations

import os
import re

import pytest

from terafold.cli import app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `python3 -m terafold <cmd>` or `terafold <cmd>` (cmd is kebab-case).
_PATTERN = re.compile(r"terafold\s+([a-z][a-z0-9-]+)")

# Tokens that follow "terafold" in prose but are not commands.
_NON_COMMANDS = {"robot", "stack", "is", "does", "live", "dry", "real", "platform"}


def _registered() -> set[str]:
    names = set()
    for c in app.registered_commands:
        names.add(c.name or c.callback.__name__.replace("_", "-"))
    return names


def _doc_files() -> list[str]:
    files = [os.path.join(REPO, "README.md")]
    docs = os.path.join(REPO, "docs")
    for fn in sorted(os.listdir(docs)):
        if fn.endswith(".md"):
            files.append(os.path.join(docs, fn))
    return files


@pytest.mark.parametrize("path", _doc_files())
def test_doc_command_examples_are_real(path):
    registered = _registered()
    with open(path) as f:
        text = f.read()
    referenced = {m.group(1) for m in _PATTERN.finditer(text)}
    referenced -= _NON_COMMANDS
    bogus = sorted(c for c in referenced if c not in registered)
    assert not bogus, f"{os.path.basename(path)} references unknown commands: {bogus}"


def test_key_platform_commands_exist():
    registered = _registered()
    for cmd in ("robot-status", "robot-model-status", "servo-scan", "characterize-servos",
                "calibrate-table", "validate-table-calibration", "calibrate-camera",
                "robot-touch-calibration", "fit-robot-table-transform", "real-image-ghost-fold",
                "episode-summary", "eval-run", "sim-real-ghost-fold", "map-servo-joints"):
        assert cmd in registered, cmd
