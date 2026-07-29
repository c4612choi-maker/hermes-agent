"""Focused default-OFF integration checks for Gateway Review Panel wiring."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.review_panel import ReviewPacket, Verdict
import gateway.review_panel_runtime as runtime


@pytest.fixture(autouse=True)
def reset_runtime(monkeypatch):
    for name in ("HERMES_REVIEW_PANEL", "GROK_REVIEWER_ENABLED", "GEMINI_ANALYST_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "_runtime", None)
    yield
    monkeypatch.setattr(runtime, "_runtime", None)


def test_unset_flags_create_default_off_runtime():
    holder = runtime.get_review_panel_runtime()
    assert holder.panel.providers == []
    assert holder.panel.submit_for_review(ReviewPacket("t", "r", "d"))[0].verdict is Verdict.SKIPPED


def test_explicit_zero_flags_remain_off(monkeypatch):
    monkeypatch.setenv("HERMES_REVIEW_PANEL", "0")
    monkeypatch.setenv("GROK_REVIEWER_ENABLED", "0")
    monkeypatch.setenv("GEMINI_ANALYST_ENABLED", "0")
    assert runtime.get_review_panel_runtime().panel.providers == []


@pytest.mark.parametrize("value", ["", " true ", "True", "yes", "2", "enabled"])
def test_invalid_master_values_fail_closed(monkeypatch, value):
    monkeypatch.setenv("HERMES_REVIEW_PANEL", value)
    assert runtime.get_review_panel_runtime().panel.providers == []


def test_master_off_blocks_provider_construction_even_if_provider_flags_are_on(monkeypatch):
    monkeypatch.setenv("HERMES_REVIEW_PANEL", "0")
    monkeypatch.setenv("GROK_REVIEWER_ENABLED", "1")
    monkeypatch.setenv("GEMINI_ANALYST_ENABLED", "1")
    with patch.object(runtime, "create_default_panel", side_effect=AssertionError("must not construct providers")):
        assert runtime.get_review_panel_runtime().panel.providers == []


def test_runtime_is_process_singleton_without_new_threads_or_processes():
    before = {thread.ident for thread in threading.enumerate()}
    with patch.object(subprocess, "Popen") as popen, patch.object(subprocess, "run") as run:
        first = runtime.get_review_panel_runtime()
        second = runtime.get_review_panel_runtime()
    assert first is second
    assert {thread.ident for thread in threading.enumerate()} == before
    popen.assert_not_called()
    run.assert_not_called()


def test_gateway_constructor_has_runtime_dependency_wiring():
    source = (Path(__file__).parents[2] / "gateway" / "run.py").read_text(encoding="utf-8")
    assert "from gateway.review_panel_runtime import get_review_panel_runtime" in source
    assert "self.review_panel_runtime = get_review_panel_runtime()" in source
    assert "self.review_panel = self.review_panel_runtime.panel" in source


def test_runtime_module_has_no_import_time_side_effect_calls():
    source = (Path(__file__).parents[2] / "gateway" / "review_panel_runtime.py").read_text(encoding="utf-8")
    assert "subprocess" not in source
    assert "socket" not in source
    assert "Thread(" not in source
    assert "slack" not in source.lower()


def test_runtime_does_not_expose_environment_values_to_logs(monkeypatch, caplog):
    secret = "runtime-secret-value"
    monkeypatch.setenv("HERMES_REVIEW_PANEL", "0")
    monkeypatch.setenv("RUNTIME_TEST_TOKEN", secret)
    runtime.get_review_panel_runtime()
    assert secret not in caplog.text
