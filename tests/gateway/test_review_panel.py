"""
Focused tests for the Multi-Model Review Panel scaffold.

All tests run with the panel OFF by default — no external model calls,
no network, no side effects. They verify the scaffold's safety properties.
"""

import os
import sys
import time
from unittest.mock import patch, MagicMock

import pytest


class FakeProcess:
    def __init__(self, stdout="PASS", stderr="", returncode=0, timeout=False, pid=4242):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode
        self.timeout, self.pid = timeout, pid
        self.killed = False
    def poll(self): return None if self.timeout and not self.killed else self.returncode
    def wait(self, timeout=None): return self.returncode
    def communicate(self, input=None, timeout=None):
        if self.timeout: raise __import__('subprocess').TimeoutExpired('grok', timeout)
        return self.stdout, self.stderr
    def kill(self): self.killed = True

# Ensure the gateway module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gateway.review_panel import (
    GeminiReviewProvider,
    GrokReviewProvider,
    ReviewPacket,
    ReviewPanel,
    ReviewResult,
    ReviewTask,
    TaskState,
    Verdict,
    _grok_enabled,
    _gemini_enabled,
    _review_panel_enabled,
    create_default_panel,
)


# ── Default-OFF regression ─────────────────────────────────────────────────

class TestDefaultOff:
    def test_panel_disabled_by_default(self):
        assert not _review_panel_enabled()

    def test_grok_disabled_by_default(self):
        assert not _grok_enabled()

    def test_gemini_disabled_by_default(self):
        assert not _gemini_enabled()

    def test_disabled_panel_returns_skipped(self):
        panel = ReviewPanel()
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        results = panel.submit_for_review(packet)
        assert len(results) == 1
        assert results[0].verdict == Verdict.SKIPPED

    def test_create_default_panel_empty_when_off(self):
        panel = create_default_panel()
        assert len(panel.providers) == 0

    def test_disabled_panel_no_event_dedup_side_effect(self):
        panel = ReviewPanel()
        # Even when disabled, dedup should work
        assert not panel.is_event_duplicate("e1")
        assert panel.is_event_duplicate("e1")

    def test_disabled_panel_no_active_tasks(self):
        panel = ReviewPanel()
        assert not panel.is_task_active("t1")


# ── Grok provider ──────────────────────────────────────────────────────────

class TestGrokProvider:
    def test_grok_skipped_when_disabled(self):
        provider = GrokReviewProvider()
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert result.verdict == Verdict.SKIPPED
        assert "disabled" in result.summary.lower()

    @patch.dict(os.environ, {"GROK_REVIEWER_ENABLED": "1"})
    def test_grok_cli_not_found(self):
        provider = GrokReviewProvider(cli_path="nonexistent_grok_binary")
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert result.verdict == Verdict.DEGRADED_WITHOUT_GROK
        assert "not found" in result.summary.lower()

    @patch.dict(os.environ, {"GROK_REVIEWER_ENABLED": "1"})
    @patch("gateway.review_panel.subprocess.run")
    @patch("gateway.review_panel.subprocess.Popen")
    def test_grok_timeout(self, mock_popen, mock_taskkill):
        mock_popen.return_value = FakeProcess(timeout=True)
        provider = GrokReviewProvider(timeout_seconds=1)
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert result.verdict == Verdict.DEGRADED_WITHOUT_GROK
        assert "timed out" in result.summary.lower()

    @patch.dict(os.environ, {"GROK_REVIEWER_ENABLED": "1"})
    @patch("gateway.review_panel.subprocess.Popen")
    def test_grok_nonzero_exit(self, mock_popen):
        mock_popen.return_value = FakeProcess(returncode=1, stdout="", stderr="error")
        provider = GrokReviewProvider()
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert result.verdict == Verdict.DEGRADED_WITHOUT_GROK

    @patch.dict(os.environ, {"GROK_REVIEWER_ENABLED": "1"})
    @patch("gateway.review_panel.subprocess.Popen")
    def test_grok_long_output_truncated(self, mock_popen):
        mock_popen.return_value = FakeProcess(stdout="x" * 20000)
        provider = GrokReviewProvider(max_output_chars=100)
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert len(result.raw_output) <= 100

    @patch.dict(os.environ, {"GROK_REVIEWER_ENABLED": "1"})
    @patch("gateway.review_panel.subprocess.Popen")
    def test_grok_secret_env_stripped(self, mock_popen):
        mock_popen.return_value = FakeProcess()
        provider = GrokReviewProvider()
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        provider.review(packet)
        # Check that the env passed to subprocess had secrets removed
        call_args = mock_popen.call_args
        env = call_args.kwargs.get("env", {})
        for key in env:
            assert not any(s in key.upper() for s in ("TOKEN", "SECRET", "KEY", "PASSWORD"))

    @patch.dict(os.environ, {"GROK_REVIEWER_ENABLED": "1"})
    @patch("gateway.review_panel.subprocess.Popen")
    def test_grok_parses_revise(self, mock_popen):
        mock_popen.return_value = FakeProcess(stdout="REVISE\n- blocker: missing error handling\n- risk: SQL injection")
        provider = GrokReviewProvider()
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert result.verdict == Verdict.REVISE

    @patch.dict(os.environ, {"GROK_REVIEWER_ENABLED": "1"})
    @patch("gateway.review_panel.subprocess.Popen")
    def test_grok_parses_blocked(self, mock_popen):
        mock_popen.return_value = FakeProcess(stdout="BLOCKED\n- blocker: security issue")
        provider = GrokReviewProvider()
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert result.verdict == Verdict.BLOCKED

    @patch.dict(os.environ, {"GROK_REVIEWER_ENABLED": "1"})
    @patch("gateway.review_panel.subprocess.Popen")
    def test_grok_parses_pass(self, mock_popen):
        mock_popen.return_value = FakeProcess()
        provider = GrokReviewProvider()
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert result.verdict == Verdict.PASS


# ── Gemini NOT_READY ───────────────────────────────────────────────────────

class TestGeminiProvider:
    def test_gemini_always_not_ready(self):
        provider = GeminiReviewProvider()
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert result.verdict == Verdict.NOT_READY
        assert "not_ready" in result.error

    @patch.dict(os.environ, {"GEMINI_ANALYST_ENABLED": "1"})
    def test_gemini_not_ready_even_when_enabled(self):
        provider = GeminiReviewProvider()
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")
        result = provider.review(packet)
        assert result.verdict == Verdict.NOT_READY


# ── Blind packet ───────────────────────────────────────────────────────────

class TestBlindPacket:
    def test_packet_no_self_evaluation(self):
        packet = ReviewPacket(
            task_id="t1",
            requirements="Build feature X",
            diff_or_artifacts="diff content",
            test_results="all pass",
        )
        prompt = packet.to_prompt()
        # Should contain requirements and diff
        assert "Build feature X" in prompt
        assert "diff content" in prompt
        # Should NOT contain self-evaluation or other AI answers
        assert "self-evaluation" not in prompt.lower()
        assert "my assessment" not in prompt.lower()

    def test_packet_has_review_instructions(self):
        packet = ReviewPacket(
            task_id="t1", requirements="req", diff_or_artifacts="diff"
        )
        prompt = packet.to_prompt()
        assert "Review Instructions" in prompt
        assert "PASS" in prompt
        assert "REVISE" in prompt
        assert "BLOCKED" in prompt

    def test_packet_optional_test_results(self):
        packet = ReviewPacket(
            task_id="t1", requirements="req", diff_or_artifacts="diff"
        )
        prompt = packet.to_prompt()
        assert "Test Results" not in prompt

        packet_with_tests = ReviewPacket(
            task_id="t1", requirements="req", diff_or_artifacts="diff", test_results="3/3 pass"
        )
        prompt_with_tests = packet_with_tests.to_prompt()
        assert "3/3 pass" in prompt_with_tests


# ── Bot approval invalid ───────────────────────────────────────────────────

class TestBotApprovalInvalid:
    def test_panel_does_not_approve(self):
        """Bot verdicts are advisory — never an approval."""
        panel = ReviewPanel()
        # Even a PASS verdict from a provider is not an approval
        result = ReviewResult(provider="bot", verdict=Verdict.PASS)
        # The panel stores results but does not auto-approve
        assert result.verdict == Verdict.PASS
        # Human approval is separate — panel has no approve() method
        assert not hasattr(panel, "approve")

    def test_registered_human_allowlist_only(self):
        panel = ReviewPanel()
        assert panel.is_human_approval_valid("U-HUMAN", {"U-HUMAN"})
        assert not panel.is_human_approval_valid("bot", {"U-HUMAN"})

    def test_secret_masking(self):
        from gateway.review_panel import _mask_secrets
        assert '***' in _mask_secrets('token=abc123')

    def test_windows_process_tree_termination(self):
        from gateway import review_panel
        process = FakeProcess(timeout=True, pid=9876)
        with patch.object(review_panel.os, 'name', 'nt'), patch('gateway.review_panel.subprocess.run') as taskkill:
            review_panel._terminate_process_tree(process)
        taskkill.assert_called_once_with(['taskkill', '/PID', '9876', '/T', '/F'], capture_output=True, text=True, shell=False, timeout=5)


# ── Event ID replay ────────────────────────────────────────────────────────

class TestEventDedup:
    def test_event_id_replay_blocked(self):
        panel = ReviewPanel()
        assert not panel.is_event_duplicate("e1")
        # Replay should be detected
        assert panel.is_event_duplicate("e1")
        assert panel.is_event_duplicate("e1")

    def test_different_events_allowed(self):
        panel = ReviewPanel()
        assert not panel.is_event_duplicate("e1")
        assert not panel.is_event_duplicate("e2")
        assert panel.is_event_duplicate("e1")


# ── Task single-flight ─────────────────────────────────────────────────────

class TestSingleFlight:
    def test_task_not_active_by_default(self):
        panel = ReviewPanel()
        assert not panel.is_task_active("t1")

    def test_task_active_check_does_not_create(self):
        panel = ReviewPanel()
        panel.is_task_active("t1")
        # Checking should not create a task
        assert "t1" not in panel._active_tasks


# ── Max 3 rounds ────────────────────────────────────────────────────────────

class TestMaxRounds:
    @patch.dict(os.environ, {"HERMES_REVIEW_PANEL": "1", "GROK_REVIEWER_ENABLED": "1"})
    @patch("gateway.review_panel.subprocess.Popen")
    def test_max_3_rounds_then_stop(self, mock_popen):
        mock_popen.return_value = FakeProcess()
        panel = ReviewPanel(providers=[GrokReviewProvider()])
        packet = ReviewPacket(task_id="t1", requirements="req", diff_or_artifacts="diff")

        # Round 1
        r1 = panel.submit_for_review(packet)
        assert len(r1) == 1
        assert r1[0].verdict == Verdict.PASS

        # Round 2
        r2 = panel.submit_for_review(packet)
        assert len(r2) == 1

        # Round 3
        r3 = panel.submit_for_review(packet)
        assert len(r3) == 1

        # Round 4 — should be rejected
        r4 = panel.submit_for_review(packet)
        assert len(r4) == 1
        assert r4[0].verdict == Verdict.BLOCKED
        assert "single-flight" in r4[0].summary.lower()


# ── State machine ──────────────────────────────────────────────────────────

class TestStateMachine:
    def test_task_starts_idle(self):
        task = ReviewTask(task_id="t1")
        assert task.state == TaskState.IDLE

    def test_task_advances_to_reviewing(self):
        task = ReviewTask(task_id="t1", state=TaskState.EXECUTING)
        assert task.advance_review()
        assert task.state == TaskState.REVIEWING
        assert task.round == 1

    def test_task_finishes_back_to_executing(self):
        task = ReviewTask(task_id="t1", state=TaskState.EXECUTING)
        task.advance_review()
        task.finish_review()
        assert task.state == TaskState.EXECUTING

    def test_task_awaiting_human_after_max_rounds(self):
        task = ReviewTask(task_id="t1", state=TaskState.EXECUTING, max_rounds=2)
        task.advance_review()  # round 1
        task.finish_review()
        task.advance_review()  # round 2
        task.finish_review()
        assert task.state == TaskState.AWAITING_HUMAN

    def test_task_complete(self):
        task = ReviewTask(task_id="t1")
        task.complete()
        assert task.state == TaskState.DONE

    def test_task_abort(self):
        task = ReviewTask(task_id="t1")
        task.abort()
        assert task.state == TaskState.ABORT
