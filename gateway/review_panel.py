"""
Multi-Model Review Panel — default-OFF scaffold.

Provides a structured review pipeline where GLM (Taeo Operator) can request
independent code/documentation review from external models (Grok, Gemini)
without granting them write access. All providers are review-only: they
return structured verdicts, never execute commands or modify files.

Gated by environment variables (all default OFF):
  HERMES_REVIEW_PANEL=0       — master switch
  GROK_REVIEWER_ENABLED=0     — Grok review provider
  GEMINI_ANALYST_ENABLED=0    — Gemini review provider

When a provider is unavailable or fails, the panel degrades gracefully:
  DEGRADED_WITHOUT_GROK / DEGRADED_WITHOUT_GEMINI
and the GLM operator path continues unaffected.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# ── Env gates (all default OFF) ────────────────────────────────────────────

def _review_panel_enabled() -> bool:
    return os.getenv("HERMES_REVIEW_PANEL", "0") == "1"

def _grok_enabled() -> bool:
    return os.getenv("GROK_REVIEWER_ENABLED", "0") == "1"

def _gemini_enabled() -> bool:
    return os.getenv("GEMINI_ANALYST_ENABLED", "0") == "1"


# ── Structured verdict ─────────────────────────────────────────────────────

class Verdict(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"
    BLOCKED = "BLOCKED"
    DEGRADED_WITHOUT_GROK = "DEGRADED_WITHOUT_GROK"
    DEGRADED_WITHOUT_GEMINI = "DEGRADED_WITHOUT_GEMINI"
    NOT_READY = "NOT_READY"
    SKIPPED = "SKIPPED"


@dataclass
class ReviewResult:
    provider: str
    verdict: Verdict
    summary: str = ""
    blockers: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    raw_output: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "verdict": self.verdict.value,
            "summary": self.summary,
            "blockers": self.blockers,
            "risks": self.risks,
            "questions": self.questions,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


# ── Blind review packet ────────────────────────────────────────────────────

@dataclass
class ReviewPacket:
    """Blind packet sent to reviewers — no self-evaluation, no other AI answers."""
    task_id: str
    requirements: str
    diff_or_artifacts: str
    test_results: str = ""

    def to_prompt(self) -> str:
        parts = [
            f"Task: {self.task_id}",
            f"\n## Requirements\n{self.requirements}",
            f"\n## Diff / Artifacts\n{self.diff_or_artifacts}",
        ]
        if self.test_results:
            parts.append(f"\n## Test Results\n{self.test_results}")
        parts.append(
            "\n## Review Instructions\n"
            "Review ONLY for correctness, security, regressions, and missing "
            "requirements. Do NOT execute, modify, or approve. Return a "
            "structured verdict: PASS, REVISE, or BLOCKED, with blockers, "
            "risks, and questions as bullet lists."
        )
        return "\n".join(parts)


# ── ReviewProvider protocol ─────────────────────────────────────────────────

class ReviewProvider(Protocol):
    name: str

    def review(self, packet: ReviewPacket) -> ReviewResult: ...


# ── GrokReviewProvider ──────────────────────────────────────────────────────

class GrokReviewProvider:
    """Grok CLI-based review provider (non-interactive, isolated)."""

    name = "Grok Reviewer"

    def __init__(
        self,
        cli_path: str = "grok",
        timeout_seconds: int = 45,
        max_output_chars: int = 8000,
    ):
        self.cli_path = cli_path
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def review(self, packet: ReviewPacket) -> ReviewResult:
        if not _grok_enabled():
            return ReviewResult(
                provider=self.name,
                verdict=Verdict.SKIPPED,
                summary="Grok reviewer disabled (GROK_REVIEWER_ENABLED=0)",
            )

        prompt = packet.to_prompt()
        start = time.monotonic()
        try:
            result = subprocess.run(
                [self.cli_path, "-p", prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
                env=self._sanitized_env(),
            )
            elapsed = time.monotonic() - start
            output = (result.stdout or "")[: self.max_output_chars]
            if result.returncode != 0:
                return ReviewResult(
                    provider=self.name,
                    verdict=Verdict.DEGRADED_WITHOUT_GROK,
                    summary=f"Grok CLI returned exit code {result.returncode}",
                    raw_output=output,
                    elapsed_seconds=elapsed,
                    error=(result.stderr or "")[:500],
                )
            return self._parse_output(output, elapsed)
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            return ReviewResult(
                provider=self.name,
                verdict=Verdict.DEGRADED_WITHOUT_GROK,
                summary=f"Grok CLI timed out after {self.timeout_seconds}s",
                elapsed_seconds=elapsed,
                error="timeout",
            )
        except FileNotFoundError:
            return ReviewResult(
                provider=self.name,
                verdict=Verdict.DEGRADED_WITHOUT_GROK,
                summary="Grok CLI not found",
                error="cli_not_found",
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return ReviewResult(
                provider=self.name,
                verdict=Verdict.DEGRADED_WITHOUT_GROK,
                summary=f"Grok review failed: {exc}",
                elapsed_seconds=elapsed,
                error=str(exc)[:500],
            )

    def _sanitized_env(self) -> Dict[str, str]:
        """Return env with secrets masked — no tokens passed to the subprocess."""
        env = dict(os.environ)
        for key in list(env.keys()):
            if any(s in key.upper() for s in ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL")):
                env.pop(key, None)
        return env

    def _parse_output(self, output: str, elapsed: float) -> ReviewResult:
        """Parse Grok's free-text output into a structured verdict."""
        text = output.strip()
        verdict = Verdict.PASS
        if "REVISE" in text.upper():
            verdict = Verdict.REVISE
        elif "BLOCKED" in text.upper():
            verdict = Verdict.BLOCKED

        blockers: List[str] = []
        risks: List[str] = []
        for line in text.split("\n"):
            stripped = line.lstrip("- *").strip()
            if line.strip().startswith(("-", "*")) and "block" in line.lower():
                blockers.append(stripped)
            elif line.strip().startswith(("-", "*")) and "risk" in line.lower():
                risks.append(stripped)

        return ReviewResult(
            provider=self.name,
            verdict=verdict,
            summary=text[:500],
            blockers=blockers,
            risks=risks,
            raw_output=text,
            elapsed_seconds=elapsed,
        )


# ── GeminiReviewProvider (NOT_READY stub) ──────────────────────────────────

class GeminiReviewProvider:
    """Gemini review provider — NOT_READY stub (no API key configured)."""

    name = "Gemini Analyst"

    def review(self, packet: ReviewPacket) -> ReviewResult:
        return ReviewResult(
            provider=self.name,
            verdict=Verdict.NOT_READY,
            summary="Gemini NOT_READY: no API key configured, gcloud auth broken",
            error="not_ready",
        )


# ── Task state machine ─────────────────────────────────────────────────────

class TaskState(str, Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    AWAITING_HUMAN = "awaiting_human"
    DONE = "done"
    ABORT = "abort"


@dataclass
class ReviewTask:
    task_id: str
    state: TaskState = TaskState.IDLE
    round: int = 0
    max_rounds: int = 3
    results: List[ReviewResult] = field(default_factory=list)

    def can_review(self) -> bool:
        return self.state == TaskState.EXECUTING and self.round < self.max_rounds

    def advance_review(self) -> bool:
        if self.round >= self.max_rounds:
            self.state = TaskState.AWAITING_HUMAN
            return False
        self.round += 1
        self.state = TaskState.REVIEWING
        return True

    def finish_review(self):
        if self.round >= self.max_rounds:
            self.state = TaskState.AWAITING_HUMAN
        else:
            self.state = TaskState.EXECUTING

    def complete(self):
        self.state = TaskState.DONE

    def abort(self):
        self.state = TaskState.ABORT


# ── Review Panel orchestrator ──────────────────────────────────────────────

class ReviewPanel:
    """Orchestrates multi-model review. Default OFF — no behavior change."""

    def __init__(self, providers: Optional[List[ReviewProvider]] = None):
        self.providers = providers or []
        self._event_ids_seen: set[str] = set()
        self._active_tasks: Dict[str, ReviewTask] = {}

    def is_enabled(self) -> bool:
        return _review_panel_enabled()

    def is_event_duplicate(self, event_id: str) -> bool:
        if event_id in self._event_ids_seen:
            return True
        self._event_ids_seen.add(event_id)
        return False

    def is_task_active(self, task_id: str) -> bool:
        task = self._active_tasks.get(task_id)
        if task and task.state in (TaskState.EXECUTING, TaskState.REVIEWING):
            return True
        return False

    def submit_for_review(self, packet: ReviewPacket) -> List[ReviewResult]:
        if not self.is_enabled():
            return [
                ReviewResult(
                    provider="panel",
                    verdict=Verdict.SKIPPED,
                    summary="Review panel disabled (HERMES_REVIEW_PANEL=0)",
                )
            ]

        task = self._active_tasks.setdefault(
            packet.task_id, ReviewTask(task_id=packet.task_id)
        )
        if not task.advance_review():
            logger.warning(
                "[ReviewPanel] Max rounds (%d) reached for task %s — awaiting human",
                task.max_rounds,
                packet.task_id,
            )
            return [
                ReviewResult(
                    provider="panel",
                    verdict=Verdict.SKIPPED,
                    summary=f"Max rounds ({task.max_rounds}) reached",
                )
            ]

        results: List[ReviewResult] = []
        for provider in self.providers:
            try:
                result = provider.review(packet)
            except Exception as exc:
                result = ReviewResult(
                    provider=getattr(provider, "name", "unknown"),
                    verdict=Verdict.SKIPPED,
                    summary=f"Provider error: {exc}",
                    error=str(exc)[:500],
                )
            results.append(result)
            logger.info(
                "[ReviewPanel] %s verdict=%s for task %s round %d",
                result.provider,
                result.verdict.value,
                packet.task_id,
                task.round,
            )

        task.finish_review()
        task.results.extend(results)
        return results


# ── Factory ────────────────────────────────────────────────────────────────

def create_default_panel() -> ReviewPanel:
    """Create a panel with Grok and Gemini providers (both default OFF)."""
    providers: List[ReviewProvider] = []
    if _grok_enabled():
        providers.append(GrokReviewProvider())
    if _gemini_enabled():
        providers.append(GeminiReviewProvider())
    return ReviewPanel(providers=providers)