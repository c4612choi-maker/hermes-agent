"""Default-OFF lifecycle holder for the optional Review Panel."""

from __future__ import annotations

import threading

from gateway.review_panel import ReviewPanel, _review_panel_enabled, create_default_panel


class ReviewPanelRuntime:
    """One process-local Review Panel dependency with no background work."""

    def __init__(self) -> None:
        # Provider construction is gated by the master flag, not provider flags.
        self.panel = create_default_panel() if _review_panel_enabled() else ReviewPanel()


_runtime: ReviewPanelRuntime | None = None
_runtime_lock = threading.Lock()


def get_review_panel_runtime() -> ReviewPanelRuntime:
    """Return the process-local runtime without starting a provider."""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = ReviewPanelRuntime()
        return _runtime
