"""
ksd/listener.py — Keyboard event logger.

Captures raw key-press events (character + high-resolution timestamp) into a
shared queue so that other modules can consume them without blocking the
pynput listener thread.
"""

import time
import queue
import threading
import logging
from pynput import keyboard

log = logging.getLogger(__name__)


class KeystrokeListener:
    """
    Starts a pynput keyboard listener in a background thread.

    Events are pushed onto ``event_queue`` as ``(key_char: str,
    timestamp: float)`` tuples, where timestamp is from
    ``time.perf_counter()``.

    Parameters
    ----------
    event_queue : queue.Queue
        Shared queue consumed by the feature-extraction layer.
    stop_event : threading.Event
        When set, the listener stops gracefully.
    """

    def __init__(self,
                 event_queue: queue.Queue,
                 stop_event: threading.Event):
        self._q     = event_queue
        self._stop  = stop_event
        self._listener: keyboard.Listener | None = None

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _on_press(self, key):
        """Push a press-event into the queue."""
        if self._stop.is_set():
            return False          # detach listener

        try:
            ch = key.char if (hasattr(key, "char") and key.char) else str(key)
        except Exception:
            ch = str(key)

        self._q.put(("press", ch, time.perf_counter()))

    def _on_release(self, key):
        """Push a release-event into the queue (used for dwell-time calc)."""
        if self._stop.is_set():
            return False

        try:
            ch = key.char if (hasattr(key, "char") and key.char) else str(key)
        except Exception:
            ch = str(key)

        self._q.put(("release", ch, time.perf_counter()))

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self):
        """Start the pynput listener (non-blocking)."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        log.info("KeystrokeListener started.")

    def stop(self):
        """Stop the listener cleanly."""
        self._stop.set()
        if self._listener:
            self._listener.stop()
        log.info("KeystrokeListener stopped.")
