"""Audible alert manager supporting multi-level alerts (overspeed alarm & caution chime)."""
from __future__ import annotations

import platform
import subprocess
import sys
import threading
import time


class AudioAlert:
    """Rate-limited audible alert for desktop demos supporting 'alarm' and 'caution' sounds."""

    def __init__(self, enabled: bool = True, repeat_seconds: float = 3.0) -> None:
        self.enabled = enabled
        self.repeat_seconds = repeat_seconds
        self._last_played = 0.0
        self._last_severity = "none"

    def trigger(self, severity: str = "alarm") -> None:
        """Trigger an audible alert.

        severity:
          - 'alarm': Urgent 2-tone alarm for overspeed (Vượt tốc).
          - 'caution': Single gentle chime for approaching speed limit (Chú ý).
        """
        if not self.enabled:
            return
        now = time.monotonic()
        # Alarm can preempt caution; caution requires a slightly longer cooldown
        cooldown = self.repeat_seconds if severity == "alarm" else max(self.repeat_seconds, 4.5)
        if now - self._last_played < cooldown and severity == self._last_severity:
            return

        self._last_played = now
        self._last_severity = severity
        threading.Thread(target=self._play, args=(severity,), daemon=True).start()

    @staticmethod
    def _play(severity: str = "alarm") -> None:
        try:
            if platform.system() == "Windows":
                import winsound

                if severity == "alarm":
                    # Urgent alarm for overspeed
                    winsound.Beep(950, 250)
                    winsound.Beep(750, 250)
                elif severity == "caution":
                    # Gentle reminder chime for caution
                    winsound.Beep(650, 180)
            elif platform.system() == "Darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"], check=False, timeout=2)
            else:
                # Terminal bell is intentionally best-effort; some terminals mute it.
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass
