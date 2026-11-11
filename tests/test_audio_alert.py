import time

from adas.core.audio_alert import AudioAlert


def test_disabled_alert_does_not_play():
    audio = AudioAlert(enabled=False)
    audio.trigger()


def test_rate_limiting():
    audio = AudioAlert(enabled=True, repeat_seconds=2.0)
    audio._last_played = time.monotonic()
    # Trigger should be rate-limited and skip
    audio.trigger()


def test_repeat_seconds_default():
    audio = AudioAlert()
    assert audio.repeat_seconds == 3.0


def test_custom_repeat_seconds():
    audio = AudioAlert(repeat_seconds=5.0)
    assert audio.repeat_seconds == 5.0
