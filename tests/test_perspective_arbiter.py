"""TDD for the v2.2 API's optional Perspective second-opinion arbiter.

The arbiter:
- Is OFF unless PERSPECTIVE_API_KEY is set
- Only fires when local model uncertainty is high (configurable threshold)
- Augments response with `perspective_score`; never overrides the local verdict
- Never blocks the response on Perspective failure (best-effort)
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _import_arbiter():
    """Lazy import — module under test."""
    from api.arbiter_helper import maybe_perspective_score
    return maybe_perspective_score


def test_arbiter_disabled_when_no_api_key(monkeypatch) -> None:
    monkeypatch.delenv("PERSPECTIVE_API_KEY", raising=False)
    fn = _import_arbiter()
    result = asyncio.run(fn(text="some text", local_confidence=0.55))
    assert result is None


def test_arbiter_skipped_when_local_is_confident(monkeypatch) -> None:
    """If local model is confident (>0.85), don't waste an API call."""
    monkeypatch.setenv("PERSPECTIVE_API_KEY", "fake-key")
    fn = _import_arbiter()
    result = asyncio.run(fn(text="text", local_confidence=0.92))
    assert result is None


def test_arbiter_invokes_perspective_when_uncertain(monkeypatch) -> None:
    monkeypatch.setenv("PERSPECTIVE_API_KEY", "fake-key")
    mock_result = MagicMock(toxicity_score=0.78, severe_toxicity_score=0.12,
                             threat_score=0.45, insult_score=0.55,
                             identity_attack_score=0.10, profanity_score=0.30)
    fake_api = MagicMock()
    fake_api.__aenter__ = AsyncMock(return_value=fake_api)
    fake_api.__aexit__ = AsyncMock(return_value=False)
    fake_api.analyze_comment = AsyncMock(return_value=mock_result)

    with patch("api.arbiter_helper.PerspectiveAPI", return_value=fake_api):
        fn = _import_arbiter()
        result = asyncio.run(fn(text="borderline text", local_confidence=0.55))
    assert result is not None
    assert result["toxicity"] == pytest.approx(0.78)
    assert result["threat"] == pytest.approx(0.45)
    fake_api.analyze_comment.assert_called_once()


def test_arbiter_returns_none_on_perspective_error(monkeypatch) -> None:
    """Network / quota failures must not break the main response."""
    monkeypatch.setenv("PERSPECTIVE_API_KEY", "fake-key")
    fake_api = MagicMock()
    fake_api.__aenter__ = AsyncMock(return_value=fake_api)
    fake_api.__aexit__ = AsyncMock(return_value=False)
    fake_api.analyze_comment = AsyncMock(side_effect=RuntimeError("quota"))
    with patch("api.arbiter_helper.PerspectiveAPI", return_value=fake_api):
        fn = _import_arbiter()
        result = asyncio.run(fn(text="x" * 50, local_confidence=0.5))
    assert result is None


def test_arbiter_threshold_configurable(monkeypatch) -> None:
    """CP_PERSPECTIVE_UNCERTAIN_BELOW lets ops tune the trigger."""
    monkeypatch.setenv("PERSPECTIVE_API_KEY", "fake-key")
    monkeypatch.setenv("CP_PERSPECTIVE_UNCERTAIN_BELOW", "0.6")
    fake_api = MagicMock()
    fake_api.__aenter__ = AsyncMock(return_value=fake_api)
    fake_api.__aexit__ = AsyncMock(return_value=False)
    fake_api.analyze_comment = AsyncMock(return_value=MagicMock(
        toxicity_score=0.5, severe_toxicity_score=0.1, threat_score=0.2,
        insult_score=0.3, identity_attack_score=0.0, profanity_score=0.0,
    ))
    with patch("api.arbiter_helper.PerspectiveAPI", return_value=fake_api):
        fn = _import_arbiter()
        # confidence 0.55 < threshold 0.6 -> should fire
        result = asyncio.run(fn(text="any", local_confidence=0.55))
        assert result is not None
        # confidence 0.65 > threshold 0.6 -> skip
        fake_api.analyze_comment.reset_mock()
        result = asyncio.run(fn(text="any", local_confidence=0.65))
        assert result is None
        fake_api.analyze_comment.assert_not_called()
