import pytest

from deep_eye import config


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch):
    """configure() changes module globals; undo it after every test."""
    for name in ("_trace_dir", "_max_chars", "_max_items", "_redact", "_extra_redact_keys",
                 "_redact_fn", "_max_age_days", "_enabled", "_sample_rate", "_sync_writes"):
        monkeypatch.setattr(config, name, getattr(config, name))
