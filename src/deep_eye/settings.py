"""User preferences (currently just the theme), stored per user, not per project.

Traces live next to the code that produced them, but a color preference should
follow you everywhere, so it goes in your home directory.
"""

import json
import os
import sys
from pathlib import Path


def settings_path() -> Path:
    """Per-user settings file, in the usual place for each OS.

    Windows: %APPDATA%/deep-eye/settings.json
    macOS:   ~/Library/Application Support/deep-eye/settings.json
    Linux:   $XDG_CONFIG_HOME or ~/.config, then deep-eye/settings.json
    """
    home = Path.home()
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
    return base / "deep-eye" / "settings.json"


def load_settings() -> dict:
    """Read settings; a missing or broken file just means 'defaults'."""
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_setting(key: str, value: str) -> None:
    """Update one key, keeping the rest. Failing to save is never fatal."""
    try:
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**load_settings(), key: value}, indent=2), encoding="utf-8")
    except OSError:
        pass
