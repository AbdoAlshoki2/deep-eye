"""What every adapter promises: no framework import unless asked for, a clear error when the
framework is missing, and no exception ever reaching the traced program."""

import subprocess
import sys
import textwrap

import pytest

ADAPTERS = {  # module: (framework import names, extra)
    "langchain": (["langchain_core", "langchain"], "langchain"),
    "openai_agents": (["agents", "openai"], "openai-agents"),
    "otel": (["opentelemetry"], "otel"),
}
FRAMEWORKS = sorted({name for names, _ in ADAPTERS.values() for name in names})

# Runs in a fresh interpreter in which the frameworks can't be imported, as in a venv
# that has none of them installed.
BLOCKER = textwrap.dedent(f"""
    import sys
    BLOCKED = {FRAMEWORKS!r}

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)

    sys.meta_path.insert(0, Blocker())
""")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCKER + textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=60)


def test_import_deep_eye_works_without_any_framework():
    result = _run(f"""
        import deep_eye, deep_eye.cli, deep_eye.export, deep_eye.schema, deep_eye.integrations
        with deep_eye.configure(enabled=False):
            deep_eye.trace(lambda: None)()
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in {FRAMEWORKS!r})
        print("loaded:", loaded)
    """)
    assert result.returncode == 0, result.stderr
    assert "loaded: []" in result.stdout


@pytest.mark.parametrize("adapter", sorted(ADAPTERS))
def test_missing_framework_names_the_extra_to_install(adapter):
    extra = ADAPTERS[adapter][1]
    result = _run(f"""
        try:
            import deep_eye.integrations.{adapter}
        except ImportError as exc:
            print("message:", exc)
    """)
    assert result.returncode == 0, result.stderr
    assert f'pip install "deep-eye[{extra}]"' in result.stdout


def test_every_adapter_has_its_own_extra():
    from pathlib import Path

    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    assert {extra for _, extra in ADAPTERS.values()} <= set(extras)
