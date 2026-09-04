"""Tests for service configuration.

Two things are worth pinning here. The port, because it is the one setting an
operator reaches for first and the one most likely to be wrong in a deployment.
And `.env.example`, because a configuration template that has drifted from the
code is worse than no template at all: it documents settings that do nothing
and hides the ones that do.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.settings import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


@pytest.fixture
def no_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every DOC_ variable from the environment.

    Real environment variables outrank a dotenv file, and the test suite sets
    some of its own. Without clearing them, a test that means to read a file
    would be reading the shell.
    """
    import os

    for name in [key for key in os.environ if key.startswith("DOC_")]:
        monkeypatch.delenv(name, raising=False)


class TestPort:
    def test_it_defaults_to_the_services_port(self, no_ambient_config: None) -> None:
        assert Settings(_env_file=None).port == 8008

    def test_it_can_be_overridden_from_the_environment(
        self, no_ambient_config: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOC_PORT", "9100")
        assert Settings(_env_file=None).port == 9100

    @pytest.mark.parametrize("bad", [0, -1, 65536, 70000])
    def test_it_refuses_a_port_that_cannot_be_bound(
        self, no_ambient_config: None, bad: int
    ) -> None:
        """A service that starts and then cannot bind is worse than one that
        refuses the configuration, which is the stance taken everywhere else."""
        with pytest.raises(ValidationError):
            Settings(_env_file=None, port=bad)


class TestEnvExample:
    """The template must describe the code, not a past version of it."""

    def _declared(self) -> dict[str, str]:
        """Every active `KEY=VALUE` in the template, comments excluded."""
        pairs: dict[str, str] = {}
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            pairs[key.strip()] = value.strip()
        return pairs

    def test_the_template_exists(self) -> None:
        assert ENV_EXAMPLE.is_file(), f"{ENV_EXAMPLE} is missing"

    def test_every_declared_key_is_a_real_setting(self) -> None:
        """`extra="ignore"` means a renamed or deleted setting leaves a line in
        the template that silently does nothing. This is what catches it."""
        fields = set(Settings.model_fields)
        unknown = {
            key
            for key in self._declared()
            if not key.startswith("DOC_") or key[len("DOC_"):].lower() not in fields
        }
        assert not unknown, f"template names settings that do not exist: {sorted(unknown)}"

    def test_the_port_is_documented(self) -> None:
        assert self._declared().get("DOC_PORT") == "8008"

    def test_the_template_matches_the_code_defaults(
        self, no_ambient_config: None
    ) -> None:
        """The file promises that copying it changes nothing. A default that
        moved in settings.py without the template following would break that
        promise quietly, and the copy would silently reconfigure the service."""
        from_file = Settings(_env_file=ENV_EXAMPLE).model_dump()
        from_code = Settings(_env_file=None).model_dump()
        differing = {
            name: (from_code[name], from_file[name])
            for name in from_code
            if from_code[name] != from_file[name]
        }
        assert not differing, f"template has drifted from the defaults: {differing}"

    def test_the_retired_backend_is_not_offered_as_configuration(self) -> None:
        """A template is where someone looks for a value to paste. It must not
        suggest a backend the service refuses."""
        assert self._declared().get("DOC_BACKEND") == "docling"
