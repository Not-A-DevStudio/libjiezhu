"""Shared fixtures for jiezhu tests.

We mock both the `openai` and `anthropic` packages so that tests run
without any real SDK installation or API key.
"""
from __future__ import annotations

import io
import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helper: build a minimal fake module hierarchy and register it in sys.modules
# ---------------------------------------------------------------------------

def _make_module(name: str, parent: types.ModuleType | None = None) -> types.ModuleType:
    mod = types.ModuleType(name)
    if parent is not None:
        setattr(parent, name.rsplit(".", 1)[-1], mod)
    return mod


# ---------------------------------------------------------------------------
# Fake OpenAI SDK
# ---------------------------------------------------------------------------

class FakeCompletions:
    """Simulates openai.resources.chat.completions.Completions."""

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"args": args, "kwargs": kwargs, "source": "openai"}


class FakeAsyncCompletions:
    """Simulates openai.resources.chat.completions.AsyncCompletions."""

    @staticmethod
    async def create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"args": args, "kwargs": kwargs, "source": "openai_async"}


@pytest.fixture()
def fake_openai():
    """Register a fake ``openai`` package in *sys.modules* and return it.

    The fixture tears down by removing all injected modules and calling
    ``uninstall()`` to reset jiezhu state.
    """
    openai = _make_module("openai")
    resources = _make_module("openai.resources", openai)
    chat = _make_module("openai.resources.chat", resources)
    completions_mod = _make_module("openai.resources.chat.completions", chat)

    completions_mod.Completions = FakeCompletions
    completions_mod.AsyncCompletions = FakeAsyncCompletions

    injected = [
        "openai",
        "openai.resources",
        "openai.resources.chat",
        "openai.resources.chat.completions",
    ]
    for mod_name in injected:
        sys.modules[mod_name] = eval(mod_name.replace("openai", "openai", 1), {"openai": openai}) if mod_name == "openai" else None

    # Simpler: just assign directly
    sys.modules["openai"] = openai
    sys.modules["openai.resources"] = resources
    sys.modules["openai.resources.chat"] = chat
    sys.modules["openai.resources.chat.completions"] = completions_mod

    yield openai

    # Teardown
    from jiezhu.hijack import uninstall
    uninstall()

    for mod_name in injected:
        sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# Fake Anthropic SDK
# ---------------------------------------------------------------------------

class FakeMessages:
    """Simulates anthropic.resources.messages.Messages."""

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"args": args, "kwargs": kwargs, "source": "anthropic"}


class FakeAsyncMessages:
    """Simulates anthropic.resources.messages.AsyncMessages."""

    @staticmethod
    async def create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"args": args, "kwargs": kwargs, "source": "anthropic_async"}


@pytest.fixture()
def fake_anthropic():
    """Register a fake ``anthropic`` package in *sys.modules* and return it."""
    anthropic = _make_module("anthropic")
    resources = _make_module("anthropic.resources", anthropic)
    messages_mod = _make_module("anthropic.resources.messages", resources)

    messages_mod.Messages = FakeMessages
    messages_mod.AsyncMessages = FakeAsyncMessages

    injected = [
        "anthropic",
        "anthropic.resources",
        "anthropic.resources.messages",
    ]
    sys.modules["anthropic"] = anthropic
    sys.modules["anthropic.resources"] = resources
    sys.modules["anthropic.resources.messages"] = messages_mod

    yield anthropic

    # Teardown
    from jiezhu.hijack import uninstall
    uninstall()

    for mod_name in injected:
        sys.modules.pop(mod_name, None)


@pytest.fixture()
def fake_both(fake_openai, fake_anthropic):
    """Convenience fixture that installs both fake SDKs."""
    return fake_openai, fake_anthropic


# ---------------------------------------------------------------------------
# Utility fixture: capture stderr for jiezhu output
# ---------------------------------------------------------------------------

@pytest.fixture()
def capture_output():
    """Return an ``io.StringIO`` that can be passed as ``output`` to install()."""
    return io.StringIO()
