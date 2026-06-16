"""@file conftest.py
@brief Shared pytest fixtures for the jiezhu Python test-suite.

The fixtures register fake ``openai`` and ``anthropic`` packages in
:data:`sys.modules` so that the unit tests can exercise the
monkey-patching code paths without depending on either SDK being
installed or on any real API key.
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
    """@brief Create a fresh :class:`types.ModuleType` and optionally
    attach it as an attribute of @p parent.

    @param name Dotted module name.
    @param parent Optional parent module that will receive the new
        module as a child attribute.
    @return The newly created module.
    """
    mod = types.ModuleType(name)
    if parent is not None:
        setattr(parent, name.rsplit(".", 1)[-1], mod)
    return mod


# ---------------------------------------------------------------------------
# Fake OpenAI SDK
# ---------------------------------------------------------------------------

class FakeCompletions:
    """@brief Drop-in replacement for
    ``openai.resources.chat.completions.Completions`` used in tests.

    The ``create`` method records the call arguments in a dict instead
    of dispatching a real HTTP request.
    """

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """@brief Record and echo the call arguments.

        @return Dict containing the original ``args``/``kwargs`` and a
            ``source`` tag identifying this fake SDK.
        """
        return {"args": args, "kwargs": kwargs, "source": "openai"}


class FakeAsyncCompletions:
    """@brief Async drop-in replacement for
    ``openai.resources.chat.completions.AsyncCompletions`` used in tests.
    """

    @staticmethod
    async def create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """@brief Record and echo the call arguments asynchronously.

        @return Dict containing the original ``args``/``kwargs`` and a
            ``source`` tag identifying this fake SDK.
        """
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
    """@brief Drop-in replacement for
    ``anthropic.resources.messages.Messages`` used in tests.

    The ``create`` method records the call arguments in a dict instead
    of dispatching a real HTTP request.
    """

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """@brief Record and echo the call arguments.

        @return Dict containing the original ``args``/``kwargs`` and a
            ``source`` tag identifying this fake SDK.
        """
        return {"args": args, "kwargs": kwargs, "source": "anthropic"}


class FakeAsyncMessages:
    """@brief Async drop-in replacement for
    ``anthropic.resources.messages.AsyncMessages`` used in tests.
    """

    @staticmethod
    async def create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        """@brief Record and echo the call arguments asynchronously.

        @return Dict containing the original ``args``/``kwargs`` and a
            ``source`` tag identifying this fake SDK.
        """
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
