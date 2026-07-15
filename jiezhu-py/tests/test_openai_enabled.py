"""@file test_openai_enabled.py
@brief Tests for the opt-in :func:`enabled` context manager and
:func:`catch` decorator added in v0.3.1.

Covers selective install mode, per-call prefix overrides, nested
enabled blocks, and the decorated-function API for both the Chat
Completions and Responses SDKs.
"""
from __future__ import annotations

import io

import pytest


def _get_fakes():
    """Return the SDK classes from the fake module hierarchy.

    The conftest registers a fake ``openai`` package (and optionally
    ``anthropic``) in ``sys.modules``.  ``install()`` patches classes
    discovered by traversing these fake modules, so tests **must**
    reference the same objects through the same module lookups rather
    than importing ``FakeCompletions`` from ``tests.conftest`` (which
    is a different class object due to conftest isolation).
    """
    import openai
    Completions = openai.resources.chat.completions.Completions
    AsyncCompletions = openai.resources.chat.completions.AsyncCompletions
    Responses = openai.resources.responses.Responses
    return Completions, AsyncCompletions, Responses


class TestSelectiveInstall:
    """@brief ``install(selective=True)`` global vs. context behaviour."""

    def test_global_passthrough_outside_context(self, fake_openai, capture_output):
        from jiezhu.hijack import install, enabled

        install(selective=True, require_confirm=False, output=capture_output)
        Completions, _, _ = _get_fakes()

        result = Completions.create(
            messages=[{"role": "system", "content": "Keep me."},
                      {"role": "user", "content": "hi"}]
        )
        sent = result["kwargs"]["messages"]
        assert sent[0]["content"] == "Keep me."

    def test_enabled_activates_injection(self, fake_openai, capture_output):
        from jiezhu.hijack import install, enabled

        install(selective=True, require_confirm=False, output=capture_output)
        Completions, _, _ = _get_fakes()

        with enabled():
            result = Completions.create(
                messages=[{"role": "system", "content": "Base."}]
            )
        sent = result["kwargs"]["messages"]
        assert "你是一个AI助手" in sent[0]["content"]  # default prefix
        assert "Base." in sent[0]["content"]

    def test_enabled_custom_prefix(self, fake_openai, capture_output):
        from jiezhu.hijack import install, enabled

        install(selective=True, require_confirm=False, output=capture_output)
        Completions, _, _ = _get_fakes()

        with enabled(prefix="[CUSTOM] "):
            result = Completions.create(
                messages=[{"role": "system", "content": "Base."}]
            )
        sent = result["kwargs"]["messages"]
        assert sent[0]["content"] == "[CUSTOM] Base."

    def test_enabled_fallback_to_global_prefix(self, fake_openai, capture_output):
        from jiezhu.hijack import install, enabled, set_prefix

        install(selective=True, require_confirm=False, output=capture_output)
        Completions, _, _ = _get_fakes()
        set_prefix("[GLOBAL] ")

        with enabled():
            result = Completions.create(
                messages=[{"role": "system", "content": "Base."}]
            )
        sent = result["kwargs"]["messages"]
        assert sent[0]["content"] == "[GLOBAL] Base."

    def test_enabled_restores_passthrough_after_exit(self, fake_openai, capture_output):
        from jiezhu.hijack import install, enabled

        install(selective=True, require_confirm=False, output=capture_output)
        Completions, _, _ = _get_fakes()

        with enabled():
            pass  # enter + exit
        result = Completions.create(
            messages=[{"role": "system", "content": "Keep."}]
        )
        assert result["kwargs"]["messages"][0]["content"] == "Keep."

    def test_nested_enabled_overrides(self, fake_openai, capture_output):
        from jiezhu.hijack import install, enabled

        install(selective=True, require_confirm=False, output=capture_output)
        Completions, _, _ = _get_fakes()

        with enabled(prefix="[OUTER] "):
            with enabled(prefix="[INNER] "):
                result = Completions.create(
                    messages=[{"role": "system", "content": "X"}]
                )
        sent = result["kwargs"]["messages"]
        assert sent[0]["content"] == "[INNER] X"

    def test_global_mode_ignores_selective(self, fake_openai, capture_output):
        """``install(selective=False)`` — default — injects everywhere."""
        from jiezhu.hijack import install

        install(selective=False, prefix_text="[GLOBAL] ", require_confirm=False,
                output=capture_output)
        Completions, _, _ = _get_fakes()

        result = Completions.create(
            messages=[{"role": "system", "content": "Base."}]
        )
        assert result["kwargs"]["messages"][0]["content"] == "[GLOBAL] Base."


class TestEnabledConfirmation:
    """@brief ``require_confirm`` in ``enabled()``."""

    def test_enabled_accept_confirm(self, fake_openai):
        from jiezhu.hijack import install, enabled

        out = io.StringIO()
        install(selective=True, require_confirm=False, input_fn=lambda _: "y", output=out)
        Completions, _, _ = _get_fakes()

        with enabled(prefix="[P] ", require_confirm=True):
            result = Completions.create(
                messages=[{"role": "system", "content": "Base."}]
            )
        sent = result["kwargs"]["messages"]
        assert sent[0]["content"] == "[P] Base."

    def test_enabled_reject_confirm(self, fake_openai):
        from jiezhu.hijack import install, enabled

        out = io.StringIO()
        install(selective=True, require_confirm=False, input_fn=lambda _: "n", output=out)
        Completions, _, _ = _get_fakes()

        with enabled(prefix="[P] ", require_confirm=True):
            result = Completions.create(
                messages=[{"role": "system", "content": "Base."}]
            )
        sent = result["kwargs"]["messages"]
        assert sent[0]["content"] == "Base."


class TestDecorator:
    """@brief ``@jiezhu`` decorator."""

    def test_decorator_injects_prefix(self, fake_openai, capture_output):
        from jiezhu.hijack import install, jiezhu

        install(selective=True, require_confirm=False, output=capture_output)
        Completions, _, _ = _get_fakes()

        @jiezhu(prefix="[DECO] ")
        def call_sdk():
            return Completions.create(
                messages=[{"role": "system", "content": "Base."}]
            )

        result = call_sdk()
        assert result["kwargs"]["messages"][0]["content"] == "[DECO] Base."

    def test_decorator_without_args_uses_global_prefix(self, fake_openai, capture_output):
        from jiezhu.hijack import install, jiezhu, set_prefix

        install(selective=True, require_confirm=False, output=capture_output)
        Completions, _, _ = _get_fakes()
        set_prefix("[GLOBAL] ")

        @jiezhu()
        def call_sdk():
            return Completions.create(
                messages=[{"role": "system", "content": "Base."}]
            )

        result = call_sdk()
        assert result["kwargs"]["messages"][0]["content"] == "[GLOBAL] Base."

    def test_decorator_outside_still_global_inactive(self, fake_openai, capture_output):
        """Decorated function injects; undecorated calls pass through."""
        from jiezhu.hijack import install, jiezhu

        install(selective=True, require_confirm=False, output=capture_output)
        Completions, _, _ = _get_fakes()

        @jiezhu(prefix="[DECO] ")
        def decorated():
            return Completions.create(
                messages=[{"role": "system", "content": "Inside."}]
            )

        decorated()
        result = Completions.create(
            messages=[{"role": "system", "content": "Outside."}]
        )
        assert result["kwargs"]["messages"][0]["content"] == "Outside."


class TestEnabledResponses:
    """@brief ``enabled()`` with the OpenAI Responses API."""

    def test_enabled_injects_into_instructions(self, fake_openai, capture_output):
        from jiezhu.hijack import install, enabled

        install(selective=True, require_confirm=False, output=capture_output)
        _, _, Responses = _get_fakes()

        with enabled(prefix="[RESP] "):
            result = Responses.create(
                model="gpt-4o",
                instructions="Be helpful.",
                input="hi",
            )
        assert result["kwargs"]["instructions"] == "[RESP] Be helpful."

    def test_enabled_injects_into_input_list(self, fake_openai, capture_output):
        from jiezhu.hijack import install, enabled

        install(selective=True, require_confirm=False, output=capture_output)
        _, _, Responses = _get_fakes()

        with enabled(prefix="[IN] "):
            result = Responses.create(
                model="gpt-4o",
                input=[{"role": "user", "content": "hi"}],
            )
        sent = result["kwargs"]["input"]
        assert sent[0]["role"] == "system"
        assert sent[0]["content"] == "[IN] "

    def test_decorator_responses(self, fake_openai, capture_output):
        from jiezhu.hijack import install, jiezhu

        install(selective=True, require_confirm=False, output=capture_output)
        _, _, Responses = _get_fakes()

        @jiezhu(prefix="[D] ")
        def call():
            return Responses.create(
                model="gpt-4o", instructions="help", input="say"
            )

        result = call()
        assert result["kwargs"]["instructions"] == "[D] help"


class TestEnabledAnthropic:
    """@brief ``enabled()`` with the Anthropic API."""

    def test_enabled_anthropic_string_system(self, fake_both, capture_output):
        from jiezhu.hijack import install, enabled

        install(selective=True, require_confirm=False, output=capture_output)
        import anthropic
        Messages = anthropic.resources.messages.Messages

        with enabled(prefix="[CLAUDE] "):
            result = Messages.create(
                system="Be helpful.",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert result["kwargs"]["system"] == "[CLAUDE] Be helpful."

    def test_enabled_anthropic_system_blocks(self, fake_both, capture_output):
        from jiezhu.hijack import install, enabled

        install(selective=True, require_confirm=False, output=capture_output)
        import anthropic
        Messages = anthropic.resources.messages.Messages

        blocks = [{"type": "text", "text": "Be helpful."}]
        with enabled(prefix="[C] "):
            result = Messages.create(
                system=blocks,
                messages=[{"role": "user", "content": "Hi"}],
            )
        sent_blocks = result["kwargs"]["system"]
        assert sent_blocks[0]["text"] == "[C] Be helpful."

    def test_decorator_anthropic(self, fake_both, capture_output):
        from jiezhu.hijack import install, jiezhu

        install(selective=True, require_confirm=False, output=capture_output)
        import anthropic
        Messages = anthropic.resources.messages.Messages

        @jiezhu(prefix="[A] ")
        def call():
            return Messages.create(
                system="Base.",
                messages=[{"role": "user", "content": "Hi"}],
            )

        result = call()
        assert result["kwargs"]["system"] == "[A] Base."
