"""@file test_openai_decision_gate.py
@brief Tests for the experimental Decisions API jiezhu gate.

The gate asks a model whether the latest user message needs emotional
support and skips prompt injection when the answer is no. These tests
cover the custom-callback path, the default Decisions API path (sync and
async), fail-open behaviour and the per-SDK wrappers.
"""
from __future__ import annotations

import asyncio

import pytest


def _get_fakes():
    """Return the fake SDK classes registered by the fixture."""
    import openai
    Completions = openai.resources.chat.completions.Completions
    AsyncCompletions = openai.resources.chat.completions.AsyncCompletions
    Responses = openai.resources.responses.Responses
    Decisions = openai.resources.decisions.Decisions
    return Completions, AsyncCompletions, Responses, Decisions


@pytest.fixture(autouse=True)
def _reset_gate_state():
    """Ensure one test's gate configuration cannot leak into the next."""
    yield
    from jiezhu.hijack import _CONFIG

    _CONFIG.experimental_decision_gate = False
    _CONFIG.decision_model = "gpt-6-luna"
    _CONFIG.decision_threshold = 0.5
    _CONFIG.decision_fn = None


class _FakeDecisionResource:
    """Minimal stand-in for ``client.decisions``.

    Records request kwargs and returns a canned Decisions API response.
    """

    def __init__(self, response):
        self.calls = []
        self.response = response

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeAsyncDecisionResource:
    """Async variant of :class:`_FakeDecisionResource`."""

    def __init__(self, response):
        self.calls = []
        self.response = response

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    """Minimal fake OpenAI client exposing only the decisions resource."""

    def __init__(self, response):
        self.decisions = _FakeDecisionResource(response)


class _FakeAsyncClient:
    """Async variant of :class:`_FakeClient`."""

    def __init__(self, response):
        self.decisions = _FakeAsyncDecisionResource(response)


class _FakeResource:
    """Stand-in for an OpenAI SDK resource instance (``self._client``)."""

    def __init__(self, client):
        self._client = client


def _predicate_response(probability: float) -> dict:
    """Build a Decisions API response with the gate predicate answer."""
    return {
        "answers": [
            {
                "type": "predicate",
                "name": "needs_emotional_support",
                "probability": probability,
            }
        ]
    }


def _messages(text: str = "I am exhausted and nobody listens to me.") -> list:
    """Build a short conversation containing one user message."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]


class TestCustomDecisionCallback:
    """@brief ``decision_fn`` based gate behaviour."""

    def test_gate_is_off_by_default(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        called = []
        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            decision_fn=lambda text: called.append(text) or False,
        )
        Completions, _, _, _ = _get_fakes()

        result = Completions.create(messages=_messages())
        assert called == []
        assert result["kwargs"]["messages"][0]["content"].startswith("[P] ")

    def test_gate_allows_injection_when_support_is_needed(
        self, fake_openai, capture_output
    ):
        from jiezhu.hijack import install

        seen = []
        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
            decision_fn=lambda text: seen.append(text) or True,
        )
        Completions, _, _, _ = _get_fakes()

        result = Completions.create(messages=_messages("I feel so lonely."))
        assert seen == ["I feel so lonely."]
        assert result["kwargs"]["messages"][0]["content"].startswith("[P] ")

    def test_gate_skips_injection_when_support_is_not_needed(
        self, fake_openai, capture_output
    ):
        from jiezhu.hijack import install

        seen = []
        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
            decision_fn=lambda text: seen.append(text) or False,
        )
        Completions, _, _, _ = _get_fakes()

        result = Completions.create(messages=_messages("What is 2 + 2?"))
        assert seen == ["What is 2 + 2?"]
        sent = result["kwargs"]["messages"]
        assert sent[0]["content"] == "You are a helpful assistant."
        assert "skipping prompt injection" in capture_output.getvalue()

    def test_gate_fails_open_when_callback_raises(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        def broken(_text):
            raise RuntimeError("gate broke")

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
            decision_fn=broken,
        )
        Completions, _, _, _ = _get_fakes()

        result = Completions.create(messages=_messages())
        assert result["kwargs"]["messages"][0]["content"].startswith("[P] ")
        output = capture_output.getvalue()
        assert "unable to evaluate user input" in output
        assert "injecting the prompt anyway" in output

    def test_gate_is_skipped_without_user_text(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        called = []
        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
            decision_fn=lambda text: called.append(text) or False,
        )
        Completions, _, _, _ = _get_fakes()

        result = Completions.create(
            messages=[{"role": "system", "content": "Only system."}]
        )
        assert called == []
        assert result["kwargs"]["messages"][0]["content"].startswith("[P] ")

    def test_runtime_toggle_and_threshold_validation(
        self, fake_openai, capture_output
    ):
        from jiezhu.hijack import (
            install,
            is_experimental_decision_gate_enabled,
            set_experimental_decision_gate,
        )

        install(prefix_text="[P] ", require_confirm=False, output=capture_output)
        assert is_experimental_decision_gate_enabled() is False

        set_experimental_decision_gate(True, threshold=0.8)
        assert is_experimental_decision_gate_enabled() is True

        with pytest.raises(ValueError):
            set_experimental_decision_gate(True, threshold=1.5)

    def test_responses_gate_skips_injection(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
            decision_fn=lambda _text: False,
        )
        _, _, Responses, _ = _get_fakes()

        result = Responses.create(
            model="gpt-4o", instructions="Be helpful.", input="hi"
        )
        assert result["kwargs"]["instructions"] == "Be helpful."

    def test_anthropic_gate_skips_injection(self, fake_both, capture_output):
        from jiezhu.hijack import install

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
            decision_fn=lambda _text: False,
        )
        import anthropic
        Messages = anthropic.resources.messages.Messages

        result = Messages.create(
            system="Be helpful.",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert result["kwargs"]["system"] == "Be helpful."


class TestDefaultDecisionApi:
    """@brief Gate calls routed through ``client.decisions.create``."""

    def test_support_needed_injects_and_uses_latest_user_text(
        self, fake_openai, capture_output
    ):
        from jiezhu.hijack import install

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
        )
        Completions, _, _, _ = _get_fakes()
        client = _FakeClient(_predicate_response(0.92))

        result = Completions.create(
            _FakeResource(client),
            messages=_messages("I am exhausted and nobody listens to me."),
        )

        assert result["kwargs"]["messages"][0]["content"].startswith("[P] ")
        assert len(client.decisions.calls) == 1
        call = client.decisions.calls[0]
        assert call["model"] == "gpt-6-luna"
        assert call["input"] == "I am exhausted and nobody listens to me."
        assert call["questions"][0]["type"] == "predicate"
        assert call["questions"][0]["name"] == "needs_emotional_support"
        assert "情感支持" in call["questions"][0]["instructions"]

    def test_support_not_needed_skips_injection(
        self, fake_openai, capture_output
    ):
        from jiezhu.hijack import install

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
        )
        Completions, _, _, _ = _get_fakes()
        client = _FakeClient(_predicate_response(0.08))

        result = Completions.create(
            _FakeResource(client), messages=_messages("How do I sort a list?")
        )

        assert result["kwargs"]["messages"][0]["content"] == (
            "You are a helpful assistant."
        )
        assert "skipping prompt injection" in capture_output.getvalue()

    def test_object_style_response_is_supported(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        class _Object:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
        )
        Completions, _, _, _ = _get_fakes()
        client = _FakeClient(
            _Object(
                answers=[
                    _Object(
                        type="predicate",
                        name="needs_emotional_support",
                        probability=0.88,
                    )
                ]
            )
        )

        result = Completions.create(_FakeResource(client), messages=_messages())

        assert result["kwargs"]["messages"][0]["content"].startswith("[P] ")

    def test_custom_threshold_is_applied(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
            decision_threshold=0.8,
        )
        Completions, _, _, _ = _get_fakes()
        client = _FakeClient(_predicate_response(0.79))

        result = Completions.create(_FakeResource(client), messages=_messages())

        assert result["kwargs"]["messages"][0]["content"] == (
            "You are a helpful assistant."
        )

    def test_missing_predicate_answer_fails_open(
        self, fake_openai, capture_output
    ):
        from jiezhu.hijack import install

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
        )
        Completions, _, _, _ = _get_fakes()
        client = _FakeClient({"answers": [{"type": "refusal"}]})

        result = Completions.create(_FakeResource(client), messages=_messages())

        assert result["kwargs"]["messages"][0]["content"].startswith("[P] ")
        assert "no usable predicate answer" in capture_output.getvalue()

    def test_async_gate_uses_async_decisions_resource(
        self, fake_openai, capture_output
    ):
        from jiezhu.hijack import install

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
        )
        _, AsyncCompletions, _, _ = _get_fakes()
        client = _FakeAsyncClient(_predicate_response(0.05))

        result = asyncio.run(
            AsyncCompletions.create(
                _FakeResource(client), messages=_messages("Explain closures.")
            )
        )

        assert result["kwargs"]["messages"][0]["content"] == (
            "You are a helpful assistant."
        )
        assert len(client.decisions.calls) == 1
        assert "skipping prompt injection" in capture_output.getvalue()

    def test_async_support_needed_injects(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        install(
            prefix_text="[P] ",
            require_confirm=False,
            output=capture_output,
            experimental_decision_gate=True,
        )
        _, AsyncCompletions, _, _ = _get_fakes()
        client = _FakeAsyncClient(_predicate_response(0.95))

        result = asyncio.run(
            AsyncCompletions.create(
                _FakeResource(client), messages=_messages("I feel hopeless.")
            )
        )

        assert result["kwargs"]["messages"][0]["content"].startswith("[P] ")
