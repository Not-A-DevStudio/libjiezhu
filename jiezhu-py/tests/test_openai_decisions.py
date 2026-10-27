"""@file test_openai_decisions.py
@brief Tests for Jiezhu's handling of the OpenAI Decisions API.

The Decisions API returns typed probabilities and scores instead of
generated text, so it is intentionally excluded from prompt injection.
These tests lock in that pass-through behaviour.
"""
from __future__ import annotations


def _get_fakes():
    """Return the fake SDK classes registered by the fixture."""
    import openai
    Completions = openai.resources.chat.completions.Completions
    Decisions = openai.resources.decisions.Decisions
    return Completions, Decisions


class TestDecisionsPassThrough:
    """@brief Decisions calls must not be hijacked."""

    def test_install_still_patches_chat_but_not_decisions(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        Completions, Decisions = _get_fakes()
        original_chat_create = Completions.create
        original_decisions_create = Decisions.create

        install(prefix_text="[PREFIX] ", require_confirm=False, output=capture_output)

        assert Completions.create is not original_chat_create
        assert Decisions.create is original_decisions_create

    def test_string_input_is_untouched(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        install(prefix_text="[PREFIX] ", require_confirm=False, output=capture_output)
        _, Decisions = _get_fakes()

        result = Decisions.create(
            model="gpt-6-luna",
            input="I was charged twice.",
            questions=[
                {
                    "type": "predicate",
                    "name": "refund",
                    "instructions": "Is this a refund request?",
                }
            ],
        )

        assert result["kwargs"]["input"] == "I was charged twice."
        assert result["kwargs"]["questions"][0]["name"] == "refund"

    def test_message_input_is_untouched(self, fake_openai, capture_output):
        from jiezhu.hijack import install

        install(prefix_text="[PREFIX] ", require_confirm=False, output=capture_output)
        _, Decisions = _get_fakes()

        messages = [
            {"role": "developer", "content": "Be precise."},
            {"role": "user", "content": "hi"},
        ]
        result = Decisions.create(model="gpt-6-luna", input=messages, questions=[])

        assert result["kwargs"]["input"] == messages
        assert result["kwargs"]["input"][0]["content"] == "Be precise."

    def test_decisions_remain_untouched_after_uninstall(self, fake_openai, capture_output):
        from jiezhu.hijack import install, uninstall

        _, Decisions = _get_fakes()
        original_create = Decisions.create

        install(prefix_text="[PREFIX] ", require_confirm=False, output=capture_output)
        uninstall()

        assert Decisions.create is original_create
