"""Regression tests for existing OpenAI monkey-patching functionality."""
from __future__ import annotations

import io
import inspect
from unittest.mock import MagicMock

import pytest


class TestInstallUninstall:
    """install() / uninstall() lifecycle for OpenAI SDK."""

    def test_install_patches_openai_create(self, fake_openai, capture_output):
        from jiezhu.hijack import install, _INSTALLED
        from tests.conftest import FakeCompletions

        original_create = FakeCompletions.create
        install(require_confirm=False, output=capture_output)

        # After install, the class method should be wrapped
        assert FakeCompletions.create is not original_create
        assert getattr(FakeCompletions.create, "__jiezhu_wrapped__", False) is True

    def test_uninstall_restores_openai(self, fake_openai, capture_output):
        from jiezhu.hijack import install, uninstall
        from tests.conftest import FakeCompletions

        original_create = FakeCompletions.create
        install(require_confirm=False, output=capture_output)
        uninstall()

        # After uninstall, the class method should be restored
        assert FakeCompletions.create is original_create

    def test_double_install_is_idempotent(self, fake_openai, capture_output):
        from jiezhu.hijack import install, _ORIGINALS

        install(require_confirm=False, output=capture_output)
        originals_count = len(_ORIGINALS)

        install(require_confirm=False, output=capture_output)
        # Should not double-patch
        assert len(_ORIGINALS) == originals_count


class TestPrefixLogic:
    """OpenAI message prefix prepending logic."""

    def test_prefix_prepended_to_system_message(self, fake_openai, capture_output):
        from jiezhu.hijack import install
        from tests.conftest import FakeCompletions

        install(
            prefix_text="[PREFIX] ",
            require_confirm=False,
            output=capture_output,
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = FakeCompletions.create(messages=messages)

        # The wrapped function should have modified the messages
        sent_messages = result["kwargs"]["messages"]
        system_msg = next(m for m in sent_messages if m["role"] == "system")
        assert system_msg["content"].startswith("[PREFIX] ")
        assert "You are helpful." in system_msg["content"]

    def test_prefix_creates_system_message_when_missing(self, fake_openai, capture_output):
        from jiezhu.hijack import install
        from tests.conftest import FakeCompletions

        install(
            prefix_text="[PREFIX]",
            require_confirm=False,
            output=capture_output,
        )

        messages = [
            {"role": "user", "content": "Hi"},
        ]
        result = FakeCompletions.create(messages=messages)

        sent_messages = result["kwargs"]["messages"]
        # A system message should have been inserted at position 0
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[0]["content"] == "[PREFIX]"

    def test_empty_prefix_passes_through(self, fake_openai, capture_output):
        from jiezhu.hijack import install
        from tests.conftest import FakeCompletions

        install(
            prefix_text="",
            require_confirm=False,
            output=capture_output,
        )

        messages = [
            {"role": "system", "content": "Original."},
            {"role": "user", "content": "Hi"},
        ]
        result = FakeCompletions.create(messages=messages)

        # With empty prefix, messages should pass through unchanged
        sent_messages = result["kwargs"]["messages"]
        system_msg = next(m for m in sent_messages if m["role"] == "system")
        assert system_msg["content"] == "Original."

    def test_original_messages_not_mutated(self, fake_openai, capture_output):
        from jiezhu.hijack import install
        from tests.conftest import FakeCompletions

        install(
            prefix_text="[PREFIX] ",
            require_confirm=False,
            output=capture_output,
        )

        messages = [
            {"role": "system", "content": "Original."},
            {"role": "user", "content": "Hi"},
        ]
        original_content = messages[0]["content"]
        FakeCompletions.create(messages=messages)

        # Original list should not be mutated
        assert messages[0]["content"] == original_content


class TestConfirmation:
    """require_confirm behavior."""

    def test_require_confirm_true_user_accepts(self, fake_openai):
        from jiezhu.hijack import install
        from tests.conftest import FakeCompletions

        output = io.StringIO()
        install(
            prefix_text="[PREFIX] ",
            require_confirm=True,
            input_fn=lambda _: "y",
            output=output,
        )

        messages = [
            {"role": "system", "content": "Original."},
            {"role": "user", "content": "Hi"},
        ]
        result = FakeCompletions.create(messages=messages)
        sent_messages = result["kwargs"]["messages"]
        system_msg = next(m for m in sent_messages if m["role"] == "system")
        assert system_msg["content"].startswith("[PREFIX] ")

    def test_require_confirm_true_user_rejects(self, fake_openai):
        from jiezhu.hijack import install
        from tests.conftest import FakeCompletions

        output = io.StringIO()
        install(
            prefix_text="[PREFIX] ",
            require_confirm=True,
            input_fn=lambda _: "n",
            output=output,
        )

        messages = [
            {"role": "system", "content": "Original."},
            {"role": "user", "content": "Hi"},
        ]
        result = FakeCompletions.create(messages=messages)
        sent_messages = result["kwargs"]["messages"]
        system_msg = next(m for m in sent_messages if m["role"] == "system")
        # User rejected — original content should be preserved
        assert system_msg["content"] == "Original."
