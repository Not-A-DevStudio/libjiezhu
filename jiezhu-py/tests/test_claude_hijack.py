"""@file test_claude_hijack.py
@brief Tests for the Claude / Anthropic API monkey-patching support.

Covers the install/uninstall lifecycle, the three ``system``
representations (str, list-of-content-blocks, missing), the
empty-prefix passthrough and the interactive confirmation prompt.
"""
from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock

import pytest


class TestClaudeInstallUninstall:
    """@brief install() / uninstall() lifecycle for the Anthropic SDK."""

    def test_install_patches_claude_create(self, fake_both, capture_output):
        from jiezhu.hijack import install
        from conftest import FakeMessages

        original_create = FakeMessages.create
        install(require_confirm=False, output=capture_output)

        # After install, Messages.create should be wrapped
        assert FakeMessages.create is not original_create
        assert getattr(FakeMessages.create, "__jiezhu_wrapped__", False) is True

    def test_uninstall_restores_claude(self, fake_both, capture_output):
        from jiezhu.hijack import install, uninstall
        from conftest import FakeMessages

        original_create = FakeMessages.create
        install(require_confirm=False, output=capture_output)
        uninstall()

        # After uninstall, Messages.create should be restored
        assert FakeMessages.create is original_create

    def test_claude_skipped_when_not_installed(self, fake_openai, capture_output):
        """@brief Verify that :func:`install` succeeds even when
        ``anthropic`` is not importable.

        Only the OpenAI targets are patched in that case; no error is
        raised for the missing Anthropic SDK.
        """
        # Ensure anthropic is NOT in sys.modules
        sys.modules.pop("anthropic", None)
        sys.modules.pop("anthropic.resources", None)
        sys.modules.pop("anthropic.resources.messages", None)

        from jiezhu.hijack import install
        # Should not raise
        install(require_confirm=False, output=capture_output)


class TestClaudePrefixLogic:
    """@brief Claude ``system``-parameter prefix prepending logic."""

    def test_prefix_prepended_to_string_system(self, fake_both, capture_output):
        from jiezhu.hijack import install
        from conftest import FakeMessages

        install(
            prefix_text="[PREFIX] ",
            require_confirm=False,
            output=capture_output,
        )

        result = FakeMessages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system="You are helpful.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        sent_system = result["kwargs"]["system"]
        assert isinstance(sent_system, str)
        assert sent_system.startswith("[PREFIX] ")
        assert "You are helpful." in sent_system

    def test_prefix_prepended_to_content_block_system(self, fake_both, capture_output):
        from jiezhu.hijack import install
        from conftest import FakeMessages

        install(
            prefix_text="[PREFIX] ",
            require_confirm=False,
            output=capture_output,
        )

        system_blocks = [{"type": "text", "text": "You are helpful."}]
        result = FakeMessages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system=system_blocks,
            messages=[{"role": "user", "content": "Hi"}],
        )

        sent_system = result["kwargs"]["system"]
        assert isinstance(sent_system, list)
        # The first text block should have the prefix prepended
        first_text_block = next(b for b in sent_system if b.get("type") == "text")
        assert first_text_block["text"].startswith("[PREFIX] ")
        assert "You are helpful." in first_text_block["text"]

    def test_prefix_creates_system_when_missing(self, fake_both, capture_output):
        from jiezhu.hijack import install
        from conftest import FakeMessages

        install(
            prefix_text="[PREFIX]",
            require_confirm=False,
            output=capture_output,
        )

        # No system kwarg at all
        result = FakeMessages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
        )

        sent_system = result["kwargs"]["system"]
        assert sent_system == "[PREFIX]"

    def test_empty_prefix_passes_through_claude(self, fake_both, capture_output):
        from jiezhu.hijack import install
        from conftest import FakeMessages

        install(
            prefix_text="",
            require_confirm=False,
            output=capture_output,
        )

        result = FakeMessages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system="Original.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        # With empty prefix, system should pass through unchanged
        sent_system = result["kwargs"]["system"]
        assert sent_system == "Original."

    def test_original_system_blocks_not_mutated(self, fake_both, capture_output):
        from jiezhu.hijack import install
        from conftest import FakeMessages

        install(
            prefix_text="[PREFIX] ",
            require_confirm=False,
            output=capture_output,
        )

        system_blocks = [{"type": "text", "text": "Original."}]
        original_text = system_blocks[0]["text"]

        FakeMessages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system=system_blocks,
            messages=[{"role": "user", "content": "Hi"}],
        )

        # Original list should not be mutated
        assert system_blocks[0]["text"] == original_text


class TestClaudeConfirmation:
    """@brief Claude confirmation behavior (mirrors the OpenAI flow)."""

    def test_require_confirm_user_accepts_claude(self, fake_both):
        from jiezhu.hijack import install
        from conftest import FakeMessages

        output = io.StringIO()
        install(
            prefix_text="[PREFIX] ",
            require_confirm=True,
            input_fn=lambda _: "y",
            output=output,
        )

        result = FakeMessages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system="Original.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        sent_system = result["kwargs"]["system"]
        assert sent_system.startswith("[PREFIX] ")

    def test_require_confirm_user_rejects_claude(self, fake_both):
        from jiezhu.hijack import install
        from conftest import FakeMessages

        output = io.StringIO()
        install(
            prefix_text="[PREFIX] ",
            require_confirm=True,
            input_fn=lambda _: "n",
            output=output,
        )

        result = FakeMessages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system="Original.",
            messages=[{"role": "user", "content": "Hi"}],
        )

        # User rejected — original system should be preserved
        sent_system = result["kwargs"]["system"]
        assert sent_system == "Original."
