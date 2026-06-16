"""@file __init__.py
@brief Public entry point for the :mod:`jiezhu` Python package.

Exposes the four functions used to install, remove and configure the
system-prompt monkey-patches for the OpenAI and (optionally) Anthropic
SDKs.
"""
from .hijack import install, uninstall, set_prefix, get_prefix

__all__ = ["install", "uninstall", "set_prefix", "get_prefix"]
