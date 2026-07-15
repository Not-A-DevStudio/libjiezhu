"""@file __init__.py
@brief Public entry point for the :mod:`jiezhu` Python package.

Exposes the functions used to install, remove and configure the
system-prompt monkey-patches for the OpenAI (Chat + Responses) and
(optionally) Anthropic SDKs, plus the opt-in :func:`enabled` context
manager and :func:`jiezhu` decorator for per-call injection.
"""
from .hijack import (
    install,
    uninstall,
    set_prefix,
    get_prefix,
    enabled,
    jiezhu,
)

__all__ = [
    "install",
    "uninstall",
    "set_prefix",
    "get_prefix",
    "enabled",
    "jiezhu",
]
