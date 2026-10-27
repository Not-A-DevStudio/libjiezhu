"""@file __init__.py
@brief Public entry point for the :mod:`jiezhu` Python package.

Exposes the functions used to install, remove and configure the
system-prompt monkey-patches for the OpenAI (Chat + Responses) and
(optionally) Anthropic SDKs, plus the opt-in :func:`enabled` context
manager and :func:`jiezhu` decorator for per-call injection.

The OpenAI Decisions API is not patched directly (it returns typed
answers rather than generated text), but it powers the opt-in
experimental decision gate configured through
:func:`set_experimental_decision_gate`.
"""
from .hijack import (
    install,
    uninstall,
    set_prefix,
    get_prefix,
    set_experimental_decision_gate,
    is_experimental_decision_gate_enabled,
    enabled,
    jiezhu,
)

__all__ = [
    "install",
    "uninstall",
    "set_prefix",
    "get_prefix",
    "set_experimental_decision_gate",
    "is_experimental_decision_gate_enabled",
    "enabled",
    "jiezhu",
]
