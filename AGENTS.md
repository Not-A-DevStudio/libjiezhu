# AGENTS.md — Jiezhu (接住)

## Project Overview

Jiezhu is a cross-language, cross-platform **system prompt injection framework**. It prepends a configurable empathetic prompt prefix to Chat Completion system messages — enabling any AI agent to "steadily catch" (稳稳接住) its users with over-empathetic responses.

The project ships two independent SDK implementations (C++ and Python) plus a shared prompt template. The C++ SDK targets the **OpenAI Chat Completions API**. The Python SDK supports both **OpenAI** and **Anthropic/Claude** APIs.

## Repository Structure

```
libjiezhu/
├── CMakeLists.txt          # Top-level CMake (builds C++ demo via FetchContent)
├── main.cpp                # C++ demo entry point
├── main.py                 # Python demo entry point
├── jiezhu-cpp/             # C++ library (standalone CMake project)
│   ├── CMakeLists.txt      # Library build config (shared/static, install, tests)
│   ├── inc/jie/            # Public headers
│   │   ├── jiezhu.hpp      # Umbrella include
│   │   └── chat.hpp        # Core API: client, request/response types
│   ├── src/                # Implementation (chat.cpp, jiezhu.cpp)
│   ├── tests/              # Catch2 tests (test_chat_curl.cpp)
│   ├── docs/               # C++ docs (api.md, build.md, quickstart.md)
│   └── third_party/        # Vendored deps (curl-8.17.0, nlohmann_json-3.12.0, Catch2-3.12.0)
├── jiezhu-py/              # Python library
│   ├── pyproject.toml      # Package metadata (setuptools, requires openai>=1.0.0)
│   └── src/jiezhu/         # Package source
│       ├── __init__.py     # Exports: install, uninstall, set_prefix, get_prefix
│       └── hijack.py       # Monkey-patching logic for OpenAI SDK
├── jiezhu-prompt/          # Shared prompt assets
│   └── prompt.md           # DEFAULT_PROMPT_PREFIX & ADVANCED_PROMPT_PREFIX templates
└── .github/workflows/      # CI: build-libraries.yml, release-packages.yml
```

## Architecture & Key Concepts

### Prompt Injection Mechanism

Both SDKs share the same core idea: prepend a Chinese empathetic prompt prefix to system messages before sending the request to the OpenAI API.

- **Python (OpenAI)**: Monkey-patches `openai.resources.chat.Completions.create` (and async variant) at runtime via `install()`. The patch intercepts the `messages` argument, finds or inserts a system message, and prepends the prefix text. Call `uninstall()` to restore originals.
- **Python (Claude)**: Also monkey-patches `anthropic.resources.messages.Messages.create` (and `AsyncMessages.create`). Claude uses a top-level `system` parameter (string or content-block list) instead of system messages in the messages array. The patch intercepts this kwarg and prepends the prefix. Anthropic SDK is an optional dependency — if not installed, Claude patching is silently skipped.
- **C++**: Provides a separate `chat_completions_jiezhu()` method on `jie::client` that clones the request, prepends the prefix, and delegates to `chat_completions_create()`. No monkey-patching — opt-in per call.

### C++ Library (`jiezhu-cpp`)

| Component | Description |
|-----------|-------------|
| `jie::client` | HTTP client wrapping libcurl. Configured via `jie::client_options`. |
| `jie::chat_completion_request` | Request struct: model, messages, temperature, max_tokens, extra JSON fields. |
| `jie::chat_completion_response` | Response struct with raw JSON + convenience accessors (`first_content()`). |
| `jie::chat_completion_stream_event` | Streaming SSE chunk representation. |
| `chat_completions_create()` | Pure mode — no prompt modification. |
| `chat_completions_jiezhu()` | Blessing mode — prepends the default or custom prompt prefix. |
| `chat_completions_stream()` / `chat_completions_stream_jiezhu()` | SSE streaming variants. |

**Build options** (CMake):

- `JIEZHU_BUILD_STATIC` — build static library (default: shared/dynamic)
- `JIEZHU_ENABLE_JIEZHU_ABILITY` — compile with full jiezhu prompt prefix baked in (defines `JIE_ENABLE_JIEZHU_ABLITY`)
- `JIEZHU_BUILD_TEST` — build Catch2 test suite
- `JIEZHU_INSTALL` — generate CMake install/export targets

**Dependencies**: libcurl, nlohmann/json (vendored in `third_party/`).

**C++ standard**: C++17.

### Python Library (`jiezhu-py`)

| Export | Description |
|--------|-------------|
| `install()` | Monkey-patches OpenAI SDK and (optionally) Anthropic SDK. Accepts `prefix_text`, `require_confirm`, `max_preview_chars`. |
| `uninstall()` | Restores all original functions (OpenAI + Anthropic). |
| `set_prefix()` | Change the prefix text at runtime. |
| `get_prefix()` | Retrieve the current prefix text. |

**Requires**: Python ≥ 3.9, `openai >= 1.0.0`. Optional: `anthropic >= 0.39.0` (install via `pip install jiezhu[claude]`).

### Prompt Templates (`jiezhu-prompt/prompt.md`)

Two prompt variants:

- **DEFAULT_PROMPT_PREFIX** — Standard empathetic mode with template phrases and example responses.
- **ADVANCED_PROMPT_PREFIX** — Intensified mode described as "overly empathetic, extremely gentle, but slightly cheesy."

Both are in Chinese and follow the pattern: `[Gentle Confirmation] + [Over-Empathy] + [Philosophical Sublimation] + [Substantive Content]`.

## Conventions & Guidelines

### Language & Naming

- C++ namespace: `jie` (abbreviated from "jiezhu"). Header path: `jie/`.
- CMake targets: `jiezhu::jie` (preferred) or `jiezhu::jiezhu`.
- Python package: `jiezhu`. Published to PyPI as `jiezhu`.
- All prompt-related content is in **Chinese** (Mandarin). Keep prompt strings in Chinese — do not translate them.

### Code Style

- **C++**: C++17 features (structured bindings, `std::optional`, `[[nodiscard]]`). Use `nlohmann::json` for JSON. Follow existing Doxygen-style `///` comments on public API.
- **Python**: Dataclasses for configs, `functools.wraps` for decorators, type hints throughout. Support both sync and async OpenAI SDK paths.

### Build & Test

- **C++ build**: `cmake -B build && cmake --build build` from repo root. The top-level CMakeLists uses `FetchContent` to pull in `jiezhu-cpp/`.
- **C++ tests**: Enable with `-DJIEZHU_BUILD_TEST=ON`. Tests use Catch2 v3. Run via `ctest`.
- **Python install**: `pip install -e ./jiezhu-py` (editable). With Claude support: `pip install -e "./jiezhu-py[claude]"`.
- **Python tests**: `pip install -e "./jiezhu-py[test]"` then `python -m pytest jiezhu-py/tests/ -v`. Tests use mocked SDKs — no API keys needed.

### CI/CD

- `.github/workflows/build-libraries.yml` — builds the C++ library.
- `.github/workflows/release-packages.yml` — publishes release artifacts.

### Important Notes

- The `JIEZHU_ENABLE_JIEZHU_ABILITY` compile definition has a known typo (`ABLITY` instead of `ABILITY`). Do **not** fix this without also updating all consumers — it is part of the public ABI.
- The Python monkey-patch stores originals in `_ORIGINALS` list and uses `_INSTALLED` flag to prevent double-patching. Always pair `install()` with `uninstall()` in tests.
- Third-party dependencies in `jiezhu-cpp/third_party/` are vendored snapshots (curl 8.17.0, nlohmann_json 3.12.0, Catch2 3.12.0). Update these by replacing the directory contents — do not use git submodules.
- Windows builds may require linking `ws2_32` for the test target (already handled in CMakeLists).

## When Adding New Language Ports

Follow the established pattern:

1. Create a `jiezhu-<lang>/` directory with its own README and build config.
2. Use the prompt text from `jiezhu-prompt/prompt.md` as the default prefix — do not duplicate or diverge.
3. Provide both an "inject" mode (blessing/jiezhu) and a "pure" mode (passthrough).
4. Add a demo entry point at the repo root (`main.<ext>`).
5. Update the multi-language table in the root `README.md`.
