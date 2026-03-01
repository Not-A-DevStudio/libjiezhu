# Jiezhu for C++

`jiezhu` is a lightweight C++17 client library for calling OpenAI-style Chat Completions API:

- Non-streaming: Request once and return the complete JSON response
- Streaming: Callback events (delta content) in segments via SSE (`text/event-stream`)

The current public headers of this library:

- `jie/jiezhu.hpp`: Aggregated header (currently only includes `jie/chat.hpp`)
- `jie/chat.hpp`: Core API (data structures + `jie::client`)

## Quick Navigation

- Quickstart: see [quickstart.md](./quickstart.md)
- Build and Install (CMake): see [build.md](./build.md)
- API Reference: see [api.md](./api.md)

## Dependencies

- libcurl (HTTP)
- nlohmann/json (JSON serialization/deserialization)
- Catch2 (testing, optional)

In this repository's CMake configuration, dependencies are built and linked as subprojects via `cpp/third_party`.

## Compatibility Notes

- The target API path is fixed as: `{base_url}/chat/completions`
- Request/response fields are organized in a manner compatible with OpenAI Chat Completions; additionally, custom fields can be passed through `extra` to maintain forward compatibilitys with potential future API changes/extensions.

## Security

- You have to enable the option `JIEZHU_ENABLE_JIEZHU_ABLITY` to use the modified OpenAI API ability of `jiezhu` (e.g., `chat_completion_jiezhu`), which will add a prefix to the system prompt. 
- If you enable this option, the default name of your pre-built library will be `libjiezhu_full` instead of `libjiezhu`.