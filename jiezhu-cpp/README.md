# Jiezhu for C++

`jiezhu-cpp` is a lightweight C++17 client library for calling OpenAI-style Chat Completions API:

- Non-streaming: Request once and return the complete JSON response
- Streaming: Callback events (delta content) in segments via SSE (`text/event-stream`)

The current public headers of this library:

- `jie/jiezhu.hpp`: Aggregated header (currently only includes `jie/chat.hpp`)
- `jie/chat.hpp`: Core API (data structures + `jie::client`)

## Dependency

All dependencies are included under folder `jiezhu-cpp/third_party`: 

- curl-8.17.0: CHANGE has made to its build options to auto enable `schannel` on Windows, so you don't need to worry about OpenSSL dependency when building on Windows.   
- nlohmann_json-3.12.0

## Quick Navigation

- Index: see [index.md](./docs/index.md)
- Quickstart: see [quickstart.md](./docs/quickstart.md)
- Build and Install (CMake): see [build.md](./docs/build.md)
- API Reference: see [api.md](./docs/api.md)