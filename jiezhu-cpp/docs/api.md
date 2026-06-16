# API Reference

Namespace: `jie`

Header files:

- `#include <jie/jiezhu.hpp>`(unified header, includes all public headers and prompt prefix)

## 1) Data Structures

### `struct message`

Represents a single message in the chat conversation. The `role` field indicates the speaker (e.g., "system", "user", "assistant"), and the `content` field contains the text of the message.

Fields:

- `std::string role`：role, ( e.g. `system` / `user` / `assistant` )
- `std::string content`：Text content

Constructors:

- `message()`
- `message(const std::string& r, const std::string& c)`

### `struct chat_completion_request`

Chat Completions request body.

Required fields:

- `std::string model`
- `std::vector<message> messages`

Streaming：

- `bool stream = false`

Optional fields：

- `std::optional<double> temperature`
- `std::optional<int> max_tokens`

Extra fields：

- `nlohmann::json extra = object()`：Allows callers to include any OpenAI-compatible fields in `extra`, such as `top_p`, `frequency_penalty`, `response_format`, etc.

Methods：
- `nlohmann::json to_json() const`
  - Returns the final JSON to be sent: `extra` is used as the base object, then `model`, `messages`, `stream`, `temperature`, and `max_tokens` are overwritten/filled in.

### `struct chat_completion_response`

Non-streaming response wrapper.

Fields：

- `nlohmann::json raw`：the raw JSON response from the server
- `std::string id`：if the response contains `id` and it is a string, it will be extracted
- `std::string model`：if the response contains `model` and it is a string, it will be extracted

Methods：
- `std::string first_content() const`
  - Tries to return `choices[0].message.content` (returns an empty string if not present)
  - This method is best-effort: exceptions during parsing are swallowed and an empty string is returned

### `struct chat_completion_stream_event`

Streaming (SSE) event wrapper.

Fields：

- `nlohmann::json raw`：the raw JSON of this chunk (if parsable)
- `std::string delta_role`：best-effort extraction of `choices[0].delta.role`
- `std::string delta_content`：best-effort extraction of `choices[0].delta.content`
- `std::string finish_reason`：best-effort extraction of `choices[0].finish_reason`
- `bool done = false`
  - When `data: [DONE]` is received, `done=true`
  - Or when `finish_reason` is non-empty, `done=true`

## 2) Client

### `struct client_options`

Client configuration.

Fields：

- `std::string api_key`：Used for `Authorization: Bearer <api_key>`
- `std::string base_url`：**Have no default value**, call `client()` to directly get a default configuration with `base_url = "https://api.openai.com/v1"`; if you want to use a different base URL, you must set this field explicitly.
- `long timeout_seconds = 300`：curl timeout (seconds)
- `std::string organization`：If non-empty, sends header `OpenAI-Organization: ...`
- `std::string project`：If non-empty, sends header `OpenAI-Project: ...`

it supports initialization via aggregate initialization, e.g.:

```cpp
client_options options{
    .api_key = "sk-xxx",
    .base_url = "https://api.openai.com/v1",
    .timeout_seconds = 300,
    .organization = "org-xxx",
    .project = "proj-xxx"};
```

so you may write `client c({...})` to construct a client with custom options.

### `class client`

#### Constructor

- `explicit client(const client_options& options)`
- `client()`: equivalent to `client(client_options{.base_url = "https://api.openai.com/v1"})`
Internally ensures curl global initialization (RAII static object).

#### Non-Streaming Requests

1. `chat_completion_response chat_completions_create(const chat_completion_request& request) const`

Behavior：

- POST to: `{base_url}/chat/completions`
- Headers:
  - `Content-Type: application/json`
  - `Accept: application/json`
  - `Authorization: Bearer ...` (if `api_key` is non-empty)
  - `OpenAI-Organization` / `OpenAI-Project` (if the corresponding fields are non-empty)

Errors：

- curl call failure will throw `std::runtime_error`
- HTTP non-2xx will throw `std::runtime_error("HTTP <code>: <body>")`

2. `chat_completion_response chat_completions_create_anthropic(const chat_completion_request& request) const`

Behavior：

- POST to: `{base_url}/messages`
- Request body follows Anthropic Messages API shape:
  - All `system` messages are concatenated into a top-level `system` field, separated by blank lines
  - Non-`system` messages are kept in `messages`
  - `model` and `stream` are written into the request body
  - `temperature` is forwarded when present
  - `max_tokens` is forwarded when present; otherwise the client inserts a default `max_tokens = 1024`
- Headers:
  - `Content-Type: application/json`
  - `Accept: application/json`
  - `anthropic-version: 2023-06-01`
  - `x-api-key: ...` (if `api_key` is non-empty)

Errors：

- curl call failure will throw `std::runtime_error`
- HTTP non-2xx will throw `std::runtime_error("HTTP <code>: <body>")`

3. `chat_completion_response chat_completions_jiezhu_anthropic(const chat_completion_request& request) const` and `chat_completion_response chat_completions_jiezhu_anthropic(const chat_completion_request& request, const std::string prompt_prefix) const`

Behavior：

- Similar to `chat_completions_create_anthropic`, but prefixes every `system` message with a "jiezhu" prompt prefix before the Anthropic request body is assembled
- The first overload uses the library's default `jiezhu` prompt prefix, while the second overload allows callers to supply a custom prefix
- The Anthropic conversion rules still apply:
  - system messages are merged into top-level `system`
  - non-system messages remain in `messages`
  - `max_tokens` defaults to `1024` when not provided

Errors：

- Same as `chat_completions_create_anthropic`
- If `JIE_ENABLE_JIEZHU_ABLITY` is not defined, these methods will throw `std::runtime_error` indicating that the jiezhu ability is not enabled

4. `chat_completion_response chat_completions_jiezhu(const chat_completion_request& request) const` and `chat_completion_response chat_completions_jiezhu(const chat_completion_request& request, const std::string prompt_prefix) const`

Behavior：
- Similar to `chat_completions_create`, but with additional processing to prepend a "jiezhu" prompt prefix to all system messages in the request. This is designed to enhance the assistant's ability to "catch" the user's input in a supportive manner.
- The first overload uses a default "jiezhu" prompt prefix, while the second allows callers to provide a custom prompt prefix.

Errors：

- Same as `chat_completions_create`
- If `JIE_ENABLE_JIEZHU_ABLITY` is not defined, these methods will throw `std::runtime_error` indicating that the jiezhu ability is not enabled.

#### Streaming Requests (SSE)

1. `void chat_completions_stream(chat_completion_request request, const std::function<bool(const chat_completion_stream_event&)>& on_event) const`

Key Points:

- This method will force `request.stream = true`
- Request header `Accept: text/event-stream`
- Parsing strategy: process lines by `\n`; only lines starting with `data:` are processed
  - `data: [DONE]` will generate an event with `done=true`
  - `data: {json}` will attempt to parse and extract delta fields
- Callback `on_event`:
  - Return `true`: continue receiving
  - Return `false`: cancel the request (by returning 0 in the curl write callback)

Note:

- Since streaming is best-effort line parsing, if the server's chunk/line behavior is inconsistent with standard SSE, you may need to implement more robust concatenation handling at a higher level.

2. `void chat_completions_stream_jiezhu(chat_completion_request request, const std::function<bool(const chat_completion_stream_event&)>& on_event) const` and `void chat_completions_stream_jiezhu(chat_completion_request request, const std::string prompt, const std::function<bool(const chat_completion_stream_event&)>& on_event) const`

Key Points:
- Similar to `chat_completions_stream`, but with additional processing to prepend a "jiezhu" prompt prefix to all system messages in the request before initiating the streaming request.
- The first overload uses a default "jiezhu" prompt prefix, while the second allows callers to provide a custom prompt prefix.

Errors：
- Same as `chat_completions_stream`
- If `JIE_ENABLE_JIEZHU_ABLITY` is not defined, these methods will throw `std::runtime_error` indicating that the jiezhu ability is not enabled.

Note:

- There is currently no Anthropic streaming equivalent in the public API; the Anthropic path is limited to non-streaming requests.

## 4) Responses API

### `struct response_request`

Request body for the OpenAI Responses API (`/v1/responses`).

Fields:

- `std::string model` — Model identifier (required)
- `std::string input_text` — Plain text input. When non-empty, used as the `input` string and takes precedence over `input_messages`
- `std::vector<message> input_messages` — Structured multi-turn input as message array
- `std::string instructions` — System / developer instructions for the model
- `bool stream = false` — Whether to stream via SSE. Set internally by `responses_stream`
- `std::optional<double> temperature` — Sampling temperature
- `std::optional<int> max_output_tokens` — Maximum output tokens (maps to `max_output_tokens` in the API)
- `nlohmann::json extra = object()` — Forward-compat bucket for additional fields

Methods:
- `nlohmann::json to_json() const` — Serializes the request. Uses `extra` as base, then overwrites `model`, `input`, `stream`, `instructions`, etc.

### `struct response_response`

Non-streaming Responses API response.

Fields:

- `nlohmann::json raw` — Raw JSON response from the server
- `std::string id` — Best-effort extraction of `id`
- `std::string model` — Best-effort extraction of `model`

Methods:
- `std::string first_content() const` — Returns the text from `output[0].content[*].text` (structured format) or `output_text` (convenience field), or empty string if neither is present

### `struct response_stream_event`

Streaming SSE event wrapper for the Responses API.

Fields:

- `nlohmann::json raw` — Raw JSON data of this event (if parsable)
- `std::string event_type` — SSE event type (e.g. `"response.output_text.delta"`)
- `std::string delta` — Best-effort extraction of incremental text from `data.delta`
- `bool done = false` — Set to `true` when a terminal event (`response.completed` or `error`) is received

### `class client` — Responses API Methods

#### Non-Streaming

1. `response_response responses_create(const response_request& request) const`

Behavior:
- POST to: `{base_url}/responses`
- Headers: same as `chat_completions_create` (`Content-Type`, `Authorization`, `OpenAI-Organization`, `OpenAI-Project`)
- Errors: same as `chat_completions_create`

2. `response_response responses_jiezhu(const response_request& request) const` and `response_response responses_jiezhu(const response_request& request, const std::string& prompt_prefix) const`

Behavior:
- Prepends the jiezhu prefix to `instructions` and any system messages in `input_messages`, then delegates to `responses_create`
- The `instructions` field is the primary target (equivalent to system messages in Chat Completions)
- Errors: same as `responses_create`
- If `JIE_ENABLE_JIEZHU_ABLITY` is not defined, throws `std::runtime_error`

#### Streaming

3. `void responses_stream(response_request request, const std::function<bool(const response_stream_event&)>& on_event) const`

Key Points:
- Sets `request.stream = true` before sending
- GET/POST to `{base_url}/responses`
- Header `Accept: text/event-stream`
- SSE parsing handles `event:` and `data:` lines (Responses API format)
- Events like `response.output_text.delta` carry incremental text in `event.delta`
- The terminal event (`response.completed` or `error`) sets `event.done = true` and terminates the stream
- Callback `on_event`: return `true` to continue, `false` to cancel

4. `void responses_stream_jiezhu(const response_request& request, const std::function<bool(const response_stream_event&)>& on_event) const` and `void responses_stream_jiezhu(const response_request& request, const std::string& prompt_prefix, const std::function<bool(const response_stream_event&)>& on_event) const`

Behavior:
- Prepends the jiezhu prefix, then delegates to `responses_stream`
- Errors: same as `responses_stream`
- If `JIE_ENABLE_JIEZHU_ABLITY` is not defined, throws `std::runtime_error`

## 3) Aggregated Header

- `jie/jiezhu.hpp` currently only `#include <jie/chat.hpp>`, providing a stable entry point for callers.
