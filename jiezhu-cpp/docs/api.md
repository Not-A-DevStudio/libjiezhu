# API Reference

Namespace: `jie`

Header files:

- `#include <jie/jiezhu.hpp>`(unified header, includes all public headers)

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
- `std::string base_url`：Default `https://api.openai.com/v1`
- `long timeout_seconds = 300`：curl timeout (seconds)
- `std::string organization`：If non-empty, sends header `OpenAI-Organization: ...`
- `std::string project`：If non-empty, sends header `OpenAI-Project: ...`

### `class client`

#### Constructor

- `explicit client(const client_options& options)`

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

2. `chat_completion_response chat_completions_jiezhu(const chat_completion_request& request) const` and `chat_completion_response chat_completions_jiezhu(const chat_completion_request& request, const std::string prompt_prefix) const`

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

## 3) Aggregated Header

- `jie/jiezhu.hpp` currently only `#include <jie/chat.hpp>`, providing a stable entry point for callers.
