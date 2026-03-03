#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace jie {
    /// @brief Represents a single message in the chat conversation.
    /// The `role` field indicates the speaker (e.g., "system", "user", "assistant"), and the `content` field contains the text of the message.
    struct message {
        std::string role;
        std::string content;

        message() {
        }

        message(const std::string &r, const std::string &c) : role(r), content(c) {
        }
    };

    /// @brief Represents a request to create a chat completion. It includes the model to use, the list of messages in the conversation, and optional parameters like temperature and max tokens. The `extra` field allows for any additional OpenAI-compatible fields to be included in the request.
    struct chat_completion_request {
        std::string model;
        std::vector<message> messages;

        bool stream = false;

        // Optional fields
        std::optional<double> temperature;
        std::optional<int> max_tokens;

        // Forward-compat: allow callers to pass any extra OpenAI-compatible fields.
        nlohmann::json extra = nlohmann::json::object();

        nlohmann::json to_json() const;
    };

    /// @brief Represents the response from a chat completion request. It contains the raw JSON response from the API, as well as convenience accessors for commonly used fields like `id` and `model`. The `first_content()` method provides a quick way to access the content of the first message in the choices array, if available.
    struct chat_completion_response {
        nlohmann::json raw;

        // Convenience accessors (best-effort)
        std::string id;
        std::string model;

        // Returns choices[0].message.content if present, else empty.
        std::string first_content() const;
    };

    /// @brief Represents an event in the chat completion streaming process. It contains the raw JSON chunk received from the stream, as well as best-effort extracted fields such as `delta_role`, `delta_content`, and `finish_reason`. The `done` flag indicates whether the stream has finished.
    struct chat_completion_stream_event {
        nlohmann::json raw;

        // Best-effort extracted fields from OpenAI Chat Completions streaming chunks.
        std::string delta_role;
        std::string delta_content;
        std::string finish_reason;
        bool done = false;
    };

    /// @brief Configuration options for the `client` class, including API key, base URL, timeout, and optional organization and project identifiers. The constructor sets a default base URL for the OpenAI API.
    struct client_options {
        std::string api_key;
        std::string base_url;
        long timeout_seconds = 300;
        std::string organization;
        std::string project;

        client_options() : base_url("https://api.openai.com/v1") {
        }
    };

    class client {
    public:
        explicit client(const client_options &options);

        /// @brief Creates a chat completion based on the provided request. This method sends a request to the API and returns the response as a `chat_completion_response` object.
        chat_completion_response chat_completions_create(const chat_completion_request &request) const;

        /// @brief A specialized version of `chat_completions_create` that applies a predefined "jiezhu" prompt prefix to all system messages in the request. This is designed to enhance the assistant's ability to "catch" the user's input in a supportive manner. The second overload allows for a custom prompt prefix to be provided.
        chat_completion_response chat_completions_jiezhu(
            const chat_completion_request &request) const;

        /// @brief A specialized version of `chat_completions_create` that applies a predefined "jiezhu" prompt prefix to all system messages in the request. This is designed to enhance the assistant's ability to "catch" the user's input in a supportive manner. The second overload allows for a custom prompt prefix to be provided.
        chat_completion_response chat_completions_jiezhu(
            const chat_completion_request &request,
            const std::string &prompt_prefix) const;

        void chat_completions_stream_jiezhu(
            const chat_completion_request &request,
            const std::function<bool(const chat_completion_stream_event &)> &on_event) const;

        void chat_completions_stream_jiezhu(
            const chat_completion_request &request,
            const std::string &prompt_prefix,
            const std::function<bool(const chat_completion_stream_event &)> &on_event) const;

        // Streams SSE chunks. Callback returns true to continue, false to cancel.
        void chat_completions_stream(chat_completion_request request,
                                     const std::function<bool(const chat_completion_stream_event &)> &on_event)
        const;

    private:
        client_options options_;
    };
}; // namespace jie
