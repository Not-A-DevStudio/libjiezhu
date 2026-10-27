/**
 * @file chat.hpp
 * @brief Public chat-completions API for the jiezhu C++ SDK.
 *
 * Defines the request/response value types and the @ref jie::client used to
 * talk to OpenAI-compatible and Anthropic Claude chat-completion endpoints,
 * including the optional "jiezhu" (blessing) prompt-injection variants that
 * prepend an empathetic Chinese system prompt.
 */
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace jie {
    /// @brief A single message in a chat conversation.
    ///
    /// The `role` field identifies the speaker (e.g. @c "system", @c "user",
    /// @c "assistant") and `content` holds the textual payload of the
    /// message.
    struct message {
        /// @brief Speaker role (e.g. @c "system", @c "user", @c "assistant").
        std::string role;
        /// @brief Textual content of the message.
        std::string content;

        /// @brief Default-construct an empty message.
        message() {
        }

        /// @brief Construct a message with explicit role and content.
        /// @param r Speaker role.
        /// @param c Message text.
        message(const std::string &r, const std::string &c) : role(r), content(c) {
        }
    };

    /// @brief Request payload for a chat-completion call.
    ///
    /// Mirrors the OpenAI Chat Completions request shape: a model name, the
    /// message history, and optional sampling parameters. Additional
    /// OpenAI-compatible fields can be passed through @c extra for
    /// forward-compatibility.
    struct chat_completion_request {
        /// @brief Model identifier (e.g. @c "gpt-4o", @c "claude-3-opus").
        std::string model;
        /// @brief Ordered list of messages that make up the conversation.
        std::vector<message> messages;

        /// @brief Whether the request should be answered as a server-sent
        /// event stream. Mutated internally by the streaming entry points.
        bool stream = false;

        /// @brief Optional sampling temperature. @c std::nullopt omits the
        /// field from the JSON payload and lets the server pick a default.
        std::optional<double> temperature;
        /// @brief Optional cap on the number of generated tokens.
        std::optional<int> max_tokens;

        /// @brief Forward-compat bucket for any extra OpenAI-compatible
        /// fields (e.g. @c top_p, @c tools, @c response_format). Defaults
        /// to an empty object.
        nlohmann::json extra = nlohmann::json::object();

        /// @brief Serialize this request to the JSON shape expected by the
        /// OpenAI Chat Completions endpoint.
        /// @return JSON object containing @c model, @c stream, @c messages
        /// and any optional/extra fields.
        [[nodiscard]] nlohmann::json to_json() const;
    };

    /// @brief Response payload returned by a chat-completion call.
    ///
    /// Stores the raw JSON returned by the server together with a few
    /// convenience accessors for the most common fields. A best-effort
    /// helper @c first_content() normalizes the most frequent response
    /// shapes (OpenAI and Anthropic).
    struct chat_completion_response {
        /// @brief Raw JSON body returned by the server.
        nlohmann::json raw;

        // Convenience accessors (best-effort)
        /// @brief Best-effort extraction of the @c id field.
        std::string id;
        /// @brief Best-effort extraction of the @c model field.
        std::string model;

        /// @brief Extract the text of the first assistant message.
        ///
        /// Supports both the OpenAI shape
        /// (`choices[0].message.content`) and the Anthropic shape
        /// (`content[0].text`). Returns an empty string if neither
        /// pattern is recognized.
        /// @return Concatenated text, or empty string when unavailable.
        [[nodiscard]] std::string first_content() const;
    };

    /// @brief A single event in a streamed chat-completion response.
    ///
    /// Carries the raw JSON chunk emitted by the server as well as
    /// best-effort extracted fields such as the delta role/content and
    /// the finish reason. The @c done flag is set for terminal events
    /// (including the @c [DONE] sentinel).
    struct chat_completion_stream_event {
        /// @brief Raw JSON chunk as received from the server.
        nlohmann::json raw;

        // Best-effort extracted fields from OpenAI Chat Completions streaming chunks.
        /// @brief Role announced in the first delta of a streamed choice.
        std::string delta_role;
        /// @brief Incremental text content from the current delta.
        std::string delta_content;
        /// @brief Reason the model stopped generating, when reported.
        std::string finish_reason;
        /// @brief @c true when no further events are expected.
        bool done = false;
    };

    /// @brief Request payload for an OpenAI Responses API call.
    ///
    /// Mirrors the OpenAI Responses request shape: a model name, input
    /// (plain text or structured messages), optional instructions, and
    /// sampling parameters. Additional fields can be passed through
    /// @c extra for forward-compatibility.
    struct response_request {
        /// @brief Model identifier (e.g. @c "gpt-4o").
        std::string model;
        /// @brief Plain-text input. When non-empty, used as the @c input
        /// string. Takes precedence over @c input_messages.
        std::string input_text;
        /// @brief Structured multi-turn input as an array of messages.
        /// Used when @c input_text is empty.
        std::vector<message> input_messages;
        /// @brief System / developer instructions for the model.
        std::string instructions;
        /// @brief Whether the response should be streamed via SSE. Set
        /// internally by the streaming entry points.
        bool stream = false;
        /// @brief Optional sampling temperature.
        std::optional<double> temperature;
        /// @brief Optional cap on output tokens (maps to
        /// @c max_output_tokens in the API).
        std::optional<int> max_output_tokens;
        /// @brief Forward-compat bucket for any extra API fields.
        nlohmann::json extra = nlohmann::json::object();
        /// @brief Serialize this request to the JSON shape expected by the
        /// OpenAI Responses endpoint.
        [[nodiscard]] nlohmann::json to_json() const;
    };

    /// @brief Response payload returned by a Responses API call.
    struct response_response {
        /// @brief Raw JSON body returned by the server.
        nlohmann::json raw;
        /// @brief Best-effort extraction of the @c id field.
        std::string id;
        /// @brief Best-effort extraction of the @c model field.
        std::string model;
        /// @brief Extract the text of the first output item.
        ///
        /// Supports both the structured format
        /// (@c output[0].content[*].text) and the convenience field
        /// (@c output_text). Returns an empty string if neither pattern
        /// is recognized.
        [[nodiscard]] std::string first_content() const;
    };

    /// @brief A single event in a streamed Responses API response.
    struct response_stream_event {
        /// @brief Raw JSON chunk as received from the server.
        nlohmann::json raw;
        /// @brief SSE event type (e.g. @c "response.output_text.delta").
        std::string event_type;
        /// @brief Incremental text content from the delta.
        std::string delta;
        /// @brief @c true when no further events are expected.
        bool done = false;
    };

    /// @brief One option for a Decisions API @c choice question.
    struct decision_choice {
        /// @brief Option value. The API accepts a string or a boolean.
        nlohmann::json value = "";
        /// @brief Optional human-readable explanation of when this
        /// option applies.
        std::string description;

        /// @brief Default-construct an empty choice.
        decision_choice() {
        }

        /// @brief Construct a string-valued choice.
        /// @param v Option value.
        /// @param d Optional description.
        decision_choice(const std::string &v, const std::string &d = "") : value(v), description(d) {
        }

        /// @brief Construct a boolean-valued choice.
        /// @param v Option value.
        /// @param d Optional description.
        decision_choice(bool v, const std::string &d = "") : value(v), description(d) {
        }

        /// @brief Serialize this choice into the Decisions API shape.
        [[nodiscard]] nlohmann::json to_json() const;
    };

    /// @brief One ordered level for a Decisions API @c score question.
    struct decision_level {
        /// @brief Level label (required).
        std::string label;
        /// @brief Optional explanation of the level's criteria.
        std::string description;

        /// @brief Default-construct an empty level.
        decision_level() {
        }

        /// @brief Construct a level with a label and description.
        /// @param l Level label.
        /// @param d Optional level description.
        decision_level(const std::string &l, const std::string &d = "") : label(l), description(d) {
        }

        /// @brief Serialize this level into the Decisions API shape.
        [[nodiscard]] nlohmann::json to_json() const;
    };

    /// @brief A single question in a Decisions API request.
    ///
    /// @c type selects the question kind: @c "predicate", @c "choice" or
    /// @c "score". Choice questions use @c choices; score questions use
    /// @c levels. Unused fields are omitted from the serialized payload.
    struct decision_question {
        /// @brief Question kind: @c "predicate", @c "choice" or @c "score".
        std::string type;
        /// @brief Optional unique name echoed back in the answer.
        std::string name;
        /// @brief Natural-language instructions for the model.
        std::string instructions;
        /// @brief Options for a @c choice question.
        std::vector<decision_choice> choices;
        /// @brief Ordered levels for a @c score question.
        std::vector<decision_level> levels;
        /// @brief Forward-compat bucket for additional question fields.
        nlohmann::json extra = nlohmann::json::object();

        /// @brief Serialize this question into the Decisions API shape.
        [[nodiscard]] nlohmann::json to_json() const;
    };

    /// @brief One message in the Decisions API @c input array.
    ///
    /// @c content is either a plain string or an array of content parts
    /// such as @c {"type":"input_text","text":"..."} and
    /// @c {"type":"input_image","image_url":"..."}. The flexible JSON
    /// representation allows text and image inputs without a dedicated
    /// part type.
    struct decision_input_message {
        /// @brief Speaker role (typically @c "user", @c "developer" or
        /// @c "assistant").
        std::string role;
        /// @brief Text string or array of content parts.
        nlohmann::json content = "";
        /// @brief Forward-compat bucket for additional message fields.
        nlohmann::json extra = nlohmann::json::object();

        /// @brief Default-construct an empty message.
        decision_input_message() {
        }

        /// @brief Construct a message with string content.
        /// @param r Speaker role.
        /// @param c Message text.
        decision_input_message(const std::string &r, const std::string &c) : role(r), content(c) {
        }

        /// @brief Construct a message with structured content parts.
        /// @param r Speaker role.
        /// @param c Content string or array of parts.
        decision_input_message(const std::string &r, const nlohmann::json &c) : role(r), content(c) {
        }

        /// @brief Serialize this message into the Decisions API shape.
        [[nodiscard]] nlohmann::json to_json() const;
    };

    /// @brief Request payload for an OpenAI Decisions API call.
    ///
    /// Mirrors the Decisions request shape: a model name, shared input
    /// evidence (plain text or structured messages) and one or more
    /// questions.
    struct decision_request {
        /// @brief Model identifier. The API currently requires
        /// @c "gpt-6-luna".
        std::string model;
        /// @brief Plain-text input. When non-empty, used as the @c input
        /// string. Takes precedence over @c input_messages.
        std::string input_text;
        /// @brief Structured input messages. Used when @c input_text is
        /// empty.
        std::vector<decision_input_message> input_messages;
        /// @brief Questions to evaluate against the shared input.
        std::vector<decision_question> questions;
        /// @brief Forward-compat bucket for additional API fields.
        nlohmann::json extra = nlohmann::json::object();

        /// @brief Serialize this request to the JSON shape expected by the
        /// OpenAI Decisions endpoint.
        [[nodiscard]] nlohmann::json to_json() const;
    };

    /// @brief One typed answer returned by a Decisions API call.
    ///
    /// The @c type field determines which of the convenience fields is
    /// populated: @c probability for @c "predicate" answers, @c choice
    /// for @c "choice" answers and @c score for @c "score" answers.
    /// @c raw always contains the complete answer object.
    struct decision_answer {
        /// @brief Raw answer object as returned by the server.
        nlohmann::json raw;
        /// @brief Answer kind: @c "predicate", @c "choice", @c "score"
        /// or @c "refusal".
        std::string type;
        /// @brief Question name echoed back by the server (may be empty).
        std::string name;
        /// @brief Probability for a @c "predicate" answer.
        std::optional<double> probability;
        /// @brief Selected value for a @c "choice" answer (string or bool).
        nlohmann::json choice = nullptr;
        /// @brief Probability-weighted score for a @c "score" answer.
        std::optional<double> score;
        /// @brief Confidence reported for @c "choice" / @c "score" answers.
        std::optional<double> confidence;
        /// @brief Raw probability distribution for choice/score answers.
        nlohmann::json probabilities = nlohmann::json::array();
    };

    /// @brief Response payload returned by a Decisions API call.
    struct decision_response {
        /// @brief Raw JSON body returned by the server.
        nlohmann::json raw;
        /// @brief Best-effort extraction of the @c model field.
        std::string model;
        /// @brief Parsed answers, in the same order as the questions.
        std::vector<decision_answer> answers;
    };

    /// @brief Configuration for a @ref jie::client instance.
    ///
    /// Holds the credentials and connection settings used to talk to a
    /// chat-completion API. The default constructor sets the base URL to
    /// the public OpenAI endpoint.
    struct client_options {
        /// @brief Bearer token sent in the @c Authorization header.
        std::string api_key;
        /// @brief Base URL of the API. @c "/chat/completions" or
        /// @c "/messages" is appended when a request is issued.
        std::string base_url;
        /// @brief Per-request libcurl timeout in seconds. Defaults to 300.
        long timeout_seconds = 300;
        /// @brief Optional OpenAI organization header value.
        std::string organization;
        /// @brief Optional OpenAI project header value.
        std::string project;

        /// @brief Experimental: gate jiezhu injection on the Decisions API.
        ///
        /// When @c true, the jiezhu variants ask the Decisions API
        /// whether the latest user message needs emotional support and,
        /// when the predicate probability is below
        /// @ref decision_threshold, send the request without the
        /// jiezhu prompt prefix. Requests without user text and
        /// Decisions API failures fail open (the prefix is injected).
        bool experimental_decision_gate = false;
        /// @brief Model used by the experimental Decisions API gate.
        std::string decision_model = "gpt-6-luna";
        /// @brief Predicate probability threshold for the gate, in
        /// the inclusive range @c [0.0, 1.0].
        double decision_threshold = 0.5;
    };

    /// @brief Thin HTTP client for OpenAI / Anthropic chat-completion APIs.
    ///
    /// Wraps libcurl and exposes a small set of request helpers for the
    /// "pure" (no prompt modification) and "jiezhu" (prefix-injection) flows
    /// in both blocking and SSE-streaming variants.
    class client {
    public:
        /// @brief Construct a client with the given options.
        /// @param options Connection settings, credentials and timeouts.
        explicit client(const client_options &options);
        /// @brief Construct a client with default options pointing at the
        /// public OpenAI endpoint.
        client() : client(client_options{"", "https://api.openai.com/v1", 300, "", ""}) {
        }

        /// @brief Issue a blocking OpenAI-style chat-completion request.
        ///
        /// Sends the request to @c {base_url}/chat/completions exactly as
        /// provided, without modifying any messages. Non-2xx responses
        /// raise a @c std::runtime_error carrying the HTTP status and body.
        /// @param request The request payload to send.
        /// @return Parsed response.
        /// @throw std::runtime_error On transport failure or non-2xx HTTP.
        [[nodiscard]] chat_completion_response chat_completions_create(const chat_completion_request &request) const;

        /// @brief Issue a blocking Anthropic Claude Messages-style request.
        ///
        /// System-role messages are lifted into a top-level @c system
        /// string, non-system messages are kept under @c messages, and the
        /// payload is POSTed to @c {base_url}/messages. When @c max_tokens
        /// is not set on the request, a default of 1024 is used (Anthropic
        /// requires the field).
        /// @param request The request payload to send.
        /// @return Parsed response.
        /// @throw std::runtime_error On transport failure or non-2xx HTTP.
        [[nodiscard]] chat_completion_response chat_completions_create_anthropic(
            const chat_completion_request &request) const;

        /// @brief Anthropic variant of @ref chat_completions_jiezhu with a
        /// custom prompt prefix.
        ///
        /// The @p prompt_prefix is prepended to every system-role message
        /// (matching the OpenAI variant's contract) before the payload is
        /// rewritten into Anthropic's format and sent.
        /// @param request The original request.
        /// @param prompt_prefix Custom prefix to prepend to system messages.
        /// @return Parsed response.
        /// @throw std::runtime_error On transport failure or non-2xx HTTP.
        [[nodiscard]] chat_completion_response chat_completions_jiezhu_anthropic(
            const chat_completion_request &request,
            const std::string &prompt_prefix) const;

        /// @brief Anthropic variant of @ref chat_completions_jiezhu using
        /// the built-in default prompt prefix.
        /// @param request The original request.
        /// @return Parsed response.
        /// @throw std::runtime_error On transport failure or non-2xx HTTP.
        [[nodiscard]] chat_completion_response chat_completions_jiezhu_anthropic(
            const chat_completion_request &request) const;

        /// @brief OpenAI variant of the jiezhu prompt-injection flow using
        /// the built-in default prompt prefix.
        ///
        /// Prepends the jiezhu prompt prefix to every system message in
        /// the request, then delegates to
        /// @ref chat_completions_create. Only available when the library
        /// is built with @c JIEZHU_ENABLE_JIEZHU_ABILITY.
        /// @param request The original request.
        /// @return Parsed response.
        /// @throw std::runtime_error When the jiezhu ability is disabled
        /// in this build, or on transport / HTTP failure.
        [[nodiscard]] chat_completion_response chat_completions_jiezhu(
            const chat_completion_request &request) const;

        /// @brief OpenAI variant of the jiezhu prompt-injection flow with
        /// a custom prompt prefix.
        ///
        /// Prepends @p prompt_prefix to every system message in the
        /// request, then delegates to
        /// @ref chat_completions_create. Only available when the library
        /// is built with @c JIEZHU_ENABLE_JIEZHU_ABILITY.
        /// @param request The original request.
        /// @param prompt_prefix Custom prefix to prepend to system messages.
        /// @return Parsed response.
        /// @throw std::runtime_error When the jiezhu ability is disabled
        /// in this build, or on transport / HTTP failure.
        [[nodiscard]] chat_completion_response chat_completions_jiezhu(
            const chat_completion_request &request,
            const std::string &prompt_prefix) const;

        /// @brief Streaming variant of the jiezhu prompt-injection flow
        /// using the built-in default prompt prefix.
        /// @param request The original request.
        /// @param on_event Callback invoked for every parsed SSE event.
        /// Returning @c false aborts the stream.
        /// @throw std::runtime_error On transport / HTTP failure.
        void chat_completions_stream_jiezhu(
            const chat_completion_request &request,
            const std::function<bool(const chat_completion_stream_event &)> &on_event) const;

        /// @brief Streaming variant of the jiezhu prompt-injection flow
        /// with a custom prompt prefix.
        /// @param request The original request.
        /// @param prompt_prefix Custom prefix to prepend to system messages.
        /// @param on_event Callback invoked for every parsed SSE event.
        /// Returning @c false aborts the stream.
        /// @throw std::runtime_error On transport / HTTP failure.
        void chat_completions_stream_jiezhu(
            const chat_completion_request &request,
            const std::string &prompt_prefix,
            const std::function<bool(const chat_completion_stream_event &)> &on_event) const;

        /// @brief Stream an OpenAI-style chat-completion response as
        /// server-sent events.
        ///
        /// The request is mutated to set @c stream = true, then POSTed to
        /// @c {base_url}/chat/completions. The @p on_event callback is
        /// invoked for each parsed SSE event.
        /// @param request Request payload. Will have @c stream set to true.
        /// @param on_event Callback invoked for every parsed SSE event.
        /// Returning @c false aborts the stream.
        /// @throw std::runtime_error On transport / HTTP failure.
        void chat_completions_stream(chat_completion_request request,
                                     const std::function<bool(const chat_completion_stream_event &)> &on_event)
        const;
        /// @brief Issue a blocking OpenAI Responses API request.
        ///
        /// POSTs the request to @c {base_url}/responses. Non-2xx
        /// responses raise a @c std::runtime_error.
        /// @param request The request payload.
        /// @return Parsed response.
        /// @throw std::runtime_error On transport failure or non-2xx HTTP.
        [[nodiscard]] response_response responses_create(const response_request &request) const;

        /// @brief Stream an OpenAI Responses API response as server-sent
        /// events.
        ///
        /// The request is mutated to set @c stream = true, then POSTed to
        /// @c {base_url}/responses. The @p on_event callback is invoked
        /// for each parsed SSE event.
        /// @param request Request payload. Will have @c stream set to true.
        /// @param on_event Callback invoked for every parsed SSE event.
        /// Returning @c false aborts the stream.
        /// @throw std::runtime_error On transport / HTTP failure.
        void responses_stream(response_request request,
                              const std::function<bool(const response_stream_event &)> &on_event) const;

        /// @brief Responses API variant of the jiezhu prompt-injection
        /// flow using the built-in default prompt prefix.
        ///
        /// Prepends the prefix to @c instructions and to any system
        /// messages in @c input_messages, then delegates to
        /// @ref responses_create. Only available when the library is
        /// built with @c JIEZHU_ENABLE_JIEZHU_ABILITY.
        [[nodiscard]] response_response responses_jiezhu(
            const response_request &request) const;

        /// @brief Responses API variant of the jiezhu prompt-injection
        /// flow with a custom prompt prefix.
        [[nodiscard]] response_response responses_jiezhu(
            const response_request &request,
            const std::string &prompt_prefix) const;

        /// @brief Streaming Responses API variant of jiezhu using the
        /// built-in default prompt prefix.
        void responses_stream_jiezhu(
            const response_request &request,
            const std::function<bool(const response_stream_event &)> &on_event) const;

        /// @brief Streaming Responses API variant of jiezhu with a
        /// custom prompt prefix.
        void responses_stream_jiezhu(
            const response_request &request,
            const std::string &prompt_prefix,
            const std::function<bool(const response_stream_event &)> &on_event) const;

        /// @brief Issue a blocking OpenAI Decisions API request.
        ///
        /// POSTs the request to @c {base_url}/decisions. This is a pure
        /// API call: the jiezhu prompt prefix is intentionally not
        /// injected because the Decisions API returns typed answers
        /// rather than generated text.
        /// @param request The request payload.
        /// @return Parsed response with typed answers.
        /// @throw std::runtime_error On transport failure or non-2xx HTTP.
        [[nodiscard]] decision_response decisions_create(const decision_request &request) const;

    private:
        /// @brief Return the latest non-empty user message text.
        /// @param request Chat-completion request to inspect.
        /// @return User text, or an empty string when unavailable.
        [[nodiscard]] static std::string latest_user_text(
            const chat_completion_request &request);

        /// @brief Return the latest non-empty user text from a Responses
        /// request.
        /// @param request Responses request to inspect.
        /// @return User text, or an empty string when unavailable.
        [[nodiscard]] static std::string latest_user_text(
            const response_request &request);

        /// @brief Ask the Decisions API whether @p user_text needs
        /// emotional support.
        ///
        /// Returns @c true when the experimental gate is disabled, when
        /// @p user_text is empty, or when the Decisions API cannot be
        /// reached, so the gate never blocks normal jiezhu behaviour.
        /// @param user_text Latest user message text.
        /// @return @c true to inject the jiezhu prefix, @c false to
        /// skip injection for this call.
        [[nodiscard]] bool user_needs_jiezhu(const std::string &user_text) const;

        /// @brief Configuration for this client.
        client_options options_;
    };
}; // namespace jie
