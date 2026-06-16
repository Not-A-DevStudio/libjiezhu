/**
 * @file chat.cpp
 * @brief Implementation of the @ref jie::client chat-completion helpers.
 *
 * Provides the blocking and SSE-streaming OpenAI / Anthropic transports
 * backed by libcurl, plus the prompt-prefix injection logic used by the
 * jiezhu (blessing) variants. When the build does not define
 * @c JIE_ENABLE_JIEZHU_ABLITY, the jiezhu helpers compile down to stubs
 * that throw a @c std::runtime_error.
 */
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string_view>

#include <curl/curl.h>
#include <jie/jiezhu.hpp>

namespace jie {
    /**
     * @brief Ensure @c curl_global_init has been called exactly once.
     *
     * Uses @c std::call_once to make the initialization thread-safe and
     * registers @c curl_global_cleanup with @c std::atexit so libcurl is
     * released when the program terminates.
     * @throw std::runtime_error If libcurl fails to initialize.
     */
    static void ensure_curl_global() {
        static std::once_flag flag;
        std::call_once(flag, []() {
            CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
            if (rc != CURLE_OK) {
                std::string msg = std::string("curl_global_init failed: ") + curl_easy_strerror(rc);
                throw std::runtime_error(msg);
            }
            std::atexit([]() { curl_global_cleanup(); });
        });
    }

    /**
     * @brief Return a copy of @p s with leading and trailing whitespace
     * (including @c \\\\r and @c \\\\n) removed.
     * @param s Input string.
     * @return Trimmed copy of @p s.
     */
    static inline std::string trim_copy(std::string s) {
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n' ||
                              std::isspace(static_cast<unsigned char>(s.back())))) {
            s.pop_back();
        }
        std::size_t i = 0;
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
        if (i > 0) s.erase(0, i);
        return s;
    }

    /**
     * @brief Test whether @p s starts with @p prefix.
     * @param s Candidate string.
     * @param prefix Prefix to look for.
     * @return @c true if @p s begins with @p prefix, @c false otherwise.
     */
    static inline bool starts_with(std::string_view s, std::string_view prefix) {
        return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
    }

    /// @brief Serialize a chat-completion request into the OpenAI JSON shape.
    /// @return JSON object containing @c model, @c stream, @c messages and
    /// any optional / extra fields.
    nlohmann::json chat_completion_request::to_json() const {
        nlohmann::json j = extra.is_object() ? extra : nlohmann::json::object();
        j["model"] = model;
        j["stream"] = stream;
        j["messages"] = nlohmann::json::array();
        for (std::size_t i = 0; i < messages.size(); ++i) {
            nlohmann::json m;
            m["role"] = messages[i].role;
            m["content"] = messages[i].content;
            j["messages"].push_back(m);
        }
        if (temperature.has_value()) j["temperature"] = *temperature;
        if (max_tokens.has_value()) j["max_tokens"] = *max_tokens;
        return j;
    }

    /**
     * @brief Convert a chat-completion request into the Anthropic
     * Messages-API JSON shape.
     *
     * All @c system-role messages are concatenated (separated by a blank
     * line) and lifted into the top-level @c system string, while
     * non-system messages are written to @c messages. When
     * @c max_tokens is not provided, a default of 1024 is injected since
     * Anthropic requires the field.
     * @param request The source chat-completion request.
     * @return JSON object ready to be sent to @c /messages.
     */
    static nlohmann::json to_anthropic_json(const chat_completion_request &request) {
        nlohmann::json j = request.extra.is_object() ? request.extra : nlohmann::json::object();
        j["model"] = request.model;
        j["stream"] = request.stream;

        std::string system_text;
        nlohmann::json anthropic_messages = nlohmann::json::array();
        for (const auto &msg: request.messages) {
            if (msg.role == "system") {
                if (!system_text.empty()) system_text += "\n\n";
                system_text += msg.content;
                continue;
            }
            nlohmann::json m;
            m["role"] = msg.role;
            m["content"] = msg.content;
            anthropic_messages.push_back(m);
        }

        if (!system_text.empty()) {
            j["system"] = system_text;
        }
        j["messages"] = anthropic_messages;

        if (request.temperature.has_value()) j["temperature"] = *request.temperature;
        if (request.max_tokens.has_value()) {
            j["max_tokens"] = *request.max_tokens;
        } else if (!j.contains("max_tokens")) {
            // Anthropic requires max_tokens in the request body.
            j["max_tokens"] = 1024;
        }

        return j;
    }

    /**
     * @brief Best-effort extraction of the assistant's first text payload.
     *
     * Tries the OpenAI shape (@c choices[0].message.content) first and
     * then the Anthropic shape (@c content[0].text). Returns an empty
     * string if neither pattern is matched.
     * @return The extracted text, or an empty string when unavailable.
     */
    std::string chat_completion_response::first_content() const {
        if (!raw.is_object()) return std::string();

        // OpenAI chat completions response shape: choices[0].message.content
        try {
            if (raw.contains("choices") && raw["choices"].is_array() && !raw["choices"].empty()) {
                const nlohmann::json &c0 = raw["choices"][0];
                if (c0.contains("message") && c0["message"].is_object()) {
                    const nlohmann::json &msg = c0["message"];
                    if (msg.contains("content") && msg["content"].is_string()) {
                        return msg["content"].get<std::string>();
                    }
                }
            }
        } catch (...) {
            // Best-effort only.
        }

        // Anthropic messages response shape: content[0].text
        try {
            if (raw.contains("content") && raw["content"].is_array() && !raw["content"].empty()) {
                const nlohmann::json &c0 = raw["content"][0];
                if (c0.is_object() && c0.contains("text") && c0["text"].is_string()) {
                    return c0["text"].get<std::string>();
                }
            }
        } catch (...) {
            // Best-effort only.
        }

        return std::string();
    }

    /// @brief Serialize a Responses API request into the expected JSON shape.
    /// @return JSON object with @c model, @c input, @c stream, and optional
    /// fields.
    nlohmann::json response_request::to_json() const {
        nlohmann::json j = extra.is_object() ? extra : nlohmann::json::object();
        j["model"] = model;
        j["stream"] = stream;
        if (!input_text.empty()) {
            j["input"] = input_text;
        } else if (!input_messages.empty()) {
            j["input"] = nlohmann::json::array();
            for (const auto &msg: input_messages) {
                nlohmann::json m;
                m["role"] = msg.role;
                m["content"] = msg.content;
                j["input"].push_back(m);
            }
        }
        if (!instructions.empty()) j["instructions"] = instructions;
        if (temperature.has_value()) j["temperature"] = *temperature;
        if (max_output_tokens.has_value()) j["max_output_tokens"] = *max_output_tokens;
        return j;
    }

    /// @brief Best-effort extraction of the first output text from a
    /// Responses API response.
    ///
    /// Tries the structured format (@c output[0].content[*].text) first
    /// and then the convenience field (@c output_text). Returns empty
    /// string when unavailable.
    /// @return The extracted text, or an empty string when unavailable.
    std::string response_response::first_content() const {
        if (!raw.is_object()) return std::string();

        // Responses API shape: output[0].content[*].text
        try {
            if (raw.contains("output") && raw["output"].is_array() && !raw["output"].empty()) {
                const nlohmann::json &first = raw["output"][0];
                if (first.contains("content") && first["content"].is_array()) {
                    for (const auto &part: first["content"]) {
                        if (part.contains("text") && part["text"].is_string()) {
                            return part["text"].get<std::string>();
                        }
                    }
                }
            }
        } catch (...) {
            // Best-effort only.
        }

        // Convenience shortcut field: output_text
        try {
            if (raw.contains("output_text") && raw["output_text"].is_string()) {
                return raw["output_text"].get<std::string>();
            }
        } catch (...) {
            // Best-effort only.
        }

        return std::string();
    }

    /**
     * @brief libcurl @c WRITEFUNCTION callback that appends the received
     * bytes to a caller-provided @c std::string.
     * @param ptr Buffer with the bytes just received.
     * @param size Size of each element in @p ptr.
     * @param nmemb Number of elements in @p ptr.
     * @param userdata Pointer to a @c std::string to append to.
     * @return Number of bytes "handled" (the entire @p size*@p nmemb).
     */
    static size_t write_to_string(void *ptr, size_t size, size_t nmemb, void *userdata) {
        const size_t n = size * nmemb;
        std::string *out = static_cast<std::string *>(userdata);
        out->append(static_cast<const char *>(ptr), n);
        return n;
    }

    /**
     * @brief Populate the OpenAI-style request headers on @p headers.
     *
     * Adds @c Content-Type, @c Authorization (when @c api_key is set) and
     * the optional @c OpenAI-Organization / @c OpenAI-Project headers.
     * @param options Client options holding credentials.
     * @param headers Pointer to a libcurl slist that receives the new
     * entries. The caller retains ownership and must free the list.
     */
    static void set_common_headers(const client_options &options, struct curl_slist **headers) {
        *headers = curl_slist_append(*headers, "Content-Type: application/json");

        if (!options.api_key.empty()) {
            std::string auth = std::string("Authorization: Bearer ") + options.api_key;
            *headers = curl_slist_append(*headers, auth.c_str());
        }
        if (!options.organization.empty()) {
            std::string org = std::string("OpenAI-Organization: ") + options.organization;
            *headers = curl_slist_append(*headers, org.c_str());
        }
        if (!options.project.empty()) {
            std::string proj = std::string("OpenAI-Project: ") + options.project;
            *headers = curl_slist_append(*headers, proj.c_str());
        }
    }

    /**
     * @brief Populate the Anthropic-style request headers on @p headers.
     *
     * Adds @c Content-Type, the @c anthropic-version marker and the
     * @c x-api-key credential header (when @c api_key is set).
     * @param options Client options holding credentials.
     * @param headers Pointer to a libcurl slist that receives the new
     * entries. The caller retains ownership and must free the list.
     */
    static void set_anthropic_headers(const client_options &options, struct curl_slist **headers) {
        *headers = curl_slist_append(*headers, "Content-Type: application/json");
        *headers = curl_slist_append(*headers, "anthropic-version: 2023-06-01");

        if (!options.api_key.empty()) {
            std::string key = std::string("x-api-key: ") + options.api_key;
            *headers = curl_slist_append(*headers, key.c_str());
        }
    }

    /**
     * @brief Raise a @c std::runtime_error describing an HTTP failure.
     * @param http_code The non-2xx HTTP status returned by the server.
     * @param body The response body, if any. Empty bodies are allowed.
     * @throw std::runtime_error Always thrown.
     */
    static void throw_http_error(long http_code, const std::string &body) {
        std::ostringstream oss;
        oss << "HTTP " << http_code;
        if (!body.empty()) {
            oss << ": " << body;
        }
        throw std::runtime_error(oss.str());
    }

    /// @brief Construct a client and ensure libcurl is initialized.
    /// @param options Connection settings, credentials and timeouts.
    client::client(const client_options &options) : options_(options) {
        ensure_curl_global();
    }


    /**
     * @brief Blocking OpenAI-style chat-completion call.
     *
     * POSTs the request to @c {base_url}/chat/completions and parses the
     * JSON response. Non-2xx responses are converted to a
     * @c std::runtime_error carrying the status and body.
     * @param request The request to send.
     * @return Parsed response.
     * @throw std::runtime_error On transport failure or non-2xx HTTP.
     */
    chat_completion_response client::chat_completions_create(const chat_completion_request &request) const {
        ensure_curl_global();

        CURL *curl = curl_easy_init();
        if (!curl) throw std::runtime_error("curl_easy_init failed");

        struct curl_slist *headers = NULL;
        std::string response_body;
        long http_code = 0;

        try {
            const std::string url = options_.base_url + "/chat/completions";
            const std::string payload = request.to_json().dump();

            set_common_headers(options_, &headers);
            headers = curl_slist_append(headers, "Accept: application/json");

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, options_.timeout_seconds);

            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_string);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);

            CURLcode rc = curl_easy_perform(curl);
            if (rc != CURLE_OK) {
                std::string msg = std::string("curl_easy_perform failed: ") + curl_easy_strerror(rc);
                throw std::runtime_error(msg);
            }

            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            if (http_code < 200 || http_code >= 300) {
                throw_http_error(http_code, response_body);
            }

            chat_completion_response resp;
            resp.raw = nlohmann::json::parse(response_body);
            if (resp.raw.contains("id") && resp.raw["id"].is_string()) resp.id = resp.raw["id"].get<std::string>();
            if (resp.raw.contains("model") && resp.raw["model"].is_string())
                resp.model = resp.raw["model"].get<std::string>();
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            return resp;
        } catch (...) {
            if (headers) curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            throw;
        }
    }

    /**
     * @brief Blocking Anthropic Claude Messages-style call.
     *
     * Rewrites the request with @ref to_anthropic_json and POSTs it to
     * @c {base_url}/messages. Non-2xx responses are converted to a
     * @c std::runtime_error carrying the status and body.
     * @param request The request to send.
     * @return Parsed response.
     * @throw std::runtime_error On transport failure or non-2xx HTTP.
     */
    chat_completion_response client::chat_completions_create_anthropic(const chat_completion_request &request) const {
        ensure_curl_global();

        CURL *curl = curl_easy_init();
        if (!curl) throw std::runtime_error("curl_easy_init failed");

        struct curl_slist *headers = NULL;
        std::string response_body;
        long http_code = 0;

        try {
            const std::string url = options_.base_url + "/messages";
            const std::string payload = to_anthropic_json(request).dump();

            set_anthropic_headers(options_, &headers);
            headers = curl_slist_append(headers, "Accept: application/json");

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, options_.timeout_seconds);

            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_string);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);

            CURLcode rc = curl_easy_perform(curl);
            if (rc != CURLE_OK) {
                std::string msg = std::string("curl_easy_perform failed: ") + curl_easy_strerror(rc);
                throw std::runtime_error(msg);
            }

            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            if (http_code < 200 || http_code >= 300) {
                throw_http_error(http_code, response_body);
            }

            chat_completion_response resp;
            resp.raw = nlohmann::json::parse(response_body);
            if (resp.raw.contains("id") && resp.raw["id"].is_string()) resp.id = resp.raw["id"].get<std::string>();
            if (resp.raw.contains("model") && resp.raw["model"].is_string())
                resp.model = resp.raw["model"].get<std::string>();
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            return resp;
        } catch (...) {
            if (headers) curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            throw;
        }
    }

    /// @brief Internal state shared by the SSE streaming callbacks.
    struct stream_state {
        /// @brief Incomplete-line carry-over between libcurl write calls.
        std::string buffer;
        /// @brief User-supplied per-event callback. May be @c nullptr.
        std::function<bool(const chat_completion_stream_event &)> callback;
        /// @brief Set to @c true when the user callback returned @c false.
        bool canceled = false;
        /// @brief Set to @c true once the @c [DONE] sentinel has been
        /// emitted to the callback.
        bool saw_done = false;
    };

    /**
     * @brief Parse a single SSE @c data: line and dispatch it to the
     * streaming callback.
     *
     * The terminal @c [DONE] sentinel produces an event with
     * @c done == @c true. Malformed JSON lines are silently skipped.
     * @param line One line of the SSE stream (without the trailing newline).
     * @param st Shared streaming state, updated in-place.
     * @return @c true to keep reading, @c false to abort the transfer.
     */
    static bool parse_and_emit_sse_line(const std::string &line, stream_state &st) {
        // Returns true to continue, false to cancel.
        const std::string s = trim_copy(line);
        if (s.empty()) return true;

        if (!starts_with(s, "data:")) {
            return true;
        }

        std::string data = trim_copy(s.substr(5));
        if (data.empty()) return true;

        if (data == "[DONE]") {
            chat_completion_stream_event ev;
            ev.done = true;
            st.saw_done = true;
            return st.callback ? st.callback(ev) : true;
        }

        chat_completion_stream_event ev;
        try {
            ev.raw = nlohmann::json::parse(data);

            if (ev.raw.contains("choices") && ev.raw["choices"].is_array() && !ev.raw["choices"].empty()) {
                const nlohmann::json &c0 = ev.raw["choices"][0];
                if (c0.contains("delta") && c0["delta"].is_object()) {
                    const nlohmann::json &d = c0["delta"];
                    if (d.contains("role") && d["role"].is_string()) ev.delta_role = d["role"].get<std::string>();
                    if (d.contains("content") && d["content"].is_string())
                        ev.delta_content = d["content"].get<std::string>();
                }
                if (c0.contains("finish_reason") && c0["finish_reason"].is_string()) {
                    ev.finish_reason = c0["finish_reason"].get<std::string>();
                    if (!ev.finish_reason.empty()) ev.done = true;
                }
            }
        } catch (...) {
            // If parsing fails, ignore this line (best-effort streaming).
            return true;
        }

        return st.callback ? st.callback(ev) : true;
    }

    /**
     * @brief libcurl @c WRITEFUNCTION callback used for SSE streaming.
     *
     * Appends the received bytes to @c stream_state::buffer, drains any
     * complete newline-terminated lines through
     * @ref parse_and_emit_sse_line, and aborts the transfer (by returning
     * @c 0) when the user callback returns @c false or a @c [DONE]
     * sentinel has been delivered.
     * @param ptr Buffer with the bytes just received.
     * @param size Size of each element in @p ptr.
     * @param nmemb Number of elements in @p ptr.
     * @param userdata Pointer to a @c stream_state.
     * @return The number of bytes "handled", or @c 0 to abort the
     * transfer on cancellation / completion.
     */
    static size_t write_stream_sse(void *ptr, size_t size, size_t nmemb, void *userdata) {
        const size_t n = size * nmemb;
        auto *st = static_cast<stream_state *>(userdata);
        if (!st) return 0;

        st->buffer.append(static_cast<const char *>(ptr), n);

        // Process complete lines. SSE events are line-based (data: ...\n).
        while (true) {
            const std::size_t pos = st->buffer.find('\n');
            if (pos == std::string::npos) break;
            std::string line = st->buffer.substr(0, pos);
            st->buffer.erase(0, pos + 1);

            bool cont = parse_and_emit_sse_line(line, *st);
            if (!cont) {
                st->canceled = true;
                return 0; // abort transfer
            }
            if (st->saw_done) {
                // We can keep reading until curl finishes, but stopping early is OK.
                // Abort politely by returning 0.
                return 0;
            }
        }

        return n;
    }

    /**
     * @brief OpenAI-style SSE streaming chat-completion call.
     *
     * Sets @c request.stream = @c true, POSTs the request to
     * @c {base_url}/chat/completions and forwards each parsed SSE event
     * to @p on_event. The transfer is aborted (politely) if the callback
     * returns @c false or once the @c [DONE] sentinel is seen.
     * @param request The request payload (will be copied and mutated).
     * @param on_event Per-event callback. Returning @c false cancels the
     * stream.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    void client::chat_completions_stream(chat_completion_request request,
                                         const std::function<bool(const chat_completion_stream_event &)> &on_event)
    const {
        ensure_curl_global();
        request.stream = true;

        CURL *curl = curl_easy_init();
        if (!curl) throw std::runtime_error("curl_easy_init failed");

        struct curl_slist *headers = nullptr;
        long http_code = 0;

        stream_state st;
        st.callback = on_event;

        try {
            const std::string url = options_.base_url + "/chat/completions";
            const std::string payload = request.to_json().dump();

            set_common_headers(options_, &headers);
            headers = curl_slist_append(headers, "Accept: text/event-stream");

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, options_.timeout_seconds);

            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_stream_sse);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &st);

            // Capture error responses (non-2xx) into a string (curl still writes to WRITEFUNCTION).
            // We'll handle status code after perform.

            CURLcode rc = curl_easy_perform(curl);
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

            if (st.canceled || st.saw_done) {
                // Treat as graceful stop.
                curl_slist_free_all(headers);
                curl_easy_cleanup(curl);
                return;
            }

            if (rc != CURLE_OK) {
                std::string msg = std::string("curl_easy_perform failed: ") + curl_easy_strerror(rc);
                throw std::runtime_error(msg);
            }

            if (http_code < 200 || http_code >= 300) {
                throw_http_error(http_code, st.buffer);
            }

            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        } catch (...) {
            if (headers) curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            throw;
        }
    }
  
    /**
     * @brief Blocking OpenAI Responses API call.
     *
     * POSTs the request to @c {base_url}/responses and parses the JSON
     * response. Non-2xx responses are converted to a
     * @c std::runtime_error carrying the status and body.
     * @param request The request to send.
     * @return Parsed response.
     * @throw std::runtime_error On transport failure or non-2xx HTTP.
     */
    response_response client::responses_create(const response_request &request) const {
        ensure_curl_global();

        CURL *curl = curl_easy_init();
        if (!curl) throw std::runtime_error("curl_easy_init failed");

        struct curl_slist *headers = NULL;
        std::string response_body;
        long http_code = 0;

        try {
            const std::string url = options_.base_url + "/responses";
            const std::string payload = request.to_json().dump();

            set_common_headers(options_, &headers);
            headers = curl_slist_append(headers, "Accept: application/json");

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, options_.timeout_seconds);

            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_string);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);

            CURLcode rc = curl_easy_perform(curl);
            if (rc != CURLE_OK) {
                std::string msg = std::string("curl_easy_perform failed: ") + curl_easy_strerror(rc);
                throw std::runtime_error(msg);
            }

            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            if (http_code < 200 || http_code >= 300) {
                throw_http_error(http_code, response_body);
            }

            response_response resp;
            resp.raw = nlohmann::json::parse(response_body);
            if (resp.raw.contains("id") && resp.raw["id"].is_string()) resp.id = resp.raw["id"].get<std::string>();
            if (resp.raw.contains("model") && resp.raw["model"].is_string())
                resp.model = resp.raw["model"].get<std::string>();
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            return resp;
        } catch (...) {
            if (headers) curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            throw;
        }
    }

    // ───── Responses API: SSE streaming ─────

    /**
     * @brief Internal state for Responses API SSE streaming.
     */
    struct response_stream_state {
        /// @brief Incomplete-line carry-over between libcurl write calls.
        std::string buffer;
        /// @brief Current SSE event type accumulated from @c event: lines.
        std::string current_event_type;
        /// @brief User-supplied per-event callback.
        std::function<bool(const response_stream_event &)> callback;
        /// @brief Set to @c true when the user callback returned @c false.
        bool canceled = false;
        /// @brief Set to @c true once a terminal event has been seen.
        bool saw_done = false;
    };

    /**
     * @brief Parse a single SSE line in the Responses API format.
     *
     * Handles @c event: and @c data: lines. When an @c event: line is
     * followed by a @c data: line, a @ref response_stream_event is
     * constructed and dispatched to the user callback.
     * @param line One line of the SSE stream (without trailing newline).
     * @param st Shared streaming state, updated in-place.
     * @return @c true to keep reading, @c false to abort.
     */
    static bool parse_and_emit_response_sse_line(const std::string &line, response_stream_state &st) {
        const std::string s = trim_copy(line);
        if (s.empty()) return true;

        if (starts_with(s, "event:")) {
            st.current_event_type = trim_copy(s.substr(6));
            return true;
        }

        if (!starts_with(s, "data:")) {
            return true;
        }

        std::string data = trim_copy(s.substr(5));
        if (data.empty()) return true;

        response_stream_event ev;
        ev.event_type = st.current_event_type;
        st.current_event_type.clear();

        try {
            ev.raw = nlohmann::json::parse(data);
            if (ev.raw.contains("delta") && ev.raw["delta"].is_string()) {
                ev.delta = ev.raw["delta"].get<std::string>();
            }
            if (ev.event_type == "response.completed" ||
                ev.event_type == "error" ||
                (ev.raw.contains("type") && ev.raw["type"].is_string() &&
                 (ev.raw["type"].get<std::string>() == "response.completed" ||
                  ev.raw["type"].get<std::string>() == "error"))) {
                ev.done = true;
                st.saw_done = true;
            }
        } catch (...) {
            return true;
        }

        return st.callback ? st.callback(ev) : true;
    }

    /**
     * @brief libcurl @c WRITEFUNCTION callback for Responses API SSE
     * streaming.
     *
     * Appends received bytes to the buffer, drains complete newline-
     * terminated lines through @ref parse_and_emit_response_sse_line,
     * and aborts when the callback returns @c false or a terminal event
     * has been delivered.
     */
    static size_t write_response_stream_sse(void *ptr, size_t size, size_t nmemb, void *userdata) {
        const size_t n = size * nmemb;
        auto *st = static_cast<response_stream_state *>(userdata);
        if (!st) return 0;

        st->buffer.append(static_cast<const char *>(ptr), n);

        while (true) {
            const std::size_t pos = st->buffer.find('\n');
            if (pos == std::string::npos) break;
            std::string line = st->buffer.substr(0, pos);
            st->buffer.erase(0, pos + 1);

            bool cont = parse_and_emit_response_sse_line(line, *st);
            if (!cont) {
                st->canceled = true;
                return 0;
            }
            if (st->saw_done) {
                return 0;
            }
        }

        return n;
    }

    /**
     * @brief Responses API SSE streaming call.
     *
     * Sets @c request.stream = @c true, POSTs to @c {base_url}/responses
     * and forwards each parsed SSE event to @p on_event. The transfer is
     * aborted if the callback returns @c false or once a terminal event
     * is seen.
     * @param request The request payload (will be copied and mutated).
     * @param on_event Per-event callback. Returning @c false cancels.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    void client::responses_stream(response_request request,
                                  const std::function<bool(const response_stream_event &)> &on_event) const {
        ensure_curl_global();
        request.stream = true;

        CURL *curl = curl_easy_init();
        if (!curl) throw std::runtime_error("curl_easy_init failed");

        struct curl_slist *headers = nullptr;
        long http_code = 0;

        response_stream_state st;
        st.callback = on_event;

        try {
            const std::string url = options_.base_url + "/responses";
            const std::string payload = request.to_json().dump();

            set_common_headers(options_, &headers);
            headers = curl_slist_append(headers, "Accept: text/event-stream");

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(payload.size()));
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, options_.timeout_seconds);

            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_response_stream_sse);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &st);

            CURLcode rc = curl_easy_perform(curl);
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

            if (st.canceled || st.saw_done) {
                curl_slist_free_all(headers);
                curl_easy_cleanup(curl);
                return;
            }

            if (rc != CURLE_OK) {
                std::string msg = std::string("curl_easy_perform failed: ") + curl_easy_strerror(rc);
                throw std::runtime_error(msg);
            }

            if (http_code < 200 || http_code >= 300) {
                throw_http_error(http_code, st.buffer);
            }

            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
        } catch (...) {
            if (headers) curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            throw;
        }
    }

#ifdef JIE_ENABLE_JIEZHU_ABLITY


    /// @brief Cached copy of the default jiezhu prompt prefix.
    const auto _default_prompt_prefix = PROMPT_PREFIX;

    /**
     * @brief OpenAI jiezhu variant using the built-in default prefix.
     *
     * Prepends @ref _default_prompt_prefix to every system message in the
     * request, then delegates to @ref chat_completions_create.
     * @param request The original request.
     * @return Parsed response.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    chat_completion_response client::chat_completions_jiezhu(
        const chat_completion_request &request)
    const {
        auto temp = request;
        for (auto &message: temp.messages) {
            if (message.role != "system") continue;
            message.content = _default_prompt_prefix + message.content;
        }
        return chat_completions_create(temp);
    };


    /**
     * @brief OpenAI jiezhu variant using a custom prefix.
     *
     * Prepends @p prompt_prefix to every system message in the request,
     * then delegates to @ref chat_completions_create.
     * @param request The original request.
     * @param prompt_prefix Custom prefix to prepend to system messages.
     * @return Parsed response.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    chat_completion_response client::chat_completions_jiezhu(
        const chat_completion_request &request,
        const std::string &prompt_prefix)
    const {
        auto temp = request;
        for (auto &message: temp.messages) {
            if (message.role != "system") continue;
            message.content = prompt_prefix + message.content;
        }
        return chat_completions_create(temp);
    }

    /**
     * @brief Anthropic jiezhu variant using the built-in default prefix.
     *
     * Prepends @ref _default_prompt_prefix to every system message in the
     * request, then delegates to @ref chat_completions_create_anthropic.
     * @param request The original request.
     * @return Parsed response.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    chat_completion_response client::chat_completions_jiezhu_anthropic(
        const chat_completion_request &request)
    const
    {
        auto temp = request;
        for (auto &message: temp.messages) {
            if (message.role != "system") continue;
            message.content = _default_prompt_prefix + message.content;
        }
        return chat_completions_create_anthropic(temp);
    }

    /**
     * @brief Anthropic jiezhu variant using a custom prefix.
     *
     * Prepends @p prompt_prefix to every system message in the request,
     * then delegates to @ref chat_completions_create_anthropic.
     * @param request The original request.
     * @param prompt_prefix Custom prefix to prepend to system messages.
     * @return Parsed response.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    chat_completion_response client::chat_completions_jiezhu_anthropic(
        const chat_completion_request &request,
        const std::string &prompt_prefix)
    const {
        auto temp = request;
        for (auto &message: temp.messages) {
            if (message.role != "system") continue;
            message.content = prompt_prefix + message.content;
        }
        return chat_completions_create_anthropic(temp);
    }


    /**
     * @brief Streaming OpenAI jiezhu variant with the default prefix.
     *
     * Prepends @ref _default_prompt_prefix to every system message in the
     * request, then delegates to @ref chat_completions_stream.
     * @param request The original request.
     * @param on_event Per-event callback. Returning @c false cancels the
     * stream.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    void client::chat_completions_stream_jiezhu(
        const chat_completion_request &request,
        const std::function<bool(const chat_completion_stream_event &)> &on_event)
    const {
        auto temp = request;
        for (auto &message: temp.messages) {
            if (message.role != "system") continue;
            message.content = _default_prompt_prefix + message.content;
        }
        chat_completions_stream(temp, on_event);
    }

    /**
     * @brief Streaming OpenAI jiezhu variant with a custom prefix.
     *
     * Prepends @p prompt_prefix to every system message in the request,
     * then delegates to @ref chat_completions_stream.
     * @param request The original request.
     * @param prompt_prefix Custom prefix to prepend to system messages.
     * @param on_event Per-event callback. Returning @c false cancels the
     * stream.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    void client::chat_completions_stream_jiezhu(
        const chat_completion_request &request,
        const std::string &prompt_prefix,
        const std::function<bool(const chat_completion_stream_event &)> &on_event)
    const {
        auto temp = request;
        for (auto &message: temp.messages) {
            if (message.role != "system") continue;
            message.content = prompt_prefix + message.content;
        }
        chat_completions_stream(temp, on_event);
    }

    // ───── Responses API jiezhu variants (enabled) ─────

    /**
     * @brief Helper: prepend a prompt prefix to @c instructions and to
     * any system messages in @c input_messages.
     * @param req The request to mutate.
     * @param prefix The prefix to prepend.
     */
    static void inject_jiezhu_prefix(response_request &req, const std::string &prefix) {
        if (!req.instructions.empty()) {
            req.instructions = prefix + req.instructions;
        } else {
            req.instructions = prefix;
        }
        for (auto &msg: req.input_messages) {
            if (msg.role == "system") {
                msg.content = prefix + msg.content;
            }
        }
    }

    /**
     * @brief Responses API jiezhu variant using the built-in default
     * prefix.
     *
     * Prepends @ref _default_prompt_prefix to @c instructions and to
     * any system messages in @c input_messages, then delegates to
     * @ref responses_create.
     * @param request The original request.
     * @return Parsed response.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    response_response client::responses_jiezhu(
        const response_request &request)
    const {
        auto temp = request;
        inject_jiezhu_prefix(temp, _default_prompt_prefix);
        return responses_create(temp);
    }

    /**
     * @brief Responses API jiezhu variant using a custom prefix.
     *
     * Prepends @p prompt_prefix to @c instructions and to any system
     * messages in @c input_messages, then delegates to
     * @ref responses_create.
     * @param request The original request.
     * @param prompt_prefix Custom prefix to prepend.
     * @return Parsed response.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    response_response client::responses_jiezhu(
        const response_request &request,
        const std::string &prompt_prefix)
    const {
        auto temp = request;
        inject_jiezhu_prefix(temp, prompt_prefix);
        return responses_create(temp);
    }

    /**
     * @brief Streaming Responses API jiezhu variant with the default
     * prefix.
     *
     * Prepends @ref _default_prompt_prefix, then delegates to
     * @ref responses_stream.
     * @param request The original request.
     * @param on_event Per-event callback.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    void client::responses_stream_jiezhu(
        const response_request &request,
        const std::function<bool(const response_stream_event &)> &on_event)
    const {
        auto temp = request;
        inject_jiezhu_prefix(temp, _default_prompt_prefix);
        responses_stream(temp, on_event);
    }

    /**
     * @brief Streaming Responses API jiezhu variant with a custom
     * prefix.
     *
     * Prepends @p prompt_prefix, then delegates to
     * @ref responses_stream.
     * @param request The original request.
     * @param prompt_prefix Custom prefix.
     * @param on_event Per-event callback.
     * @throw std::runtime_error On transport / HTTP failure.
     */
    void client::responses_stream_jiezhu(
        const response_request &request,
        const std::string &prompt_prefix,
        const std::function<bool(const response_stream_event &)> &on_event)
    const {
        auto temp = request;
        inject_jiezhu_prefix(temp, prompt_prefix);
        responses_stream(temp, on_event);
    }
#endif
#ifndef JIE_ENABLE_JIEZHU_ABLITY
    /**
     * @brief Stub used when jiezhu ability is disabled at build time.
     * @throw std::runtime_error Always, with a message explaining the
     * build configuration.
     */
    chat_completion_response client::chat_completions_jiezhu(
        const chat_completion_request &request)
    const {
        (void)request;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    };

    /**
     * @brief Stub used when jiezhu ability is disabled at build time.
     * @throw std::runtime_error Always, with a message explaining the
     * build configuration.
     */
    chat_completion_response client::chat_completions_jiezhu(
        const chat_completion_request &request,
        const std::string &prompt_prefix)
    const {
        (void)prompt_prefix;
        (void)request;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    }

    /**
     * @brief Stub used when jiezhu ability is disabled at build time.
     * @throw std::runtime_error Always, with a message explaining the
     * build configuration.
     */
    void client::chat_completions_stream_jiezhu(
        const chat_completion_request &request,
        const std::function<bool(const chat_completion_stream_event &)> &
        on_event)
    const {
        (void)request;
        (void)on_event;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    };

    /**
     * @brief Stub used when jiezhu ability is disabled at build time.
     * @throw std::runtime_error Always, with a message explaining the
     * build configuration.
     */
    void client::chat_completions_stream_jiezhu(
        const chat_completion_request &request, const std::string &prompt_prefix,
        const std::function<bool(const chat_completion_stream_event &)> &
        on_event)
    const {
        (void)request;
        (void)prompt_prefix;
        (void)on_event;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    };
    /**
     * @brief Stub used when jiezhu ability is disabled at build time.
     * @throw std::runtime_error Always, with a message explaining the
     * build configuration.
     */
    chat_completion_response client::chat_completions_jiezhu_anthropic(
        const chat_completion_request &request)
    const {
        (void)request;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    }

    /**
     * @brief Stub used when jiezhu ability is disabled at build time.
     * @throw std::runtime_error Always, with a message explaining the
     * build configuration.
     */
    chat_completion_response client::chat_completions_jiezhu_anthropic(
    const chat_completion_request &request,
    const std::string &prompt_prefix)
const {
        (void)request;
        (void)prompt_prefix;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    }

    // ───── Responses API jiezhu stubs ─────

    response_response client::responses_jiezhu(
        const response_request &request)
    const {
        (void)request;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    }

    response_response client::responses_jiezhu(
        const response_request &request,
        const std::string &prompt_prefix)
    const {
        (void)request;
        (void)prompt_prefix;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    }

    void client::responses_stream_jiezhu(
        const response_request &request,
        const std::function<bool(const response_stream_event &)> &on_event)
    const {
        (void)request;
        (void)on_event;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    }

    void client::responses_stream_jiezhu(
        const response_request &request,
        const std::string &prompt_prefix,
        const std::function<bool(const response_stream_event &)> &on_event)
    const {
        (void)request;
        (void)prompt_prefix;
        (void)on_event;
        throw std::runtime_error("jiezhu ability is not enabled in this build");
    }
#endif
} // namespace jie
