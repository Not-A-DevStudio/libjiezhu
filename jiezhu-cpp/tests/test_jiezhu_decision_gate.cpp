/**
 * @file test_jiezhu_decision_gate.cpp
 * @brief Catch2 tests for the experimental Decisions API jiezhu gate.
 *
 * These tests are only built when @c JIEZHU_ENABLE_JIEZHU_ABILITY is
 * enabled. They verify that the jiezhu variants consult the Decisions
 * API, skip prompt injection when the user does not need emotional
 * support, and fail open when the decision request fails.
 */
#include <catch2/catch_test_macros.hpp>

#include <jie/chat.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>

#include "test_http_server.hpp"

namespace {
	/// @brief Return the body portion of a raw HTTP request.
	/// @param request Raw request bytes captured by the test server.
	/// @return Everything after the blank line that ends the headers.
	std::string request_body(const std::string& request) {
		const std::string delimiter = "\r\n\r\n";
		const std::size_t pos = request.find(delimiter);
		if (pos == std::string::npos) return "";
		return request.substr(pos + delimiter.size());
	}

	/// @brief Build a Decisions API response with one predicate answer.
	/// @param probability Probability reported for the predicate.
	/// @return JSON body string.
	std::string decision_body(double probability) {
		return std::string("{\"model\":\"gpt-6-luna\",\"answers\":[")
			+ "{\"type\":\"predicate\",\"name\":\"needs_emotional_support\","
			+ "\"probability\":" + std::to_string(probability) + "}]}";
	}

	/// @brief Standard chat-completion response body.
	/// @return JSON body string.
	std::string chat_body() {
		return R"({"id":"chat-id","model":"test-model","choices":[{"message":{"content":"hello"}}]})";
	}
}

/// @brief The gate injects the prefix and uses the latest user message.
TEST_CASE("decision gate injects jiezhu when support is needed") {
	std::atomic<int> decision_calls{0};
	std::mutex request_mutex;
	std::string decision_request;

	test_support::tiny_http_server server([&](const std::string& request) {
		if (test_support::starts_with(request, "POST /v1/decisions ")) {
			{
				std::lock_guard<std::mutex> lock(request_mutex);
				decision_request = request;
				++decision_calls;
			}
			return test_support::build_http_response(
				"200 OK", "application/json", decision_body(0.91));
		}
		return test_support::build_http_response("200 OK", "application/json", chat_body());
	});

	jie::client_options opt;
	opt.api_key = "test-key";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;
	opt.experimental_decision_gate = true;

	jie::client c(opt);
	jie::chat_completion_request r;
	r.model = "gpt-test";
	r.messages = {
		{"system", "BASE"},
		{"user", "I feel lonely and exhausted."},
	};

	const jie::chat_completion_response resp = c.chat_completions_jiezhu(r);
	REQUIRE(resp.first_content() == "hello");
	REQUIRE(decision_calls.load() == 1);

	{
		std::lock_guard<std::mutex> lock(request_mutex);
		const std::string body = request_body(decision_request);
		const nlohmann::json json = nlohmann::json::parse(body);
		REQUIRE(json["model"] == "gpt-6-luna");
		REQUIRE(json["input"] == "I feel lonely and exhausted.");
		REQUIRE(json["questions"][0]["type"] == "predicate");
		REQUIRE(json["questions"][0]["name"] == "needs_emotional_support");
		REQUIRE(json["questions"][0]["instructions"] ==
			"Determine whether the user's latest message is seeking emotional "
			"support, comfort, empathy, or companionship. Return true if the user "
			"is expressing negative emotions, venting about a problem, seeking "
			"understanding, or asking to be treated gently. Return false if the "
			"user is only asking for facts, troubleshooting a technical issue, "
			"giving a task instruction, or making ordinary small talk.");
	}

	const std::string chat_request = server.last_request();
	REQUIRE(test_support::starts_with(chat_request, "POST /v1/chat/completions "));
	const std::string chat = request_body(chat_request);
	REQUIRE(chat.find("\"content\":\"BASE\"") == std::string::npos);
}

/// @brief The gate skips prefix injection when support is not needed.
TEST_CASE("decision gate skips jiezhu when support is not needed") {
	std::atomic<int> decision_calls{0};

	test_support::tiny_http_server server([&](const std::string& request) {
		if (test_support::starts_with(request, "POST /v1/decisions ")) {
			++decision_calls;
			return test_support::build_http_response(
				"200 OK", "application/json", decision_body(0.08));
		}
		return test_support::build_http_response("200 OK", "application/json", chat_body());
	});

	jie::client_options opt;
	opt.api_key = "test-key";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;
	opt.experimental_decision_gate = true;

	jie::client c(opt);
	jie::chat_completion_request r;
	r.model = "gpt-test";
	r.messages = {
		{"system", "BASE"},
		{"user", "How do I sort a vector in C++?"},
	};

	const jie::chat_completion_response resp = c.chat_completions_jiezhu(r);
	REQUIRE(resp.first_content() == "hello");
	REQUIRE(decision_calls.load() == 1);

	const std::string chat = request_body(server.last_request());
	REQUIRE(chat.find("\"content\":\"BASE\"") != std::string::npos);
}

/// @brief Disabled gate never issues a Decisions request.
TEST_CASE("decision gate is disabled by default") {
	std::atomic<int> decision_calls{0};

	test_support::tiny_http_server server([&](const std::string& request) {
		if (test_support::starts_with(request, "POST /v1/decisions ")) {
			++decision_calls;
			return test_support::build_http_response(
				"200 OK", "application/json", decision_body(0.0));
		}
		return test_support::build_http_response("200 OK", "application/json", chat_body());
	});

	jie::client_options opt;
	opt.api_key = "test-key";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;
	REQUIRE(opt.experimental_decision_gate == false);

	jie::client c(opt);
	jie::chat_completion_request r;
	r.model = "gpt-test";
	r.messages = {
		{"system", "BASE"},
		{"user", "hello"},
	};

	(void)c.chat_completions_jiezhu(r);
	REQUIRE(decision_calls.load() == 0);
	const std::string chat = request_body(server.last_request());
	REQUIRE(chat.find("\"content\":\"BASE\"") == std::string::npos);
}

/// @brief Decisions API failures fail open and keep jiezhu enabled.
TEST_CASE("decision gate fails open on Decisions API errors") {
	std::atomic<int> decision_calls{0};

	test_support::tiny_http_server server([&](const std::string& request) {
		if (test_support::starts_with(request, "POST /v1/decisions ")) {
			++decision_calls;
			return test_support::build_http_response(
				"500 Internal Server Error", "application/json", "{\"error\":\"boom\"}");
		}
		return test_support::build_http_response("200 OK", "application/json", chat_body());
	});

	jie::client_options opt;
	opt.api_key = "test-key";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;
	opt.experimental_decision_gate = true;

	jie::client c(opt);
	jie::chat_completion_request r;
	r.model = "gpt-test";
	r.messages = {
		{"system", "BASE"},
		{"user", "I feel sad."},
	};

	const jie::chat_completion_response resp = c.chat_completions_jiezhu(r);
	REQUIRE(resp.first_content() == "hello");
	REQUIRE(decision_calls.load() == 1);
	const std::string chat = request_body(server.last_request());
	REQUIRE(chat.find("\"content\":\"BASE\"") == std::string::npos);
}

/// @brief The Responses API variant also honors the gate.
TEST_CASE("decision gate applies to Responses jiezhu requests") {
	std::atomic<int> decision_calls{0};

	test_support::tiny_http_server server([&](const std::string& request) {
		if (test_support::starts_with(request, "POST /v1/decisions ")) {
			++decision_calls;
			return test_support::build_http_response(
				"200 OK", "application/json", decision_body(0.04));
		}
		return test_support::build_http_response(
			"200 OK",
			"application/json",
			R"({"id":"resp-id","output_text":"hello"})");
	});

	jie::client_options opt;
	opt.api_key = "test-key";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;
	opt.experimental_decision_gate = true;

	jie::client c(opt);
	jie::response_request r;
	r.model = "gpt-test";
	r.instructions = "BASE";
	r.input_text = "What is the capital of France?";

	const jie::response_response resp = c.responses_jiezhu(r);
	REQUIRE(resp.first_content() == "hello");
	REQUIRE(decision_calls.load() == 1);

	const std::string request = server.last_request();
	REQUIRE(test_support::starts_with(request, "POST /v1/responses "));
	const std::string body = request_body(request);
	REQUIRE(body.find("\"instructions\":\"BASE\"") != std::string::npos);
}
