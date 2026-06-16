/**
 * @file test_chat_create_curl.cpp
 * @brief Catch2 tests for @ref jie::client::chat_completions_create.
 *
 * Verifies the OpenAI-style blocking call: it posts the expected
 * request, sets the right headers, and parses the standard response
 * shape via @ref jie::chat_completion_response.
 */
#include <catch2/catch_test_macros.hpp>

#include <jie/chat.hpp>

#include <chrono>
#include <string>

#include "test_http_server.hpp"

/// @brief Verifies request URL/headers and the parsed response payload.
TEST_CASE("chat_completions_create uses curl and parses JSON") {
	test_support::tiny_http_server server([](const std::string&) {
		const std::string body = R"({
			"id":"test-id",
			"model":"test-model",
			"choices":[{"message":{"content":"hello"}}]
		})";
		return test_support::build_http_response("200 OK", "application/json", body);
	});

	jie::client_options opt;
	opt.api_key = "test";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;

	jie::client c(opt);
	jie::chat_completion_request r;
	r.model = "gpt-test";
	r.messages.push_back({"user", "hi"});

	jie::chat_completion_response resp = c.chat_completions_create(r);
	REQUIRE(server.wait_for_request(std::chrono::milliseconds(1500)));
	const std::string req = server.last_request();
	REQUIRE(test_support::starts_with(req, "POST /v1/chat/completions "));
	REQUIRE(test_support::icontains(req, "http/"));
	REQUIRE(test_support::icontains(req, "content-type: application/json"));
	REQUIRE(test_support::icontains(req, "accept: application/json"));
	REQUIRE(resp.id == "test-id");
	REQUIRE(resp.model == "test-model");
	REQUIRE(resp.first_content() == "hello");
}
