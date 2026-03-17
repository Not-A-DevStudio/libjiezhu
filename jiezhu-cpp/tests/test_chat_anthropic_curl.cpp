#include <catch2/catch_test_macros.hpp>

#include <jie/chat.hpp>

#include <chrono>
#include <string>

#include "test_http_server.hpp"

TEST_CASE("first_content is consistent for OpenAI and Anthropic response shapes") {
	jie::chat_completion_response openai_resp;
	openai_resp.raw = nlohmann::json::parse(R"({
		"choices":[{"message":{"content":"same-text"}}]
	})");

	jie::chat_completion_response anthropic_resp;
	anthropic_resp.raw = nlohmann::json::parse(R"({
		"content":[{"type":"text","text":"same-text"}]
	})");

	REQUIRE(openai_resp.first_content() == "same-text");
	REQUIRE(anthropic_resp.first_content() == "same-text");
}

TEST_CASE("chat_completions_create_anthropic sends Claude-style request and parses response") {
	test_support::tiny_http_server server([](const std::string&) {
		const std::string body = R"({
			"id":"msg_test",
			"model":"claude-test",
			"content":[{"type":"text","text":"hello-claude"}]
		})";
		return test_support::build_http_response("200 OK", "application/json", body);
	});

	jie::client_options opt;
	opt.api_key = "claude-key";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port());
	opt.timeout_seconds = 5;

	jie::client c(opt);
	jie::chat_completion_request r;
	r.model = "claude-test";
	r.max_tokens = 128;
	r.messages.push_back({"system", "sys"});
	r.messages.push_back({"user", "hi"});

	jie::chat_completion_response resp = c.chat_completions_create_anthropic(r);
	REQUIRE(server.wait_for_request(std::chrono::milliseconds(1500)));

	const std::string req = server.last_request();
	REQUIRE(test_support::starts_with(req, "POST /messages "));
	REQUIRE(test_support::icontains(req, "content-type: application/json"));
	REQUIRE(test_support::icontains(req, "accept: application/json"));
	REQUIRE(test_support::icontains(req, "x-api-key: claude-key"));
	REQUIRE(test_support::icontains(req, "anthropic-version: 2023-06-01"));
	REQUIRE_FALSE(test_support::icontains(req, "authorization: bearer"));
	REQUIRE(test_support::icontains(req, "\"system\":\"sys\""));
	REQUIRE(test_support::icontains(req, "\"messages\":[{"));
	REQUIRE(test_support::icontains(req, "\"role\":\"user\""));
	REQUIRE(test_support::icontains(req, "\"content\":\"hi\""));

	REQUIRE(resp.id == "msg_test");
	REQUIRE(resp.model == "claude-test");
	REQUIRE(resp.first_content() == "hello-claude");
}
