#include <catch2/catch_test_macros.hpp>

#include <jie/chat.hpp>

#include <chrono>
#include <string>

#include "test_http_server.hpp"

TEST_CASE("chat_completions_stream consumes SSE events") {
	test_support::tiny_http_server server([](const std::string&) {
		std::string sse;
		sse += "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
		sse += "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n";
		sse += "data: [DONE]\n\n";

		return test_support::build_http_response("200 OK", "text/event-stream", sse);
	});

	jie::client_options opt;
	opt.api_key = "test";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;

	jie::client c(opt);
	jie::chat_completion_request r;
	r.model = "gpt-test";
	r.messages.push_back({"user", "hi"});

	std::string all;
	bool saw_done = false;
	c.chat_completions_stream(r, [&](const jie::chat_completion_stream_event& ev) {
		all += ev.delta_content;
		if (ev.done) saw_done = true;
		return true;
	});

	REQUIRE(server.wait_for_request(std::chrono::milliseconds(1500)));
	const std::string req = server.last_request();
	REQUIRE(test_support::starts_with(req, "POST /v1/chat/completions "));
	REQUIRE(test_support::icontains(req, "accept: text/event-stream"));
	REQUIRE(all == "Hello");
	REQUIRE(saw_done);
}
