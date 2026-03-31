#include <catch2/catch_test_macros.hpp>

#include <jie/chat.hpp>

#include <chrono>
#include <stdexcept>
#include <string>

#include "test_http_server.hpp"

TEST_CASE("non-2xx responses throw with body") {
	test_support::tiny_http_server server([](const std::string&) {
		return test_support::build_http_response("401 Unauthorized", "text/plain", "bad key");
	});

	jie::client_options opt;
	opt.api_key = "test";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;

	jie::client c(opt);
	jie::chat_completion_request r;
	r.model = "gpt-test";
	r.messages.push_back({"user", "hi"});

	try {
		(void)c.chat_completions_create(r);
		FAIL("expected exception");
	} catch (const std::runtime_error& e) {
		std::string msg = e.what();
		REQUIRE(msg.find("HTTP 401") != std::string::npos);
		REQUIRE(msg.find("bad key") != std::string::npos);
	}

	REQUIRE(server.wait_for_request(std::chrono::milliseconds(1500)));
	const std::string req = server.last_request();
	REQUIRE(test_support::starts_with(req, "POST /v1/chat/completions "));
}
