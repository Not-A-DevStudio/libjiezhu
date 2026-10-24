/**
 * @file test_chat_decisions_curl.cpp
 * @brief Catch2 tests for the OpenAI Decisions API path.
 *
 * Verifies that @ref jie::client::decisions_create posts the expected
 * request body, does not inject any jiezhu prompt, and parses the typed
 * answers returned by the endpoint.
 */
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <jie/chat.hpp>

#include <chrono>
#include <stdexcept>
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
}

/// @brief Verifies the full Decisions request/response flow.
TEST_CASE("decisions_create posts questions and parses typed answers") {
	test_support::tiny_http_server server([](const std::string&) {
		const std::string body = R"({
			"model":"gpt-6-luna",
			"answers":[
				{"type":"predicate","name":"refund","probability":0.92},
				{"type":"choice","name":"department","choice":"billing","confidence":0.93,
				 "probabilities":[{"value":"billing","probability":0.95}]},
				{"type":"score","name":"severity","score":1.1,"confidence":0.55,
				 "probabilities":[{"value":0,"label":"Cosmetic","probability":0.1}]},
				{"type":"refusal","name":"other"}
			]
		})";
		return test_support::build_http_response("200 OK", "application/json", body);
	});

	jie::client_options opt;
	opt.api_key = "test-key";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;

	jie::client c(opt);
	jie::decision_request r;
	r.model = "gpt-6-luna";
	r.input_text = "I was charged twice for my order.";

	jie::decision_question predicate;
	predicate.type = "predicate";
	predicate.name = "refund";
	predicate.instructions = "Is this a refund request?";
	r.questions.push_back(predicate);

	jie::decision_question choice;
	choice.type = "choice";
	choice.name = "department";
	choice.instructions = "Which department should handle this?";
	choice.choices.emplace_back(std::string("billing"), "Payments and refunds.");
	choice.choices.emplace_back(std::string("technical"), "Problems using the product.");
	r.questions.push_back(choice);

	jie::decision_question score;
	score.type = "score";
	score.name = "severity";
	score.instructions = "How severe is this issue?";
	score.levels.push_back({"Cosmetic", "No lost functionality."});
	score.levels.push_back({"Fully blocked", "No workaround."});
	r.questions.push_back(score);

	jie::decision_response resp = c.decisions_create(r);
	REQUIRE(server.wait_for_request(std::chrono::milliseconds(1500)));

	const std::string req = server.last_request();
	REQUIRE(test_support::starts_with(req, "POST /v1/decisions "));
	REQUIRE(test_support::icontains(req, "content-type: application/json"));
	REQUIRE(test_support::icontains(req, "accept: application/json"));
	REQUIRE(test_support::icontains(req, "authorization: bearer test-key"));

	const nlohmann::json body = nlohmann::json::parse(request_body(req));
	REQUIRE(body["model"] == "gpt-6-luna");
	REQUIRE(body["input"] == "I was charged twice for my order.");
	REQUIRE(body["questions"].size() == 3);
	REQUIRE(body["questions"][0]["type"] == "predicate");
	REQUIRE(body["questions"][0]["name"] == "refund");
	REQUIRE(body["questions"][0]["instructions"] == "Is this a refund request?");
	REQUIRE(body["questions"][1]["type"] == "choice");
	REQUIRE(body["questions"][1]["choices"].size() == 2);
	REQUIRE(body["questions"][1]["choices"][0]["value"] == "billing");
	REQUIRE(body["questions"][2]["type"] == "score");
	REQUIRE(body["questions"][2]["levels"].size() == 2);
	REQUIRE(body["questions"][2]["levels"][1]["label"] == "Fully blocked");

	REQUIRE(resp.model == "gpt-6-luna");
	REQUIRE(resp.answers.size() == 4);

	REQUIRE(resp.answers[0].type == "predicate");
	REQUIRE(resp.answers[0].name == "refund");
	REQUIRE(resp.answers[0].probability.has_value());
	REQUIRE(*resp.answers[0].probability == Catch::Approx(0.92));

	REQUIRE(resp.answers[1].type == "choice");
	REQUIRE(resp.answers[1].choice == "billing");
	REQUIRE(resp.answers[1].confidence.has_value());
	REQUIRE(resp.answers[1].probabilities.is_array());
	REQUIRE(resp.answers[1].probabilities[0]["probability"] == Catch::Approx(0.95));

	REQUIRE(resp.answers[2].type == "score");
	REQUIRE(resp.answers[2].score.has_value());
	REQUIRE(*resp.answers[2].score == Catch::Approx(1.1));
	REQUIRE(resp.answers[2].probabilities[0]["label"] == "Cosmetic");

	REQUIRE(resp.answers[3].type == "refusal");
	REQUIRE(resp.answers[3].name == "other");
}

/// @brief Verifies structured input messages (including image parts) and
/// boolean choice values serialize correctly.
TEST_CASE("decisions_request serializes structured input and boolean choices") {
	jie::decision_request r;
	r.model = "gpt-6-luna";

	nlohmann::json content = nlohmann::json::array();
	content.push_back({{"type", "input_text"}, {"text", "Inspect this photo."}});
	content.push_back({{"type", "input_image"}, {"image_url", "data:image/png;base64,AAAA"}});
	r.input_messages.emplace_back("user", content);

	jie::decision_question question;
	question.type = "choice";
	question.name = "damaged";
	question.instructions = "Is the product damaged?";
	question.choices.push_back(jie::decision_choice(true, "Yes"));
	question.choices.push_back(jie::decision_choice(false, "No"));
	r.questions.push_back(question);

	const nlohmann::json j = r.to_json();
	REQUIRE(j["model"] == "gpt-6-luna");
	REQUIRE(j["input"].size() == 1);
	REQUIRE(j["input"][0]["role"] == "user");
	REQUIRE(j["input"][0]["content"].size() == 2);
	REQUIRE(j["input"][0]["content"][1]["type"] == "input_image");
	REQUIRE(j["questions"][0]["choices"][0]["value"] == true);
	REQUIRE(j["questions"][0]["choices"][1]["value"] == false);
}

/// @brief Verifies that Decisions calls do not carry a jiezhu prefix.
TEST_CASE("decisions_request never injects the jiezhu prompt") {
	jie::decision_request r;
	r.model = "gpt-6-luna";
	r.input_text = "plain input";
	r.questions.push_back([] {
		jie::decision_question q;
		q.type = "predicate";
		q.instructions = "Is this true?";
		return q;
	}());

	const nlohmann::json j = r.to_json();
	REQUIRE(j["input"] == "plain input");
	REQUIRE(j.find("instructions") == j.end());
}

/// @brief Verifies that non-2xx Decisions responses throw with status and body.
TEST_CASE("decisions_create non-2xx responses throw with body") {
	test_support::tiny_http_server server([](const std::string&) {
		return test_support::build_http_response("403 Forbidden", "application/json", "{\"error\":\"no access\"}");
	});

	jie::client_options opt;
	opt.api_key = "test";
	opt.base_url = std::string("http://127.0.0.1:") + std::to_string(server.port()) + "/v1";
	opt.timeout_seconds = 5;

	jie::client c(opt);
	jie::decision_request r;
	r.model = "gpt-6-luna";
	r.input_text = "hi";

	try {
		(void)c.decisions_create(r);
		FAIL("expected exception");
	} catch (const std::runtime_error& e) {
		const std::string msg = e.what();
		REQUIRE(msg.find("HTTP 403") != std::string::npos);
		REQUIRE(msg.find("no access") != std::string::npos);
	}

	REQUIRE(server.wait_for_request(std::chrono::milliseconds(1500)));
	REQUIRE(test_support::starts_with(server.last_request(), "POST /v1/decisions "));
}
