#include <catch2/catch_test_macros.hpp>

#include <jie/chat.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <winsock2.h>
	#include <ws2tcpip.h>
	#pragma comment(lib, "Ws2_32.lib")
	using socket_t = SOCKET;
	static constexpr socket_t invalid_socket_v = INVALID_SOCKET;
	static void close_socket(socket_t s) {
		if (s != INVALID_SOCKET) {
			closesocket(s);
		}
	}
	static int last_socket_error() { return WSAGetLastError(); }
#else
	#include <arpa/inet.h>
	#include <netinet/in.h>
	#include <sys/socket.h>
	#include <sys/types.h>
	#include <unistd.h>
	using socket_t = int;
	static constexpr socket_t invalid_socket_v = -1;
	static void close_socket(socket_t s) {
		if (s >= 0) {
			::close(s);
		}
	}
	static int last_socket_error() { return errno; }
#endif

namespace {

struct socket_runtime {
	socket_runtime() {
#if defined(_WIN32)
		WSADATA wsa;
		int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
		if (rc != 0) throw std::runtime_error("WSAStartup failed");
#endif
	}
	~socket_runtime() {
#if defined(_WIN32)
		WSACleanup();
#endif
	}
};

static std::string to_lower_copy(std::string s) {
	for (char& ch : s) {
		ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	}
	return s;
}

static bool starts_with(std::string_view s, std::string_view prefix) {
	return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

static bool icontains(std::string_view haystack, std::string_view needle) {
	std::string h(haystack);
	std::string n(needle);
	h = to_lower_copy(std::move(h));
	n = to_lower_copy(std::move(n));
	return h.find(n) != std::string::npos;
}

static std::string build_http_response(std::string_view status,
							std::string_view content_type,
							std::string body) {
	std::string resp;
	resp += "HTTP/1.1 ";
	resp += status;
	resp += "\r\n";
	resp += "Content-Type: ";
	resp += content_type;
	resp += "\r\n";
	resp += "Content-Length: ";
	resp += std::to_string(body.size());
	resp += "\r\n";
	resp += "Connection: close\r\n";
	resp += "\r\n";
	resp += std::move(body);
	return resp;
}

class tiny_http_server {
public:
	using handler_t = std::function<std::string(const std::string& request)>;

	explicit tiny_http_server(handler_t handler) : handler_(std::move(handler)) {
		static socket_runtime rt;

		listen_sock_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
		if (listen_sock_ == invalid_socket_v) throw std::runtime_error("socket() failed");

		int yes = 1;
#if defined(_WIN32)
		setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
		setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif

		sockaddr_in addr{};
		addr.sin_family = AF_INET;
		addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
		addr.sin_port = htons(0);
		if (::bind(listen_sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
			close_socket(listen_sock_);
			throw std::runtime_error("bind() failed");
		}

		sockaddr_in bound{};
#if defined(_WIN32)
		int bound_len = sizeof(bound);
#else
		socklen_t bound_len = sizeof(bound);
#endif
		if (::getsockname(listen_sock_, reinterpret_cast<sockaddr*>(&bound), &bound_len) != 0) {
			close_socket(listen_sock_);
			throw std::runtime_error("getsockname() failed");
		}
		port_ = ntohs(bound.sin_port);

		if (::listen(listen_sock_, 8) != 0) {
			close_socket(listen_sock_);
			throw std::runtime_error("listen() failed");
		}

		running_.store(true);
		worker_ = std::thread([this] { this->run(); });
	}

	~tiny_http_server() {
		stop();
	}

	uint16_t port() const { return port_; }

	bool wait_for_request(std::chrono::milliseconds timeout) {
		std::unique_lock<std::mutex> lock(req_mu_);
		return req_cv_.wait_for(lock, timeout, [&] { return request_count_ > 0; });
	}

	std::string last_request() const {
		std::lock_guard<std::mutex> lock(req_mu_);
		return last_request_;
	}

	void stop() {
		bool expected = true;
		if (!running_.compare_exchange_strong(expected, false)) return;

		// Unblock accept by connecting once.
		try {
			static socket_runtime rt;
			socket_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
			if (s != invalid_socket_v) {
				sockaddr_in addr{};
				addr.sin_family = AF_INET;
				addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
				addr.sin_port = htons(port_);
				::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
				close_socket(s);
			}
		} catch (...) {
		}

		if (worker_.joinable()) worker_.join();
		close_socket(listen_sock_);
		listen_sock_ = invalid_socket_v;
	}

private:
	void run() {
		while (running_.load()) {
			sockaddr_in peer{};
#if defined(_WIN32)
			int peer_len = sizeof(peer);
#else
			socklen_t peer_len = sizeof(peer);
#endif
			socket_t client = ::accept(listen_sock_, reinterpret_cast<sockaddr*>(&peer), &peer_len);
			if (client == invalid_socket_v) {
				if (!running_.load()) break;
				continue;
			}

			std::string req;
			req.reserve(4096);
			char buf[4096];

			// Read headers first.
			while (req.find("\r\n\r\n") == std::string::npos) {
#if defined(_WIN32)
				int n = ::recv(client, buf, static_cast<int>(sizeof(buf)), 0);
#else
				int n = static_cast<int>(::recv(client, buf, sizeof(buf), 0));
#endif
				if (n <= 0) break;
				req.append(buf, buf + n);
				if (req.size() > 1024 * 1024) break;
			}

			// If Content-Length exists, read full body.
			std::size_t header_end = req.find("\r\n\r\n");
			std::size_t body_start = header_end == std::string::npos ? req.size() : header_end + 4;

			std::size_t content_length = 0;
			{
				std::string headers = header_end == std::string::npos ? req : req.substr(0, header_end);
				std::string lower = to_lower_copy(headers);
				std::string key = "content-length:";
				std::size_t pos = lower.find(key);
				if (pos != std::string::npos) {
					std::size_t line_end = lower.find("\r\n", pos);
					std::size_t vpos = pos + key.size();
					while (vpos < lower.size() && (lower[vpos] == ' ' || lower[vpos] == '\t')) ++vpos;
					std::string_view v(lower.c_str() + vpos,
									  (line_end == std::string::npos ? lower.size() : line_end) - vpos);
					try {
						content_length = static_cast<std::size_t>(std::stoul(std::string(v)));
					} catch (...) {
						content_length = 0;
					}
				}
			}

			while (req.size() < body_start + content_length) {
#if defined(_WIN32)
				int n = ::recv(client, buf, static_cast<int>(sizeof(buf)), 0);
#else
				int n = static_cast<int>(::recv(client, buf, sizeof(buf), 0));
#endif
				if (n <= 0) break;
				req.append(buf, buf + n);
				if (req.size() > 1024 * 1024) break;
			}

			std::string resp;
			{
				std::lock_guard<std::mutex> lock(req_mu_);
				last_request_ = req;
				++request_count_;
			}
			req_cv_.notify_all();
			try {
				resp = handler_ ? handler_(req) : build_http_response("500 Internal Server Error", "text/plain", "no handler");
			} catch (const std::exception& e) {
				resp = build_http_response("500 Internal Server Error", "text/plain", e.what());
			}

			// Best-effort send all.
			std::size_t sent = 0;
			while (sent < resp.size()) {
#if defined(_WIN32)
				int n = ::send(client, resp.data() + sent, static_cast<int>(resp.size() - sent), 0);
#else
				int n = static_cast<int>(::send(client, resp.data() + sent, resp.size() - sent, 0));
#endif
				if (n <= 0) break;
				sent += static_cast<std::size_t>(n);
			}

			close_socket(client);
		}
	}

	std::atomic<bool> running_{false};
	socket_t listen_sock_{invalid_socket_v};
	uint16_t port_{0};
	handler_t handler_;
	std::thread worker_;

	mutable std::mutex req_mu_;
	mutable std::condition_variable req_cv_;
	std::string last_request_;
	std::size_t request_count_{0};
};

} // namespace

TEST_CASE("chat_completions_create uses curl and parses JSON") {
	tiny_http_server server([](const std::string&) {
		const std::string body = R"({
			"id":"test-id",
			"model":"test-model",
			"choices":[{"message":{"content":"hello"}}]
		})";
		return build_http_response("200 OK", "application/json", body);
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
	REQUIRE(starts_with(req, "POST /v1/chat/completions "));
	REQUIRE(icontains(req, "http/"));
	REQUIRE(icontains(req, "content-type: application/json"));
	REQUIRE(icontains(req, "accept: application/json"));
	REQUIRE(resp.id == "test-id");
	REQUIRE(resp.model == "test-model");
	REQUIRE(resp.first_content() == "hello");
}

TEST_CASE("chat_completions_stream consumes SSE events") {
	tiny_http_server server([](const std::string&) {
		std::string sse;
		sse += "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n";
		sse += "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n";
		sse += "data: [DONE]\n\n";

		// Note: we send as one body; curl callback will still parse by lines.
		return build_http_response("200 OK", "text/event-stream", sse);
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
	REQUIRE(starts_with(req, "POST /v1/chat/completions "));
	REQUIRE(icontains(req, "accept: text/event-stream"));
	REQUIRE(all == "Hello");
	REQUIRE(saw_done);
}

TEST_CASE("non-2xx responses throw with body") {
	tiny_http_server server([](const std::string&) {
		return build_http_response("401 Unauthorized", "text/plain", "bad key");
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
	REQUIRE(starts_with(req, "POST /v1/chat/completions "));
}
