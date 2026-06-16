/**
 * @file test_http_server.hpp
 * @brief Single-threaded loopback HTTP server used by the Catch2 tests.
 *
 * Provides a RAII helper that binds to an ephemeral port on the
 * loopback interface, dispatches incoming connections to a user-supplied
 * handler and exposes the captured request for assertions. The class is
 * intentionally minimal — it is not intended for production use.
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

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
#endif

namespace test_support {

/// @brief RAII wrapper that initializes Winsock on Windows.
///
/// @c socket_runtime is constructed once via @c tiny_http_server and
/// guarantees matching @c WSACleanup on Windows. On POSIX systems the
/// type is empty and incurs no cost.
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

/// @brief Return a lower-cased copy of @p s.
/// @param s Input string.
/// @return Lowercase copy of @p s.
inline std::string to_lower_copy(std::string s) {
	for (char& ch : s) {
		ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	}
	return s;
}

/// @brief Test whether @p s starts with @p prefix.
/// @param s Candidate string.
/// @param prefix Prefix to look for.
/// @return @c true if @p s begins with @p prefix, @c false otherwise.
inline bool starts_with(std::string_view s, std::string_view prefix) {
	return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

/// @brief Case-insensitive substring search.
/// @param haystack String to search in.
/// @param needle Substring to look for.
/// @return @c true if @p haystack contains @p needle (ignoring case).
inline bool icontains(std::string_view haystack, std::string_view needle) {
	std::string h(haystack);
	std::string n(needle);
	h = to_lower_copy(std::move(h));
	n = to_lower_copy(std::move(n));
	return h.find(n) != std::string::npos;
}

/// @brief Build a minimal HTTP/1.1 response with @c Connection: close.
/// @param status Status line text (e.g. @c "200 OK").
/// @param content_type Value for the @c Content-Type header.
/// @param body Response body. The function sets @c Content-Length for
/// you and appends it to the response.
/// @return The fully formatted HTTP response, ready to be written to a
/// socket.
inline std::string build_http_response(std::string_view status,
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

/// @brief A single-threaded, loopback-only HTTP/1.1 test server.
///
/// The server binds to an ephemeral port on @c 127.0.0.1, accepts
/// connections one at a time, hands the raw request bytes to a
/// user-supplied handler, and writes the handler's response back. The
/// most recently received request can be retrieved with
/// @ref last_request() for assertion purposes.
class tiny_http_server {
public:
	/// @brief Handler signature. Receives the raw request string and
	/// must return the full HTTP response (headers + body).
	using handler_t = std::function<std::string(const std::string& request)>;

	/// @brief Construct and start a server backed by @p handler.
	/// @param handler Function used to synthesize each response.
	/// @throw std::runtime_error If the listening socket cannot be
	/// created, bound or put into the listening state.
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

	/// @brief Stop the server (if running) and release the listening
	/// socket. Safe to call multiple times.
	~tiny_http_server() {
		stop();
	}

	/// @brief Return the ephemeral port the server bound to.
	/// @return The TCP port in host byte order.
	uint16_t port() const { return port_; }

	/// @brief Block up to @p timeout waiting for the first request to
	/// arrive.
	/// @param timeout Maximum time to wait.
	/// @return @c true if a request was observed before the timeout
	/// elapsed, @c false otherwise.
	bool wait_for_request(std::chrono::milliseconds timeout) {
		std::unique_lock<std::mutex> lock(req_mu_);
		return req_cv_.wait_for(lock, timeout, [&] { return request_count_ > 0; });
	}

	/// @brief Return a copy of the most recently received raw request.
	/// @return Raw request string (headers + body).
	std::string last_request() const {
		std::lock_guard<std::mutex> lock(req_mu_);
		return last_request_;
	}

	/// @brief Stop the accept loop and close the listening socket.
	/// Idempotent.
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
	/// @brief Accept loop run on the worker thread.
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

} // namespace test_support
