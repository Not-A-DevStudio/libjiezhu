# Quick Start Guide

The following examples demonstrate:

- Initializing `jie::client`
- Sending a non-streaming chat completion
- Sending a streaming chat completion and concatenating the output

> Note: This library only handles HTTP calls and JSON processing. It does not manage your API Key. Please read it from environment variables/configuration files and pass it in.

## 1) Including via CMake

If you have installed this library (`cmake --install`) and exported the CMake package, you can:

```cmake
find_package(jiezhu CONFIG REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE jiezhu::jiezhu)
```

If you prefer to build it as part of your project, you can use `add_subdirectory`:

```cmake
add_subdirectory(path/to/jiezhu/cpp)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE jiezhu::jiezhu)
```

More detail, see [build.md](build.md)

## 2) Non-Streaming Example

```cpp
#include <jie/jiezhu.hpp>

#include <iostream>

int main() {
    jie::client_options opt;
    opt.api_key = "YOUR_API_KEY";
    // opt.base_url = "https://api.openai.com/v1"; // default value

    jie::client c(opt);

    jie::chat_completion_request req;
    req.model = "gpt-4.1-mini"; // model name, e.g., "gpt-4.1-mini", "gpt-3.5-turbo", etc.
    req.messages = {
        {"system", "You are a helpful assistant."},
        {"user", "Introduce yourself in 20 words."},
    };

    // options
    req.temperature = 0.7;
    // req.max_tokens = 256;

    // pass through any extra fields (forward compatibility)
    // req.extra["top_p"] = 0.9;

    auto resp = c.chat_completions_create(req);
    std::cout << resp.first_content() << "\n";
}
```

## 3) Streaming Example

```cpp
#include <jie/jiezhu.hpp>

#include <iostream>
#include <string>

int main() {
    jie::client_options opt;
    opt.api_key = "YOUR_API_KEY";

    jie::client c(opt);

    jie::chat_completion_request req;
    req.model = "gpt-4.1-mini";
    req.messages = {
        {"user", "Introduce yourself in 20 words."},
    };

    std::string full;
    c.chat_completions_stream(req, [&](const jie::chat_completion_stream_event& ev) {
        if (ev.done) {
            // may also check ev.finish_reason for more details
            return false; // stream ended, stop
        }

        if (!ev.delta_content.empty()) {
            full += ev.delta_content;
            std::cout << ev.delta_content << std::flush;
        }
        return true; // continue streaming
    });

    std::cout << "\n---\nFULL: " << full << "\n";
}
```

## 4) FAQ

- **Q: How to cancel a streaming request?**
  - Return `false` in the callback to cancel.

- **Q: What happens if the request fails?**
  - A `std::runtime_error` will be thrown:
    - libcurl failure: `curl_easy_perform failed: ...`
    - HTTP non-2xx: `HTTP <code>: <body>` (body may be JSON text or partial response)

- **Q: Do I need to parse JSON myself?**
  - You can directly use `chat_completion_response::raw` (`nlohmann::json`) to get the full response.
  - You can also use `first_content()` for the most common case of "get choices[0].message.content".