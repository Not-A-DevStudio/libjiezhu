#include <iostream>
#include <string>
#include <jie/jiezhu.hpp>
#include <clocale>
#ifdef _WIN32
#include <windows.h>
#endif
int main() {
    #ifdef _WIN32
    // Enable UTF-8 output in Windows console.
    SetConsoleOutputCP(CP_UTF8);
    #endif
    std::setlocale(LC_ALL, ".UTF-8");
    try {
        jie::client_options opt{};
        std::cout << "Your API Key: " << std::endl;
        std::getline(std::cin, opt.api_key);
        if (opt.api_key.empty() && std::cin.good()) {
            // In case a previous formatted input left a trailing newline.
            std::getline(std::cin, opt.api_key);
        }
        // opt.base_url = "https://api.openai.com/v1"; // default, uncomment and set your base URL if needed

        jie::client client(opt);

        jie::chat_completion_request req{};
        // req.model = "gpt-4o"; // set your model
        req.messages = {
            {"system", "You are a helpful assistant."},
            {"user", "Please introduce yourself."}, // We reconmend using Chinese prompt and module for better demonstration of Jiezhu ability and understanding of Chinese, but English also works.
        };

        req.temperature = 0.7;
        req.max_tokens = 512;

        auto resp = client.chat_completions_create(req);
        const std::string content = resp.first_content();
        std::cout << "Response with `chat_completions_create` (no Jiezhu):" << resp.id << std::endl;
        if (!content.empty()) {
            std::cout << content << std::endl;
        } else {
            std::cout << resp.raw.dump(2) << std::endl;
        }
        auto resp_jiezhu = client.chat_completions_jiezhu(req);
        const std::string content_jiezhu = resp_jiezhu.first_content();
        std::cout << "\nResponse with `chat_completions_jiezhu`:" << resp_jiezhu.id << std::endl;
        if (!content_jiezhu.empty()) {
            std::cout << content_jiezhu << std::endl;
        } else {
            std::cout << resp_jiezhu.raw.dump(2) << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
