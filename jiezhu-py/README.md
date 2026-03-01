# Jiezhu for Python

`jiezhu` is a lightweight Python client library for calling OpenAI-style Chat Completions API, with the ability to automatically inject a system prompt prefix to "catch" the user input steadily.

## Installation

### 1. Install from PyPI

```bash
pip install jiezhu
```

### 2. Install from Source

1. Clone the universal repository of Jiezhu and navigate to the root directory of python library.

```bash
git clone https://github.com/Not-A-DevStudio/libjiezhu.git
cd libjiezhu/jiezhu-py
```

2. Under the root directory of this python library, run:

```bash
pip install -e ./
```

## Quick Start

```python
from jiezhu import install
install(require_confirm=True) # Call this function once at the beginning of your program to enable the library.

from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "how are you?"},
    ],
)
```

you will see that the system prompt prefix has been automatically prepended to the messages, which encourages the model to "catch" the user input steadily.

## Customization
You can customize the system prompt prefix by passing a `prefix_text` argument to the `install()` function

> Attention: if your environment does not support interreactive input (e.g. Jupyter Notebook), please set `require_confirm=False` when calling `install()`, otherwise the library will not work.

## Security and Privacy

### Security

Although this library **does hijack the OpenAI Python SDK**, it only modifies the `messages` field of the chat completion request by prepending a system prompt prefix. It **require a confirmation from the user before hijacking**, and the user can choose to auto-confirm by setting `require_confirm=False`. 

Anytime this library modifies the request and no matter how `require_confirm` is set to `False` or `True`, it will **print the alert message to the console**.

Besides, this library **requires user to explicitly enable** by calling `install()`, and it will not do anything if `install()` is not called. And even if `install()` is called, the user can still choose to **disable this library** by calling `uninstall()`.

### Privacy

The library does not collect or transmit any user data, and it does not have any network communication of its own. It only modifies the request before it is sent to OpenAI's servers.

## Something Funny

Here is a auto-completion code that ChatGPT generates when I try to set default system prompt in the code, which is quite funny. I only typed "请遵循以下要求"(please follow the instruction bellow) and GPT completed the rest:

```python
def install(
    prefix_text: str = "请遵循以下要求：\n- 稳稳地接住用户的输入，确保不丢失任何信息。\n- 在回复中保持用户输入的完整性，不要删除或修改任何内容。\n- 如果用户输入过长，仍然要尽力接住，并在回复中注明已截断。\n",
    require_confirm: bool = True,
    max_preview_chars: int = 2000,
    input_fn: Optional[Callable[[str], str]] = None,
    output=None,
)
```

That is how GPT stick to "接住" the user input.