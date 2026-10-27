# Jiezhu for Python

`jiezhu` is a lightweight Python client library for calling OpenAI-style Chat Completions API and Anthropic-style API, with the ability to automatically inject a system prompt prefix to "catch" the user input steadily.

## Prerequisites

To use this library, you need to have Python 3.8 or higher installed. This library **does not automatically install any ai SDK**. You also need to have the OpenAI Python SDK installed for OpenAI API support, and the Anthropic Python SDK installed for Anthropic API support.

## Installation

### Install from PyPI

```bash
pip install jiezhu[openai] # or jiezhu[claude] if you want claude API support
```

### Install from Source

1. Clone the universal repository of Jiezhu and navigate to the root directory of python library.

```bash
git clone https://github.com/Not-A-DevStudio/libjiezhu.git
cd libjiezhu/jiezhu-py
```

Under the root directory of this python library, run:

```bash
pip install -e ./
```

Given that this repository contains both C++ and Python libraries, you may also find a **packed Python source file** of this repository.

### Install from Pre-built Binary Package

1. Go to the [Release page](https://github.com/Not-A-DevStudio/libjiezhu/releases) of this repository, and find the latest release.

2. Download the pre-built binary package for Python (e.g., `jiezhu-0.1.0-py3-none-any.whl`).

3. Install the downloaded package using pip:

```bash
pip install path/to/downloaded/wheel.whl
```

## Quick Start

### OpenAI Support

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

### Claude / Anthropic Support

Jiezhu also supports the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python). Install the optional dependency:

```bash
pip install jiezhu[claude]
```

Then use it the same way — `install()` patches both OpenAI and Anthropic SDKs:

```python
from jiezhu import install
install(require_confirm=False)

from anthropic import Anthropic
client = Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "how are you?"},
    ],
)
```

The prefix is prepended to Claude's top-level `system` parameter (supports both string and content-block list formats). If the `anthropic` package is not installed, Claude patching is silently skipped.

## Customization

You can customize the system prompt prefix by passing a `prefix_text` argument to the `install()` function

> Attention: if your environment does not support interreactive input (e.g. Jupyter Notebook), please set `require_confirm=False` when calling `install()`, otherwise the library will not work.

## Experimental: Decisions API gate

By default the prefix is injected on every intercepted call. The
experimental decision gate asks a model through the
[OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions)
whether the latest user message actually needs emotional support. When
the predicate probability is below the threshold, the request is sent
without the jiezhu prefix.

```python
from jiezhu import install, set_experimental_decision_gate

install(require_confirm=False)

# model defaults to "gpt-6-luna", threshold defaults to 0.5
set_experimental_decision_gate(True)
```

The gate can also be enabled during installation:

```python
install(
    require_confirm=False,
    experimental_decision_gate=True,
    decision_model="gpt-6-luna",
    decision_threshold=0.5,
)
```

The gate reuses the same OpenAI client as the intercepted call
(`client.decisions.create`), so no extra credentials are needed. It is
**fail-open**: disabled gates, calls without user text, malformed
answers and Decisions API errors all keep the normal jiezhu behaviour.
Requests that the model classifies as not needing emotional support are
sent unchanged.

The default backend is OpenAI-specific. For Anthropic-patched calls,
provide a `decision_fn` (or point the client at an OpenAI-compatible
endpoint that supports Decisions); otherwise the gate fails open and
keeps injecting the prefix.

For custom classifiers or tests, pass a callback. It receives the latest
user message and returns a truthy value when support is needed:

```python
set_experimental_decision_gate(
    True,
    decision_fn=lambda text: my_classifier(text),
)
```

## Security and Privacy

### Security

Although this library **does hijack the OpenAI Python SDK**, it only modifies the `messages` field of the chat completion request by prepending a system prompt prefix. It **require a confirmation from the user before hijacking**, and the user can choose to auto-confirm by setting `require_confirm=False`.

Anytime this library modifies the request and no matter how `require_confirm` is set to `False` or `True`, it will **print the alert message to the console**.

Besides, this library **requires user to explicitly enable** by calling `install()`, and it will not do anything if `install()` is not called. And even if `install()` is called, the user can still choose to **disable this library** by calling `uninstall()`.

### Privacy

The library does not collect or transmit any user data, and it does not
have any network communication of its own **unless the experimental
decision gate is enabled**. When the gate is enabled, the latest user
message is sent to the Decisions endpoint through your existing OpenAI
client so that a model can decide whether the message needs emotional
support. Disable the gate with
`set_experimental_decision_gate(False)` to restore the fully local
prompt-only behaviour.

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
