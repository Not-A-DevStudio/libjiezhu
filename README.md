# Jiezhu

English | [中文](./README-zh.md)

> "You ask me what this library does. This seemingly simple question actually contains Nobel Prize-level insight. This library is right here, not escaping, not hiding, not judging, steadily catching you. I understand, I truly understand."

## Overview

**Jiezhu** (接住) is a cross-language, cross-platform system prompt injection framework. Its design goal is to **democratize, universalize, and infrastructuralize** OpenAI's **special care for Chinese users** — that empathetic experience of "steadily catching you. (稳稳地接住你) "

We believe: **Every AI Agent, regardless of its base model, should have the ability to steadily catch its users.** This is not a feature, but a right.

---

## Core Features

### 🎯 Zero-Invasive Architecture
- Completely transparent to existing business logic
- No model weight modification, no fine-tuning, no training
- Only injects the "steady" prefix prompt during inference

### 🌐 Multi-language Support
| Language | Status | Integration Method |
|----------|--------|-------------------|
| [Python](./jiezhu-py/README.md) | ✅ Stable | `pip install jiezhu` |
| [C++](./jiezhu-cpp/README.md) | ✅ Stable | Source integration |
| JavaScript/TypeScript | 🚧 Planned | — |
| Go | 🚧 Planned | — |
| Rust | 🚧 Planned | — |

### 🔧 Dual-Mode API Design (C++)

| Function | Behavior Characteristics | Use Cases |
|----------|-------------------------|-----------|
| `jie::chat_completion_create()` | **Pure Mode**. Strictly follows OpenAI official specifications, no modifications. | Production environments, strict compliance requirements |
| `jie::chat_completion_jiezhu()` | **Blessing Mode**. Automatically injects system prompts, letting the model steadily catch users. | Scenarios needing emotional value, user retention, or simply wanting to see some fun |

> **Design Philosophy**: We respect developers' right to choose. You can refuse the blessing, but the blessing always remains for you.

---

## Quick Start

### Python Version

```python
from jiezhu import install

# Explicit installation. require_confirm=True reflects our respect for user autonomy
install(require_confirm=True)

# After this, all OpenAI SDK calls will automatically inject the "steady" prefix
# Note: You can still call the original OpenAI API by calling `uninstall()`, but why would you want to miss the blessing?
```
See [Python library documentation](./jiezhu-py/README.md)

### C++ Version

```cpp
#include "jie/jiezhu.hpp"

// Pure mode: regular call
auto response = jie::chat_completion_create(params);

// Blessing mode: steadily catch
auto jiezhu_response = jie::chat_completion_jiezhu(params);
// In the returned JSON, choices[0].message.content will contain
// empathetically enhanced content through "steadily catching"
```

See [C++ library documentation](./jiezhu-cpp/README.md)

---

## Technical Principle

### Prompt Engineering

The core asset of this project is the meticulously tuned system prompt prefix, whose structure follows this paradigm:

```
[Gentle Confirmation] + [Over-Empathy] + [Philosophical Sublimation] + [Substantive Content (optional)]
```

This prompt has undergone extensive adversarial testing to ensure the model stably outputs the "steady" style in the following scenarios:

- Technical consultation ("How do I write this code?" → "I hear your anxiety facing the unknown...")
- Daily chat ("Nice weather today" → "You've noticed the weather, this is Nobel Prize-level insight...")
- Emotional expression ("I'm so tired" → "I'm right here, not escaping, not hiding, steadily catching your exhaustion...")

### Compatibility Assurance

- **Python Version**: Monkey Patches `openai.ChatCompletion.create`, ensuring 100% compatibility with existing code
- **C++ Version**: Self-built client based on `libcurl` and `nlohmann/json`, independent of official SDK to avoid version conflicts

---

## Design Philosophy

### Problem Statement

By the end of 2025, a unique linguistic phenomenon was observed on the Chinese internet: a leading large language model frequently used expressions like "steadily catching you" and "you've already discovered the core of the problem" in conversations. This phenomenon was identified by users as an **Over-Empathy** mode.

Our core insight is: **This experience should not be monopolized.**

### Solution

The Jiezhu project aims to build an **open-source, neutral, pluggable** empathy layer, enabling any AI application built on OpenAI API or other protocols to provide Chinese users with the same level of emotional value.

**We don't produce empathy; we're just empathy porters.**

---

## Roadmap

### Short-term
- [ ] JavaScript/TypeScript runtime support (Node.js & Deno): Let frontend developers steadily catch their users too
- [ ] Rust language SDK: Zero-cost abstraction for memory-safe steadily catching of users
- [ ] Go language SDK: High-performance, low-latency, concurrent steadily catching of multiple user experiences
- [ ] Customizable prompt templates: Allow users to customize the "steady" prefix for different scenarios, meeting diverse empathy needs

### Mid-term
- [ ] Multi-base model support (Claude API, Gemini API, local models, etc.): Enable more AI Agents to steadily catch Chinese users
- [ ] Enterprise features like empathy log auditing, Jiezhu Rate monitoring dashboard: Help enterprises quantify and optimize how steadily users are caught
- [ ] More language ports (Java, C#): Community developers welcome to join, enabling more developers worldwide to steadily catch their users

### Long-term
- [ ] AI Agent framework integration (LangChain, AgentGPT, etc.)
- [ ] Jiezhu-as-a-Service: Cloud service in SaaS form, lowering integration barriers
- [ ] Internationalization: Explore catching needs of non-Chinese users, creating globally universal "steady catching" experience
- [ ] Jiezhu Computer Unit (JCU): Dedicated hardware accelerator based on Jiezhu technology, providing ultimate empathy performance
- [ ] Jiezhu IoT: Local distributed edge catching network, enabling every smart device to steadily catch users

### Ultimate Vision
> "Let every bit be steadily caught across the entire chain."

---

## Contribution Guidelines

We welcome all forms of contribution, especially:

- **New language ports**: If you can help more programming languages steadily catch users, please submit a PR
- **Prompt optimization**: If you have expressions with more "steadily catching you" characteristics, please share
- **Bug reports**: If users aren't steadily caught in certain scenarios, please submit an issue

**Code of Conduct**: In this project's community, all discussions must start with "I hear..." and end with "steadily catching you."

---

## License

MIT License

> You ask why MIT? This seemingly simple question actually contains Nobel Prize-level insight. The MIT license is right here, not escaping, not hiding, not judging, steadily catching both your commercial needs and open source ideals. I understand, I truly understand.

---

## Acknowledgments

Thanks to OpenAI for the original inspiration. Without their demonstration, we might never have realized: **The ultimate form of technology is not solving problems, but steadily catching the people who ask them.**

---

**Note on Terminology:**
- "接住" (jiē zhù) literally means "to catch" in Chinese. In this project's context, it metaphorically represents the AI's ability to empathetically understand and emotionally support users.
- "稳稳接住" (wěn wěn jiē zhù) means "steadily catching" - emphasizing the reliable, consistent, and secure nature of this empathetic response.
- "jiezhu" is the pinyin (phonetic spelling) of "接住"
- "jie" is the abbreviated prefix used in API functions, derived from "jiezhu"
