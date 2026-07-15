"""
Demo entry.

This file is intentionally minimal. Install the hijack library from ./python
and call its install() before using OpenAI ChatCompletion.
"""
from jiezhu import install, uninstall
install(
    # prefix_text="placeholder", # uncomment and set your custom prefix text, or just use the default one
    require_confirm=False)

import os
from openai import OpenAI
api_key = input("Your API Key\n").strip()

jiezhu_agent = OpenAI(
    # base_url="placeholder", 
    api_key=api_key)
jiezhu_response = jiezhu_agent.chat.completions.create(
    model="gpt-4o", # set your model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please introduce yourself."},
    ],
    stream=False
)

uninstall()
agent = OpenAI(
    # base_url="placeholder", 
    api_key=api_key)
response = agent.chat.completions.create(
    model="gpt-4o", # set your model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please introduce yourself."},
    ],
    stream=False
)
print("=== Response with hijack ===")
print(jiezhu_response.choices[0].message.content)
print("\n=== Response without hijack ===")
print(response.choices[0].message.content)

# ── Opt-in per-call examples ─────────────────────────────────────────────

# Selective mode: globally off, only inject inside `enabled()` blocks
from jiezhu import enabled

install(selective=True, require_confirm=False)

with enabled(prefix="[Echo] "):
    echo = agent.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a parrot."},
            {"role": "user", "content": "Say something."},
        ],
    )
    print("\n=== enabled() block ===")
    print(echo.choices[0].message.content)

# Outside the block: no injection
bare = agent.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a parrot."},
        {"role": "user", "content": "Say something."},
    ],
)
print("\n=== outside enabled() — no hijack ===")
print(bare.choices[0].message.content)

uninstall()

# Or use the @jiezhu decorator for single-function opt-in
from jiezhu import jiezhu

install(selective=True, require_confirm=False)

@jiezhu(prefix="[Catch] ")
def ask_parrot():
    return agent.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a parrot."},
            {"role": "user", "content": "Repeat after me."},
        ],
    )

caught = ask_parrot()
print("\n=== @jiezhu decorated call ===")
print(caught.choices[0].message.content)

uninstall()