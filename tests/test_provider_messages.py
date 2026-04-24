from __future__ import annotations

import yuullm
from yuullm.providers.anthropic import AnthropicMessagesProvider
from yuullm.providers.openai import OpenAIChatCompletionProvider
from yuullm.providers.openrouter import OpenRouterProvider
from yuullm.types import with_last_item_cache_control


def test_openai_converts_message_structs_and_provider_extra() -> None:
    messages = [
        yuullm.system("sys", name="system_name"),
        yuullm.user("hello"),
        yuullm.assistant(
            "using tool",
            {
                "type": "tool_call",
                "id": "call_1",
                "name": "search",
                "arguments": '{"q":"x"}',
            },
        ),
        yuullm.tool("call_1", "done"),
    ]

    assert OpenAIChatCompletionProvider._convert_messages(messages) == [
        {"role": "system", "content": "sys", "name": "system_name"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "using tool",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
    ]


def test_anthropic_extracts_system_and_converts_tool_results() -> None:
    messages = [
        yuullm.system("sys"),
        yuullm.user("hello", metadata={"k": "v"}),
        yuullm.tool("call_1", [{"type": "text", "text": "done"}]),
    ]

    system, rest = AnthropicMessagesProvider._extract_system(messages)
    assert system == [{"type": "text", "text": "sys"}]
    assert AnthropicMessagesProvider._convert_messages(rest) == [
        {"role": "user", "content": "hello", "metadata": {"k": "v"}},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [{"type": "text", "text": "done"}],
                }
            ],
        },
    ]


def test_openrouter_preserves_cache_control_with_message_structs() -> None:
    message = with_last_item_cache_control(
        yuullm.user("cache me"), {"type": "ephemeral"}
    )

    assert OpenRouterProvider._convert_messages([message]) == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "cache me",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
