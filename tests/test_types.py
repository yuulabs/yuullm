"""Tests for yuullm.types."""

import msgspec

from yuullm import (
    AssistantMessage,
    Cost,
    Reasoning,
    Response,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
)


class TestStreamItems:
    def test_reasoning(self):
        r = Reasoning(text="thinking...")
        assert r.text == "thinking..."

    def test_response(self):
        r = Response(text="hello")
        assert r.text == "hello"

    def test_tool_call(self):
        tc = ToolCall(id="tc_1", name="search", arguments='{"q": "test"}')
        assert tc.id == "tc_1"
        assert tc.name == "search"
        assert tc.arguments == '{"q": "test"}'

    def test_frozen(self):
        r = Reasoning(text="x")
        try:
            r.text = "y"
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestMessages:
    def test_system_message(self):
        m = SystemMessage(content="You are helpful.")
        assert m.role == "system"
        assert m.content == "You are helpful."

    def test_user_message(self):
        m = UserMessage(content="Hi")
        assert m.role == "user"

    def test_assistant_message_text(self):
        m = AssistantMessage(content="Hello!")
        assert m.role == "assistant"
        assert m.content == "Hello!"
        assert m.tool_calls is None

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="1", name="fn", arguments="{}")
        m = AssistantMessage(tool_calls=[tc])
        assert m.tool_calls == [tc]

    def test_tool_result_message(self):
        m = ToolResultMessage(tool_call_id="1", content="result")
        assert m.role == "tool"
        assert m.tool_call_id == "1"


class TestToolSpec:
    def test_basic(self):
        ts = ToolSpec(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert ts.name == "search"


class TestUsage:
    def test_defaults(self):
        u = Usage(provider="openai", model="gpt-4o")
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_read_tokens == 0
        assert u.cache_write_tokens == 0
        assert u.total_tokens is None
        assert u.request_id is None

    def test_full(self):
        u = Usage(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="req_123",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=20,
            cache_write_tokens=10,
            total_tokens=180,
        )
        assert u.input_tokens == 100
        assert u.total_tokens == 180


class TestCost:
    def test_defaults(self):
        c = Cost(input_cost=0.01, output_cost=0.02, total_cost=0.03)
        assert c.cache_read_cost == 0.0
        assert c.cache_write_cost == 0.0
        assert c.source == ""

    def test_full(self):
        c = Cost(
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.035,
            cache_read_cost=0.003,
            cache_write_cost=0.002,
            source="yaml",
        )
        assert c.source == "yaml"


class TestSerialization:
    def test_usage_roundtrip(self):
        u = Usage(provider="openai", model="gpt-4o", input_tokens=100, output_tokens=50)
        data = msgspec.json.encode(u)
        u2 = msgspec.json.decode(data, type=Usage)
        assert u == u2

    def test_cost_roundtrip(self):
        c = Cost(input_cost=0.01, output_cost=0.02, total_cost=0.03, source="yaml")
        data = msgspec.json.encode(c)
        c2 = msgspec.json.decode(data, type=Cost)
        assert c == c2
