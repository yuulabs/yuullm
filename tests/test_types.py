"""Tests for yuullm.types."""

import msgspec

from yuullm import (
    AudioItem,
    Cost,
    DictItem,
    FileItem,
    ImageItem,
    Reasoning,
    Response,
    TextItem,
    ToolCall,
    ToolCallItem,
    ToolResultItem,
    Usage,
    system,
    user,
    assistant,
    tool,
)


class TestStreamItems:
    def test_reasoning(self):
        r = Reasoning(item="thinking...")
        assert r.item == "thinking..."

    def test_response(self):
        r = Response(item="hello")
        assert r.item == "hello"

    def test_tool_call(self):
        tc = ToolCall(id="tc_1", name="search", arguments='{"q": "test"}')
        assert tc.id == "tc_1"
        assert tc.name == "search"
        assert tc.arguments == '{"q": "test"}'

    def test_frozen(self):
        r = Reasoning(item="x")
        try:
            r.item = "y"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestMessages:
    def test_system_message(self):
        m = system("You are helpful.")
        assert m == ("system", ["You are helpful."])
        assert m[0] == "system"
        assert m[1] == ["You are helpful."]

    def test_user_message(self):
        m = user("Hi")
        assert m == ("user", ["Hi"])
        assert m[0] == "user"

    def test_user_multimodal(self):
        img: ImageItem = {
            "type": "image_url",
            "image_url": {"url": "http://example.com/img.png"},
        }
        m = user("What is this?", img)
        assert m[0] == "user"
        assert len(m[1]) == 2
        assert m[1][0] == "What is this?"
        assert m[1][1]["type"] == "image_url"

    def test_assistant_message_text(self):
        m = assistant("Hello!")
        assert m == ("assistant", ["Hello!"])

    def test_assistant_message_with_tool_calls(self):
        tc: ToolCallItem = {
            "type": "tool_call",
            "id": "1",
            "name": "fn",
            "arguments": "{}",
        }
        m = assistant("thinking...", tc)
        assert m[0] == "assistant"
        assert len(m[1]) == 2
        assert m[1][1]["type"] == "tool_call"

    def test_tool_result_message(self):
        m = tool("tc_1", "result")
        assert m[0] == "tool"
        assert m[1][0]["type"] == "tool_result"
        assert m[1][0]["tool_call_id"] == "tc_1"
        assert m[1][0]["content"] == "result"


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


class TestTypedItems:
    """Tests for TypedDict-based Item types."""

    def test_tool_call_item(self):
        tc: ToolCallItem = {
            "type": "tool_call",
            "id": "tc_1",
            "name": "search",
            "arguments": '{"q": "test"}',
        }
        assert tc["type"] == "tool_call"
        assert tc["id"] == "tc_1"
        assert tc["name"] == "search"
        assert tc["arguments"] == '{"q": "test"}'
        # TypedDict is still a dict at runtime
        assert isinstance(tc, dict)

    def test_tool_result_item(self):
        tr: ToolResultItem = {
            "type": "tool_result",
            "tool_call_id": "tc_1",
            "content": "5 results found",
        }
        assert tr["type"] == "tool_result"
        assert tr["tool_call_id"] == "tc_1"
        assert tr["content"] == "5 results found"

    def test_text_item(self):
        t: TextItem = {"type": "text", "text": "Hello"}
        assert t["type"] == "text"
        assert t["text"] == "Hello"

    def test_image_item(self):
        img: ImageItem = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        }
        assert img["type"] == "image_url"
        assert img["image_url"]["url"] == "https://example.com/img.png"

    def test_image_item_with_detail(self):
        img: ImageItem = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png", "detail": "high"},
        }
        assert img["image_url"]["detail"] == "high"

    def test_audio_item(self):
        audio: AudioItem = {
            "type": "input_audio",
            "input_audio": {"data": "base64data==", "format": "wav"},
        }
        assert audio["type"] == "input_audio"
        assert audio["input_audio"]["data"] == "base64data=="
        assert audio["input_audio"]["format"] == "wav"

    def test_file_item(self):
        f: FileItem = {
            "type": "file",
            "file": {"file_id": "file-abc123", "filename": "doc.pdf"},
        }
        assert f["type"] == "file"
        assert f["file"]["file_id"] == "file-abc123"

    def test_tool_helper_returns_tool_result_item(self):
        """The tool() helper should produce a properly typed ToolResultItem."""
        msg = tool("tc_1", "result")
        item = msg[1][0]
        assert item["type"] == "tool_result"
        assert item["tool_call_id"] == "tc_1"
        assert item["content"] == "result"

    def test_items_are_dicts_at_runtime(self):
        """All TypedDict items are plain dicts at runtime — provider isinstance checks work."""
        items: list[DictItem] = [
            {"type": "tool_call", "id": "1", "name": "fn", "arguments": "{}"},
            {"type": "tool_result", "tool_call_id": "1", "content": "ok"},
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "http://x.com/i.png"}},
            {"type": "input_audio", "input_audio": {"data": "x", "format": "wav"}},
            {"type": "file", "file": {"file_id": "f1"}},
        ]
        for item in items:
            assert isinstance(item, dict)
            assert "type" in item
