# yuullm

Unified streaming LLM interface with provider-agnostic reasoning / tool-call abstraction.

## Overview

yuullm provides a standardised streaming abstraction layer over different LLM providers. It has two core responsibilities:

1. **Stream standardisation** — normalises differences in thinking formats (`reasoning_content` / `thinking` / …) and tool-call protocols across providers, outputting a uniform `AsyncIterator[Reasoning | ToolCall | Response]` stream.
2. **Usage + Cost collection** — after the stream ends, structured `Usage` (from the API) and `Cost` (calculated by yuullm) are available via a store dict.

yuullm is **stateless** — it has no session concept and does not maintain conversation history.

### Design Philosophy

yuullm intentionally avoids heavy abstractions:

- **Messages are tuples**, not classes. `("role", [items])` — no `SystemMessage`, `UserMessage` imports needed.
- **Tools are dicts**, not a custom `ToolSpec`. Pass `list[dict]` directly — works seamlessly with `yuutools.ToolManager.specs()`, but with zero dependency.
- **Helper functions** `system()`, `user()`, `assistant()`, `tool()` for ergonomic one-liner message construction.
- **Multimodal native** — `Item = str | dict`, so images, audio, and structured content are first-class.

## Installation

```bash
pip install yuullm
```

## Quick Start

### Basic Chat (with helpers)

```python
import yuullm

client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIProvider(api_key="sk-..."),
    default_model="gpt-4o",
)

messages = [
    yuullm.system("You are a helpful assistant."),
    yuullm.user("What is 2+2?"),
]

stream, store = await client.stream(messages)
async for item in stream:
    match item:
        case yuullm.Reasoning(text=t):
            print(f"[thinking] {t}", end="")
        case yuullm.Response(text=t):
            print(t, end="")

# After stream ends
usage = store["usage"]
print(f"\nTokens: {usage.input_tokens} in / {usage.output_tokens} out")
```

### Basic Chat (raw tuples)

Messages are just `(role, items)` tuples — no imports needed beyond `yuullm`:

```python
import yuullm

client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIProvider(api_key="sk-..."),
    default_model="gpt-4o",
)

messages = [
    ("system", ["You are a helpful assistant."]),
    ("user", ["What is 2+2?"]),
]

stream, store = await client.stream(messages)
async for item in stream:
    match item:
        case yuullm.Reasoning(text=t):
            print(f"[thinking] {t}", end="")
        case yuullm.Response(text=t):
            print(t, end="")
```

### Multimodal (with helpers)

```python
messages = [
    yuullm.system("You are a vision assistant."),
    yuullm.user("What is in this image?", {
        "type": "image_url",
        "image_url": {"url": "https://example.com/photo.png"},
    }),
]
```

### Multimodal (raw tuples)

```python
messages = [
    ("system", ["You are a vision assistant."]),
    ("user", [
        "What is in this image?",
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.png"}},
    ]),
]
```

### Tool Calling (with helpers)

Tools are plain `list[dict]` — pass json_schema dicts directly:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}]

client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIProvider(api_key="sk-..."),
    default_model="gpt-4o",
    tools=tools,
)

messages = [yuullm.user("What's the weather in Tokyo?")]
stream, store = await client.stream(messages)

async for item in stream:
    match item:
        case yuullm.ToolCall(id=tid, name=name, arguments=args):
            print(f"Tool call: {name}({args})")
        case yuullm.Response(text=t):
            print(t, end="")
```

Or override tools per-request:

```python
stream, store = await client.stream(messages, tools=other_tools)
```

### Integration with yuutools

```python
import yuutools as yt
import yuullm

manager = yt.ToolManager([search_tool, calculator_tool])

# manager.specs() returns list[dict] in OpenAI function-calling format
# pass directly to yuullm — no conversion needed
stream, store = await client.stream(messages, tools=manager.specs())
```

### Multi-turn Conversation (with helpers)

yuullm is stateless — you manage the message list yourself:

```python
messages = [
    yuullm.system("You are a helpful assistant."),
    yuullm.user("Hi, my name is Alice."),
]

# First turn
stream, store = await client.stream(messages)
reply = ""
async for item in stream:
    if isinstance(item, yuullm.Response):
        reply += item.text

# Append assistant reply and next user message
messages.append(yuullm.assistant(reply))
messages.append(yuullm.user("What's my name?"))

# Second turn
stream, store = await client.stream(messages)
async for item in stream:
    if isinstance(item, yuullm.Response):
        print(item.text, end="")
```

### Multi-turn Conversation (raw tuples)

```python
messages = [
    ("system", ["You are a helpful assistant."]),
    ("user", ["Hi, my name is Alice."]),
]

# First turn
stream, store = await client.stream(messages)
reply = ""
async for item in stream:
    if isinstance(item, yuullm.Response):
        reply += item.text

# Append assistant reply and next user message
messages.append(("assistant", [reply]))
messages.append(("user", ["What's my name?"]))

# Second turn
stream, store = await client.stream(messages)
async for item in stream:
    if isinstance(item, yuullm.Response):
        print(item.text, end="")
```

### Tool Call Round-trip (with helpers)

A full tool-use loop: model calls a tool, you execute it, then feed the result back:

```python
import json

messages = [yuullm.user("What's the weather in Paris?")]

stream, store = await client.stream(messages)
tool_calls = []
async for item in stream:
    match item:
        case yuullm.ToolCall() as tc:
            tool_calls.append(tc)
        case yuullm.Response(text=t):
            print(t, end="")

if tool_calls:
    # Append assistant message with tool calls as dicts
    messages.append(yuullm.assistant(
        *[{"type": "tool_call", "id": tc.id, "name": tc.name, "arguments": tc.arguments}
          for tc in tool_calls]
    ))

    # Execute each tool and append results
    for tc in tool_calls:
        result = execute_tool(tc.name, json.loads(tc.arguments))  # your function
        messages.append(yuullm.tool(tc.id, json.dumps(result)))

    # Continue the conversation
    stream, store = await client.stream(messages)
    async for item in stream:
        if isinstance(item, yuullm.Response):
            print(item.text, end="")
```

### Tool Call Round-trip (raw tuples)

```python
import json

messages = [("user", ["What's the weather in Paris?"])]

stream, store = await client.stream(messages)
tool_calls = []
async for item in stream:
    match item:
        case yuullm.ToolCall() as tc:
            tool_calls.append(tc)
        case yuullm.Response(text=t):
            print(t, end="")

if tool_calls:
    # Append assistant message with tool call dicts
    messages.append(("assistant", [
        {"type": "tool_call", "id": tc.id, "name": tc.name, "arguments": tc.arguments}
        for tc in tool_calls
    ]))

    # Execute each tool and append results
    for tc in tool_calls:
        result = execute_tool(tc.name, json.loads(tc.arguments))
        messages.append(("tool", [
            {"type": "tool_result", "tool_call_id": tc.id, "content": json.dumps(result)}
        ]))

    # Continue the conversation
    stream, store = await client.stream(messages)
    async for item in stream:
        if isinstance(item, yuullm.Response):
            print(item.text, end="")
```

### Cost Tracking

```python
client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIProvider(api_key="sk-..."),
    default_model="gpt-4o",
    price_calculator=yuullm.PriceCalculator(
        yaml_path="./custom_prices.yaml",  # optional, for custom pricing
    ),
)

stream, store = await client.stream(messages)
async for item in stream:
    ...  # consume the stream

usage: yuullm.Usage = store["usage"]
cost: yuullm.Cost | None = store["cost"]

print(f"Tokens: {usage.input_tokens} in / {usage.output_tokens} out")
print(f"Cache:  {usage.cache_read_tokens} read / {usage.cache_write_tokens} write")
if cost:
    print(f"Cost: ${cost.total_cost:.6f} (source: {cost.source})")
else:
    print("Cost: unavailable (model price not found)")
```

## Providers

### OpenAI / OpenAI-compatible

```python
provider = yuullm.providers.OpenAIProvider(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",  # or any compatible endpoint
    provider_name="openai",                 # used for price lookup
)
```

Works with any OpenAI-compatible API (Azure, OpenRouter, vLLM, etc.) by setting `base_url` and `provider_name`.

### Anthropic

```python
provider = yuullm.providers.AnthropicProvider(
    api_key="sk-ant-...",
    provider_name="anthropic",
)
```

Handles Anthropic-specific streaming events including `thinking_delta` for extended thinking and `tool_use` content blocks.

## Pricing

Cost is calculated using a three-level priority system:

| Priority | Source | Description |
|----------|--------|-------------|
| 1 (highest) | Provider-supplied | Aggregators like OpenRouter / LiteLLM return cost in the API response |
| 2 | YAML config | User-supplied price table for custom / negotiated pricing |
| 3 (lowest) | genai-prices | Community-maintained database via [pydantic/genai-prices](https://github.com/pydantic/genai-prices) |

If none of the sources can determine the price, `store["cost"]` is `None`.

### YAML Price File Format

```yaml
- provider: openai
  models:
    - id: gpt-4o
      prices:
        input_mtok: 2.5        # USD per million input tokens
        output_mtok: 10         # USD per million output tokens
        cache_read_mtok: 1.25   # optional

- provider: anthropic
  models:
    - id: claude-sonnet-4-20250514
      prices:
        input_mtok: 3
        output_mtok: 15
        cache_read_mtok: 0.3
        cache_write_mtok: 3.75
```

Matching is exact on `(provider, model_id)`. No fuzzy matching.

## API Reference

### YLLMClient

```python
YLLMClient(
    provider: Provider,
    default_model: str,
    tools: list[dict] | None = None,              # json_schema tool dicts
    price_calculator: PriceCalculator | None = None,
)
```

#### `client.stream(messages, *, model=None, tools=None, **kwargs)`

Returns `(AsyncIterator[StreamItem], store)`. The `model` and `tools` params override the defaults set at init.

### Messages

```python
Message = tuple[str, list[Item]]  # (role, items)
Item = str | dict[str, Any]       # text or structured content
History = list[Message]
```

Helper functions:

| Function | Signature | Example |
|----------|-----------|---------|
| `system` | `system(content: str)` | `system("You are helpful.")` |
| `user` | `user(*items: Item)` | `user("Hello!")` / `user("Look:", {"type": "image_url", ...})` |
| `assistant` | `assistant(*items: Item)` | `assistant("Sure!", {"type": "tool_call", ...})` |
| `tool` | `tool(tool_call_id: str, content: str)` | `tool("tc_1", '{"result": 42}')` |

Tool call items in assistant messages use this dict shape:

```python
{"type": "tool_call", "id": "...", "name": "...", "arguments": "..."}
```

Tool result items in tool messages use this dict shape:

```python
{"type": "tool_result", "tool_call_id": "...", "content": "..."}
```

### Stream Items

| Type | Fields | Description |
|------|--------|-------------|
| `Reasoning` | `text: str` | Chain-of-thought / extended thinking fragment |
| `ToolCall` | `id: str`, `name: str`, `arguments: str` | Tool invocation request (`arguments` is raw JSON) |
| `Response` | `text: str` | Final text reply fragment |

### Usage

```python
Usage(
    provider: str,
    model: str,
    request_id: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    total_tokens: int | None = None,
)
```

### Cost

```python
Cost(
    input_cost: float,
    output_cost: float,
    total_cost: float,
    cache_read_cost: float = 0.0,
    cache_write_cost: float = 0.0,
    source: str = "",  # "provider" | "yaml" | "genai-prices"
)
```
