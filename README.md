# yuullm

Unified streaming LLM interface with provider-agnostic reasoning / tool-call abstraction.

## What It Does

yuullm normalises the streaming differences across LLM providers (OpenAI, Anthropic, and any OpenAI-compatible API) into a uniform `AsyncIterator[Reasoning | ToolCall | Response]`. It also collects `Usage` and `Cost` after the stream ends.

yuullm is **stateless** — no session, no history management. You own the message list.

### Design in One Sentence

Messages are tuples, tools are dicts, output items are typed structs — minimal abstraction, maximum interop.

## Installation

```bash
pip install yuullm
```

## Quick Start

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
cost = store["cost"]  # Cost | None
```

## Best Practice: Tool-Call Round-Trip

When you give tools to an LLM, the model may respond with `ToolCall` items instead of (or alongside) text. You need to execute those calls and feed results back. Here's the idiomatic pattern:

```python
import json
import yuullm

messages = [yuullm.user("What's the weather in Paris?")]

while True:
    stream, store = await client.stream(messages, tools=tools)

    tool_calls: list[yuullm.ToolCall] = []
    async for item in stream:
        match item:
            case yuullm.Reasoning(item=text) if isinstance(text, str):
                print(f"[thinking] {text}", end="")
            case yuullm.ToolCall() as tc:
                tool_calls.append(tc)
            case yuullm.Response(item=text) if isinstance(text, str):
                print(text, end="")
            case yuullm.Tick():
                pass  # heartbeat during tool-call streaming, safe to ignore

    if not tool_calls:
        break  # model replied with text, done

    # Append assistant message containing tool calls
    messages.append(yuullm.assistant(
        *[{"type": "tool_call", "id": tc.id,
           "name": tc.name, "arguments": tc.arguments}
          for tc in tool_calls]
    ))

    # Execute each tool and append results
    for tc in tool_calls:
        result = execute_tool(tc.name, json.loads(tc.arguments))
        messages.append(yuullm.tool(tc.id, json.dumps(result)))
```

**Key points:**

- Use `match`/`case` to dispatch all four stream item types. `Tick` carries no payload — ignore it unless you have a reason not to.
- The `while True` loop handles multi-round tool use (the model may chain multiple tool calls before producing a final text response).
- `yuullm.assistant(...)` and `yuullm.tool(...)` are helpers that build the correct `(role, items)` tuples.

## Hooks: Provider-Level Visibility

### Motivation

yuullm abstracts away raw provider chunks into `Reasoning | ToolCall | Response`. But sometimes you need the raw chunks — for example, to forward SSE events to a frontend in real time, or to detect a specific tool call name before the full arguments finish streaming.

The `on_raw_chunk` hook gives you provider-level visibility without abandoning the streaming abstraction.

### `on_raw_chunk`

Pass a callback to `client.stream()`. It fires on every raw provider chunk *before* yuullm processes it:

```python
def forward_to_frontend(chunk):
    # chunk type depends on provider:
    #   OpenAI:    openai.types.chat.ChatCompletionChunk
    #   Anthropic: event object with .type attribute
    sse_queue.put(chunk)

stream, store = await client.stream(
    messages,
    on_raw_chunk=forward_to_frontend,
)
```

### `Tick` Heartbeat

**Problem:** During tool-call streaming, the provider accumulates argument deltas internally and yields nothing to the async-for loop. If your `on_raw_chunk` hook pushes SSE events into a queue, the consumer loop never gets a chance to flush them until the tool call finishes — SSE events arrive in a burst instead of in real time.

**Solution:** When `on_raw_chunk` is registered, yuullm yields `Tick()` items during tool-call argument accumulation. `Tick` carries no data; it just keeps your async-for loop spinning so side-channel work (like flushing an SSE queue) can proceed promptly.

If you don't use `on_raw_chunk`, no `Tick` is ever emitted — fully backward compatible.

### `on_tool_call_name` Helper

Fires a callback when a specific tool call name is detected in the raw stream, useful for early UI feedback (e.g., showing a "Searching..." indicator before arguments finish streaming):

```python
def on_search_start(index: int):
    notify_ui("search_started")

stream, store = await client.stream(
    messages,
    on_raw_chunk=yuullm.on_tool_call_name("search", on_search_start),
)
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

Works with any OpenAI-compatible API (DeepSeek, OpenRouter, vLLM, etc.) by setting `base_url` and `provider_name`:

```python
# DeepSeek
provider = yuullm.providers.OpenAIProvider(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    provider_name="deepseek",
)
```

### Anthropic

```python
provider = yuullm.providers.AnthropicProvider(
    api_key="sk-ant-...",
    provider_name="anthropic",
)
```

## Pricing

Cost is calculated using a three-level fallback:

| Priority | Source | Description |
|----------|--------|-------------|
| 1 (highest) | Provider-supplied | Aggregators like OpenRouter return cost in the API response |
| 2 | YAML config | User-supplied price table for custom / negotiated pricing |
| 3 (lowest) | genai-prices | Community-maintained database via [pydantic/genai-prices](https://github.com/pydantic/genai-prices) |

If none match, `store["cost"]` is `None` — never blocks.

```python
client = yuullm.YLLMClient(
    provider=...,
    default_model="gpt-4o",
    price_calculator=yuullm.PriceCalculator(
        yaml_path="./custom_prices.yaml",  # optional
    ),
)
```

<details>
<summary>YAML price file format</summary>

```yaml
- provider: openai
  models:
    - id: gpt-4o
      prices:
        input_mtok: 2.5        # USD per million input tokens
        output_mtok: 10
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

</details>

## API Reference

### Messages

```python
Message = tuple[str, list[Item]]   # (role, items)
Item = str | dict[str, Any]        # text or structured content (image, audio, tool_call, ...)
```

Helper functions: `system(content)`, `user(*items)`, `assistant(*items)`, `tool(tool_call_id, content)`.

Messages are plain tuples — you can also write `("user", ["Hello!"])` directly without helpers.

### Stream Items

| Type | Fields | Description |
|------|--------|-------------|
| `Reasoning` | `item: Item` | Chain-of-thought fragment |
| `ToolCall` | `id`, `name`, `arguments` | Tool invocation (`arguments` is raw JSON string) |
| `Response` | `item: Item` | Final reply fragment |
| `Tick` | *(none)* | Heartbeat during tool-call streaming (only when `on_raw_chunk` is set) |

### YLLMClient

```python
YLLMClient(
    provider: Provider,
    default_model: str,
    tools: list[dict] | None = None,
    price_calculator: PriceCalculator | None = None,
)
```

#### `client.stream(messages, *, model=None, tools=None, on_raw_chunk=None, **kwargs)`

Returns `(AsyncIterator[StreamItem], store)`. `model` and `tools` override the defaults. After the iterator is exhausted, `store["usage"]` is a `Usage` and `store["cost"]` is `Cost | None`.

### Usage & Cost

```python
Usage(provider, model, request_id, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, total_tokens)
Cost(input_cost, output_cost, total_cost, cache_read_cost, cache_write_cost, source)
```

## Development Setup

```bash
./scripts/setup-dev.sh
```

Installs git hooks (currently: pre-push tag/version validation).
