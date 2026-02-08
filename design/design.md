# yuullm Design

## Overview

yuullm 提供统一的流式 LLM 接口抽象层。它有两个核心职责：

1. **流标准化** -- 屏蔽不同 API 协议在思考格式（`reasoning` / `reasoning_content` / `reasoning_details.summary`）和工具调用协议上的差异，输出标准化的 `AsyncIterator[Reasoning | ToolCall | Response]` 流。
2. **Usage + Cost 收集** -- 在流结束后，通过 store 提供结构化的 `Usage`（来自 API 原始数据）和 `Cost`（yuullm 自行计算）。

yuullm **不负责埋点**（无 session 概念）。它只关注：把请求发出去，把流标准化地收回来，再把 usage/cost 算清楚。

### 设计原则：轻量抽象

yuullm 刻意避免引入过重的类型抽象。核心哲学：

- **消息是元组**，不是类。`Message = (role, items)` -- 用户不需要导入 `SystemMessage`, `UserMessage` 等一堆类名。
- **工具是 dict**，不是 ToolSpec。直接使用 `list[dict]`，与 yuutools 的 `manager.specs()` 输出无缝对接，但不产生硬依赖。
- **输出流用 struct** -- `Reasoning | ToolCall | Response` 是 msgspec 冻结结构体，因为这些是 yuullm 产出的数据，需要类型安全。
- **提供便利函数** -- `system()`, `user()`, `assistant()`, `tool()` 让消息构造一行搞定。

## API Type 与 Provider 的分离

yuullm 将两个正交概念明确分离：

### api_type（API 协议类型）

指的是使用哪种 wire protocol 与 LLM 服务通信：

| api_type | 说明 | 对应端点 |
|---|---|---|
| `openai-chat-completion` | OpenAI Chat Completion API | `/v1/chat/completions` |
| `openai-responses` | OpenAI Responses API | `/v1/responses` |
| `anthropic-messages` | Anthropic Messages API | `/v1/messages` |

这是**协议层面**的区分。`openai-chat-completion` 和 `openai-responses` 是两个完全不同的 API 接口，请求/响应格式不同，能力也不同（例如某些 provider 不支持 responses API）。

### provider（供应商）

指的是实际的模型供应商 / 服务商：

- `openai` -- OpenAI 官方
- `deepseek` -- DeepSeek（使用 OpenAI-compatible chat completion API）
- `openrouter` -- OpenRouter（聚合多家模型，使用 OpenAI-compatible API）
- `anthropic` -- Anthropic 官方
- `together` -- Together AI
- `groq` -- Groq

provider 决定了 `base_url`、`api_key` 的来源、以及 pricing 查询时的供应商标识。

### 为什么要分离？

之前的设计将 `provider` 笼统地设为 `"openai"` / `"anthropic"`，这导致：

1. **语义混淆** -- DeepSeek 使用 OpenAI-compatible API，但它不是 OpenAI。`provider="openai"` 会导致 pricing 查询错误。
2. **API 差异被忽略** -- OpenAI 的 Chat Completion 和 Responses 是两个完全不同的 API，不应该用同一个 Provider 类处理。
3. **扩展困难** -- 新增一个使用 OpenAI-compatible API 的供应商时，不应该需要新建 Provider 类，只需要传入不同的 `provider_name` 和 `base_url`。

分离后：
- `api_type` 决定使用哪个 Provider **类**（协议实现）
- `provider` 决定使用哪个**供应商**（通过构造参数传入）

```python
# DeepSeek: 使用 OpenAI Chat Completion 协议，但供应商是 deepseek
client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIChatCompletionProvider(
        api_key="sk-...",
        base_url="https://api.deepseek.com/v1",
        provider_name="deepseek",
    ),
    default_model="deepseek-chat",
)

# OpenRouter: 使用 OpenAI Chat Completion 协议，但供应商是 openrouter
client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIChatCompletionProvider(
        api_key="sk-or-...",
        base_url="https://openrouter.ai/api/v1",
        provider_name="openrouter",
    ),
    default_model="anthropic/claude-sonnet-4-20250514",
)
```

## Provider 抽象：隔离 API 协议细节

不同 API 协议的差异由 Provider 类内部消化，对上层暴露统一接口。

### 需要隔离的差异

| 差异维度 | OpenAI Chat Completion | Anthropic Messages | 说明 |
|---|---|---|---|
| **响应结构** | `response.choices[0].delta` | `event.delta` (SSE) | 流式响应的数据路径完全不同 |
| **思考格式** | `reasoning_content` / `reasoning.summary` | `content[type=thinking].thinking` | 各家 CoT 表示方式不统一 |
| **工具调用** | `tool_calls` 数组 (含 index) | `content[type=tool_use]` | 工具调用的编码方式不同 |
| **Usage 字段** | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` | 字段名和结构不同 |
| **Cache 计量** | `cached_tokens` (嵌套在 details 中) | `cache_creation_input_tokens` / `cache_read_input_tokens` (顶层) | cache 指标位置和语义不同 |

### 隔离策略

每个 Provider 实现内部处理所有协议特有逻辑，对外只暴露统一的 `stream()` 协议。具体来说：

- **消息转换** -- Provider 内部将 `(role, items)` 元组转换为协议特有的消息格式。
- **工具转换** -- Provider 接受 `list[dict]` 格式的工具定义（兼容 OpenAI function-calling 格式和裸 dict），内部转换为协议格式。
- **流解析** -- Provider 内部将协议私有的 SSE/streaming 格式映射为标准的 `StreamItem`（Reasoning / ToolCall / Response）。
- **Usage 提取** -- Provider 在流结束后从响应中提取 usage 信息，统一映射到 `Usage` struct。不同协议的字段名差异（如 `prompt_tokens` vs `input_tokens`）在 Provider 内部消化。
- **上层无感** -- Client 和使用者不需要知道底层是哪种 API 协议，拿到的都是同一套类型。

## Key Concepts

### Message

消息是简单的元组：

```python
Message = tuple[str, list[Item]]  # (role, items)
Item = str | DictItem             # 文本 或 TypedDict 结构化内容
DictItem = ToolCallItem | ToolResultItem | TextItem | ImageItem | AudioItem | FileItem
History = list[Message]
```

role 是字符串：`"system"`, `"user"`, `"assistant"`, `"tool"`。items 是内容列表，可以是纯文本字符串，也可以是 TypedDict（图片、音频、工具调用、工具结果等）。所有 TypedDict 结构对齐 OpenAI API 格式。

便利函数：

```python
import yuullm

yuullm.system("You are helpful.")
yuullm.user("Hello!")
yuullm.user("What is this?", {"type": "image_url", "image_url": {"url": "..."}})
yuullm.assistant("Let me search.", {"type": "tool_call", "id": "tc_1", "name": "search", "arguments": '{"q": "test"}'})
yuullm.tool("tc_1", "Search returned 5 results.")
```

### StreamItem

流中的每个元素是以下三种之一：

- **Reasoning** -- 模型的思考过程片段（chain-of-thought / extended thinking），内容是 `Item` 类型（支持文本或多模态）
- **ToolCall** -- 模型发起的工具调用请求（包含 id, name, arguments）
- **Response** -- 模型的最终回复片段，内容是 `Item` 类型（支持文本或多模态）

### Tools

工具定义使用 `list[dict]` -- 原始 json_schema 字典。yuullm 不定义自己的 ToolSpec 类。

直接使用 yuutools 输出：

```python
import yuutools as yt
manager = yt.ToolManager([...])
tools = manager.specs()  # list[dict], OpenAI function-calling 格式

# 直接传给 yuullm
stream, store = await client.stream(messages, tools=tools)
```

也可以手写：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
    }
}]
```

Provider 内部负责将 json_schema 格式转换为各协议的工具格式。

### Provider

对接一个具体的 LLM API 协议。每个 Provider 实现统一的 `stream()` 协议，内部负责将协议私有格式映射到 `StreamItem`，并提取标准化的 `Usage`。

Provider 暴露两个属性：
- `api_type` -- API 协议标识（如 `"openai-chat-completion"`）
- `provider` -- 供应商标识（如 `"openai"`, `"deepseek"`）

### Client

面向用户的入口。持有一个 Provider 和一个可选的 `PriceCalculator`，暴露简洁的 `stream()` 方法，并在调用结束后通过 store 返回 `Usage` 和 `Cost`。

### Store

store 是一个 mutable dict，在流结束后包含：
- `store["usage"]` -- `Usage` struct（来自 API 的原始 token 计数）
- `store["cost"]` -- `Cost` struct（yuullm 计算的费用），若无法计算则为 `None`

## Store 中存什么

### Usage（来自 API）

Usage 由 Provider 在流结束后从 API 响应中提取。不同协议返回的原始字段不同（见上表），Provider 负责将它们映射到统一的 `Usage` struct。

Usage 中的 `provider` 字段存储的是**供应商名称**（如 `"deepseek"`），用于 pricing 查询。

### Cost（yuullm 计算）

**API 只返回 token 数量（usage），不返回费用。** Cost 是 yuullm 根据 usage + 价格信息自行计算得出的。

价格来源按优先级从高到低：

1. **供应商自带价格** -- 部分聚合类供应商（如 OpenRouter、LiteLLM）在 API 响应中直接包含费用信息。若 Provider 能提取到，直接使用，优先级最高。
2. **用户 YAML 配置文件** -- 用户启动时传入路径，包含自定义价格表。适用于私有部署、谈判价格、或上述来源缺失的场景。
3. **genai-prices 回退** -- 若以上均无，使用 [pydantic/genai-prices](https://github.com/pydantic/genai-prices) 作为回退。通过 `(provider_id, model)` 查询。若匹配失败，cost 为 `None`。

## Price Source 详解

### 来源 1: 供应商自带

某些聚合供应商在响应中直接返回费用：
- **OpenRouter**: 响应头或 `generation` 字段中可能包含 `total_cost`
- **LiteLLM proxy**: 可配置返回 `_litellm_response_cost`

当 Provider 检测到此类字段时，直接将其写入 Cost，跳过计算。这是最准确的来源。

### 来源 2: YAML 配置文件

用户可通过启动参数提供自定义价格文件路径。格式贴近 genai-prices 的 provider YAML schema，降低迁移成本：

```yaml
# custom_prices.yaml
# 按 provider（供应商）+ model 组织
- provider: deepseek
  models:
    - id: deepseek-chat
      prices:
        input_mtok: 0.14
        output_mtok: 0.28

- provider: openai
  models:
    - id: gpt-4o
      prices:
        input_mtok: 2.5
        output_mtok: 10
        cache_read_mtok: 1.25

    - id: my-fine-tuned-model
      prices:
        input_mtok: 5
        output_mtok: 15

- provider: anthropic
  models:
    - id: claude-sonnet-4-20250514
      prices:
        input_mtok: 3
        output_mtok: 15
        cache_read_mtok: 0.3
        cache_write_mtok: 3.75
```

**规格说明：**
- `provider`: 供应商标识符，对应 Provider 的 `provider` 属性（如 `"deepseek"`, `"openai"`, `"anthropic"`）
- `models[].id`: 模型标识符，精确匹配
- `models[].prices`: 价格字段与 genai-prices 一致（`input_mtok`, `output_mtok`, `cache_read_mtok`, `cache_write_mtok`），单位为 USD / 百万 token
- **不支持** genai-prices 中的高级特性（tiered pricing、time-of-day constraint、conditional prices）。这是复杂度与简洁度的有意 trade-off -- 不同供应商的计费算法差异很大（阶梯定价、峰谷时段、cache 层级），yuullm 只覆盖最常用的平价模型，复杂场景建议使用来源 1（供应商自带）或直接依赖 genai-prices 库
- 匹配策略：`(provider, model_id)` 精确匹配。不做模糊匹配，避免意外

### 来源 3: genai-prices 回退

当来源 1 和 2 均无法确定价格时，调用 [genai-prices](https://github.com/pydantic/genai-prices) 库：

```python
from genai_prices import calc_price, Usage as GPUsage

price_data = calc_price(
    GPUsage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens),
    model_ref=usage.model,
    provider_id=usage.provider,
)
# price_data.total_price -> float | None
```

genai-prices 内部有丰富的模型匹配逻辑（regex、前缀、模糊匹配）和完整的供应商价格数据库（581+ OpenRouter 模型、63+ OpenAI 模型等）。若匹配失败（模型未收录、provider 未知），**cost 设为 `None`**，不抛异常。

**注意：** genai-prices 是尽力而为（best-effort）的社区维护数据，价格不保证 100% 准确。详见其 [README 中的 warning](https://github.com/pydantic/genai-prices#%EF%B8%8F-warning-these-prices-will-not-be-100-accurate)。

## Module Layout

```
src/yuullm/
    __init__.py          # public API re-exports
    py.typed
    types.py             # Message (tuple), Item, History, StreamItem, Usage, Cost, helper functions
    client.py            # Client 类 -- 面向用户的统一入口
    provider.py          # Provider 协议 -- 定义 api_type + provider 属性
    pricing.py           # PriceCalculator -- 价格计算引擎（三级来源）
    providers/
        __init__.py
        openai.py        # OpenAIChatCompletionProvider -- /v1/chat/completions
        anthropic.py     # AnthropicMessagesProvider -- /v1/messages
```

## Key Types (sketch)

```python
from typing import Literal, Required, TypedDict
import msgspec

# -- Content Item TypedDict (结构对齐 OpenAI API) --
class ToolCallItem(TypedDict):
    type: Literal["tool_call"]
    id: str
    name: str
    arguments: str

class ToolResultItem(TypedDict):
    type: Literal["tool_result"]
    tool_call_id: str
    content: str

class TextItem(TypedDict):
    type: Literal["text"]
    text: str

class ImageItem(TypedDict):
    type: Literal["image_url"]
    image_url: _ImageURL            # {url: str, detail?: "auto"|"low"|"high"}

class AudioItem(TypedDict):
    type: Literal["input_audio"]
    input_audio: _InputAudio        # {data: str, format: "wav"|"mp3"}

class FileItem(TypedDict):
    type: Literal["file"]
    file: _FileData                 # {file_data?: str, file_id?: str, filename?: str}

DictItem = ToolCallItem | ToolResultItem | TextItem | ImageItem | AudioItem | FileItem
Item = str | DictItem

# -- Messages: 轻量元组，不是类 --
Message = tuple[str, list[Item]]  # (role, items)
History = list[Message]

# 便利函数
def system(content: str) -> Message: ...
def user(*items: Item) -> Message: ...
def assistant(*items: Item) -> Message: ...
def tool(tool_call_id: str, content: str) -> Message: ...

# -- 输出流: msgspec struct --
class Reasoning(msgspec.Struct, frozen=True):
    item: Item              # 思考内容，可以是文本(str)或多模态(TypedDict)

class ToolCall(msgspec.Struct, frozen=True):
    id: str
    name: str
    arguments: str          # raw JSON string

class Response(msgspec.Struct, frozen=True):
    item: Item              # 回复内容，可以是文本(str)或多模态(TypedDict)

StreamItem = Reasoning | ToolCall | Response

# -- Usage & Cost --
class Usage(msgspec.Struct, frozen=True):
    provider: str           # 供应商名称 (e.g. "deepseek", "openai", "anthropic")
    model: str
    request_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int | None = None

class Cost(msgspec.Struct, frozen=True):
    """单次请求的费用明细。所有金额单位为 USD。"""
    input_cost: float
    output_cost: float
    total_cost: float
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    source: str = ""          # "provider" | "yaml" | "genai-prices"
```

## Provider Protocol

```python
from collections.abc import AsyncIterator
from typing import Any, Protocol

class Provider(Protocol):
    @property
    def api_type(self) -> str:
        """API 协议标识: "openai-chat-completion" | "openai-responses" | "anthropic-messages" """
        ...

    @property
    def provider(self) -> str:
        """供应商标识: "openai" | "deepseek" | "openrouter" | "anthropic" | ... """
        ...

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> tuple[AsyncIterator[StreamItem], dict]:
        """
        Returns:
            - async iterator of StreamItem
            - a mutable dict (store) that will be populated with:
              - "usage": Usage struct (after iterator exhausted)
              - "provider_cost": float | None (if provider returns cost directly)
        """
        ...
```

返回 `(iterator, store)` 的设计允许调用方在流结束后从 `store` 中获取 usage/cost 等信息，无需额外回调。

Provider 如果检测到供应商自带的费用信息（来源 1），应将其写入 `store["provider_cost"]`。Client 会优先使用此值。

## Client API

```python
class YLLMClient:
    def __init__(
        self,
        provider: Provider,
        default_model: str,
        tools: list[dict[str, Any]] | None = None,
        price_calculator: PriceCalculator | None = None,
    ): ...

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[AsyncIterator[StreamItem], dict]:
        """
        Delegates to provider.stream(), then enriches store with Cost.

        After the iterator is exhausted, store will contain:
            store["usage"] -> Usage
            store["cost"]  -> Cost | None
        """
        ...
```

## PriceCalculator

```python
class PriceCalculator:
    def __init__(
        self,
        yaml_path: str | Path | None = None,
        enable_genai_prices: bool = True,
    ): ...

    def calculate(self, usage: Usage, provider_cost: float | None = None) -> Cost | None:
        """
        按优先级计算 cost：
        1. provider_cost (来源 1: 供应商自带)
        2. yaml 配置 (来源 2: 精确匹配 provider + model)
        3. genai-prices (来源 3: 回退)

        全部失败则返回 None。
        """
        ...
```

## Example Usage

```python
import yuullm

# --- 创建客户端 (OpenAI 官方) ---
client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIChatCompletionProvider(api_key="sk-..."),
    default_model="gpt-4o",
    price_calculator=yuullm.PriceCalculator(
        yaml_path="./custom_prices.yaml",  # 可选
    ),
)

# --- 创建客户端 (DeepSeek, 使用 OpenAI-compatible API) ---
client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIChatCompletionProvider(
        api_key="sk-...",
        base_url="https://api.deepseek.com/v1",
        provider_name="deepseek",
    ),
    default_model="deepseek-chat",
)

# --- 创建客户端 (Anthropic) ---
client = yuullm.YLLMClient(
    provider=yuullm.providers.AnthropicMessagesProvider(api_key="sk-ant-..."),
    default_model="claude-sonnet-4-20250514",
)

# --- 构造消息（纯函数调用，无需类名） ---
messages = [
    yuullm.system("You are a helpful assistant."),
    yuullm.user("What is 2+2?"),
]

# --- 流式调用 ---
stream, store = await client.stream(messages)
async for stream_item in stream:
    match stream_item:
        case yuullm.Reasoning(item=i):
            if isinstance(i, str):
                print(f"[thinking] {i}", end="")
            else:
                print(f"[thinking] {i}", end="")
        case yuullm.Response(item=i):
            if isinstance(i, str):
                print(i, end="")
            else:
                print(f"[{i.get('type', 'content')}]", end="")
        case yuullm.ToolCall() as tc:
            print(f"[tool_call] {tc.name}({tc.arguments})")

# --- 流结束后获取 usage/cost ---
usage: yuullm.Usage = store["usage"]
cost: yuullm.Cost | None = store["cost"]

print(f"Provider: {usage.provider}")
print(f"Tokens: {usage.input_tokens} in / {usage.output_tokens} out")
if cost:
    print(f"Cost: ${cost.total_cost:.6f} (source: {cost.source})")
else:
    print("Cost: unavailable (model price not found)")
```

### 多模态示例

```python
messages = [
    yuullm.system("You are a vision assistant."),
    yuullm.user("What is in this image?", {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}),
]
```

### 与 yuutools 集成

```python
import yuutools as yt
import yuullm

# yuutools 定义工具
manager = yt.ToolManager([search_tool, calculator_tool])

# 直接传给 yuullm，无需任何转换
tools = manager.specs()  # list[dict]
stream, store = await client.stream(messages, tools=tools)
```

## Design Decisions

1. **无 session / 无状态** -- yuullm 不维护对话历史，history 由上层（yuuagents）管理。
2. **消息是元组，不是类** -- `Message = (role, items)` 避免引入 SystemMessage/UserMessage/AssistantMessage 等重量级抽象。用户不需要记忆和导入一堆类名。便利函数 `system()`, `user()`, `assistant()`, `tool()` 提供人体工学。
3. **工具是 dict，不是 ToolSpec** -- yuullm 不定义自己的工具规格类型。直接接受 `list[dict]`（json_schema 格式），与 yuutools 的输出无缝对接但不产生包依赖。Provider 内部负责格式转换。
4. **支持多模态** -- Item 是 `str | DictItem`，DictItem 是一组 TypedDict 的 union（ToolCallItem, ToolResultItem, TextItem, ImageItem, AudioItem, FileItem），结构对齐 OpenAI API 格式。TypedDict 在运行时仍是 dict，provider 的 `isinstance(it, dict)` 检查无需改动。
5. **msgspec structs 用于输出** -- StreamItem (Reasoning/ToolCall/Response) 和 Usage/Cost 使用 `msgspec.Struct`，因为这些是 yuullm 产出的类型安全数据。
6. **store dict 模式** -- 流式场景下 usage/cost 只有在流结束后才可用；通过传入可变 dict 收集，避免回调或 sentinel 值。
7. **Provider 可扩展** -- 新增 provider 只需实现 `stream()` 协议，无需注册表或插件系统。
8. **api_type 与 provider 分离** -- `api_type` 标识 API 协议（如 `"openai-chat-completion"`），`provider` 标识供应商（如 `"deepseek"`）。同一个 Provider 类可以服务多个供应商（通过 `provider_name` 参数），而不同的 API 协议使用不同的 Provider 类。
9. **三级价格来源** -- 供应商自带 > YAML 配置 > genai-prices 回退。优先使用最准确的来源，逐级降级。
10. **Cost 可为 None** -- 价格计算是尽力而为。找不到价格时 cost 为 None，不阻塞业务流程。
11. **YAML 只覆盖简单场景** -- 有意不支持 tiered pricing、时段定价等复杂特性，在复杂度与抽象简洁度之间取 trade-off。复杂定价建议直接使用供应商自带费用或 genai-prices 库。
