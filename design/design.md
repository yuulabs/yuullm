# yuullm Design

## Overview

yuullm 提供统一的流式 LLM 接口抽象层。它有两个核心职责：

1. **流标准化** -- 屏蔽不同 provider 在思考格式（`reasoning` / `reasoning_content` / `reasoning_details.summary`）和工具调用协议上的差异，输出标准化的 `AsyncIterator[Reasoning | ToolCall | Response]` 流。
2. **Usage + Cost 收集** -- 在流结束后，通过 store 提供结构化的 `Usage`（来自 API 原始数据）和 `Cost`（yuullm 自行计算）。

yuullm **不负责埋点**（无 session 概念）。它只关注：把请求发出去，把流标准化地收回来，再把 usage/cost 算清楚。

## Provider 抽象：隔离供应商 API 细节

不同 LLM 供应商的 API 存在显著差异，yuullm 的核心价值之一就是将这些差异隔离在 Provider 内部，对上层暴露统一接口。

### 需要隔离的差异

| 差异维度 | OpenAI | Anthropic | 说明 |
|---|---|---|---|
| **响应结构** | `response.choices[0].delta` | `event.delta` (SSE) | 流式响应的数据路径完全不同 |
| **思考格式** | `reasoning_content` / `reasoning.summary` | `content[type=thinking].thinking` | 各家 CoT 表示方式不统一 |
| **工具调用** | `tool_calls` 数组 (含 index) | `content[type=tool_use]` | 工具调用的编码方式不同 |
| **Usage 字段** | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` | 字段名和结构不同 |
| **Cache 计量** | `cached_tokens` (嵌套在 details 中) | `cache_creation_input_tokens` / `cache_read_input_tokens` (顶层) | cache 指标位置和语义不同 |

### 隔离策略

每个 Provider 实现内部处理所有供应商特有逻辑，对外只暴露统一的 `stream()` 协议。具体来说：

- **流解析** -- Provider 内部将供应商私有的 SSE/streaming 格式映射为标准的 `StreamItem`（Reasoning / ToolCall / Response）。
- **Usage 提取** -- Provider 在流结束后从供应商响应中提取 usage 信息，统一映射到 `Usage` struct。不同 provider 的字段名差异（如 `prompt_tokens` vs `input_tokens`）在 Provider 内部消化。
- **上层无感** -- Client 和使用者不需要知道底层是 OpenAI 还是 Anthropic，拿到的都是同一套类型。

## Key Concepts

### StreamItem

流中的每个元素是以下三种之一：

- **Reasoning** -- 模型的思考过程片段（chain-of-thought / extended thinking）
- **ToolCall** -- 模型发起的工具调用请求（包含 id, name, arguments）
- **Response** -- 模型的最终文本回复片段

### Provider

对接一个具体的 LLM 服务（OpenAI, Anthropic, Google, etc.）。每个 Provider 实现统一的 `stream()` 协议，内部负责将供应商私有格式映射到 `StreamItem`，并提取标准化的 `Usage`。

### Client

面向用户的入口。持有一个 Provider 和一个可选的 `PriceCalculator`，暴露简洁的 `stream()` / `complete()` 方法，并在调用结束后通过 store 返回 `Usage` 和 `Cost`。

### Store

store 是一个 mutable dict，在流结束后包含：
- `store["usage"]` -- `Usage` struct（来自 API 的原始 token 计数）
- `store["cost"]` -- `Cost` struct（yuullm 计算的费用），若无法计算则为 `None`

## Store 中存什么

### Usage（来自 API）

Usage 由 Provider 在流结束后从 API 响应中提取。不同供应商返回的原始字段不同（见上表），Provider 负责将它们映射到统一的 `Usage` struct。

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
# 按 provider + model 组织
- provider: openai
  models:
    - id: gpt-4o
      prices:
        input_mtok: 2.5        # USD per million input tokens
        output_mtok: 10         # USD per million output tokens
        cache_read_mtok: 1.25   # USD per million cached input tokens (可选)

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
- `provider`: 供应商标识符，对应 Provider 的 name/id
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
    types.py             # StreamItem, Reasoning, ToolCall, Response, Usage, Cost, Message 等
    client.py            # Client 类 -- 面向用户的统一入口
    provider.py          # Provider 抽象基类
    pricing.py           # PriceCalculator -- 价格计算引擎（三级来源）
    providers/
        __init__.py
        openai.py        # OpenAI / OpenAI-compatible 实现
        anthropic.py     # Anthropic 实现
```

## Key Types (sketch)

```python
import msgspec

class Reasoning(msgspec.Struct, frozen=True):
    text: str

class ToolCall(msgspec.Struct, frozen=True):
    id: str
    name: str
    arguments: str          # raw JSON string

class Response(msgspec.Struct, frozen=True):
    text: str

StreamItem = Reasoning | ToolCall | Response

class Usage(msgspec.Struct, frozen=True):
    provider: str
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
from typing import Protocol

class Provider(Protocol):
    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
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
        price_calculator: PriceCalculator | None = None,
    ): ...

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
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

client = yuullm.YLLMClient(
    provider=yuullm.providers.OpenAIProvider(api_key="sk-..."),
    default_model="gpt-4o",
    price_calculator=yuullm.PriceCalculator(
        yaml_path="./custom_prices.yaml",  # 可选
    ),
)

messages = [
    yuullm.SystemMessage(content="You are a helpful assistant."),
    yuullm.UserMessage(content="What is 2+2?"),
]

stream, store = await client.stream(messages)
async for item in stream:
    match item:
        case yuullm.Reasoning(text=t):
            print(f"[thinking] {t}", end="")
        case yuullm.Response(text=t):
            print(t, end="")
        case yuullm.ToolCall() as tc:
            print(f"[tool_call] {tc.name}({tc.arguments})")

# After stream ends, usage and cost are available
usage: yuullm.Usage = store["usage"]
cost: yuullm.Cost | None = store["cost"]

print(f"Tokens: {usage.input_tokens} in / {usage.output_tokens} out")
if cost:
    print(f"Cost: ${cost.total_cost:.6f} (source: {cost.source})")
else:
    print("Cost: unavailable (model price not found)")
```

## Design Decisions

1. **无 session / 无状态** -- yuullm 不维护对话历史，history 由上层（yuuagents）管理。
2. **msgspec structs** -- 所有数据类型使用 `msgspec.Struct`，零拷贝序列化，与项目其他部分保持一致。
3. **store dict 模式** -- 流式场景下 usage/cost 只有在流结束后才可用；通过传入可变 dict 收集，避免回调或 sentinel 值。
4. **Provider 可扩展** -- 新增 provider 只需实现 `stream()` 协议，无需注册表或插件系统。
5. **Provider 负责隔离 API 细节** -- 不同供应商的响应结构、思考格式、工具调用协议、usage 字段等差异全部在 Provider 内部消化，上层看到的永远是统一的 StreamItem + Usage。
6. **三级价格来源** -- 供应商自带 > YAML 配置 > genai-prices 回退。优先使用最准确的来源，逐级降级。
7. **Cost 可为 None** -- 价格计算是尽力而为。找不到价格时 cost 为 None，不阻塞业务流程。
8. **YAML 只覆盖简单场景** -- 有意不支持 tiered pricing、时段定价等复杂特性，在复杂度与抽象简洁度之间取 trade-off。复杂定价建议直接使用供应商自带费用或 genai-prices 库。
