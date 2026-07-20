# 摘要 — Commit ID 区间：[`798b455`](https://github.com/KevinWu8192/MyMinivLLM/commit/798b4552a1401f1eebbfff865816a95169aada5f)

**Tag：** [`release-2`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-2)

此 Release 防止请求超过 KV Cache 或上下文容量，并修复 Decode 阶段判断何时需要新 KV Block 的边界条件。

## 主要修复

### 1. 引擎与请求级上下文限制

**涉及文件**

* `src/myvllm/engine/llm_engine.py`
* `src/myvllm/engine/scheduler.py`
* `tests/test_scheduler.py`

**问题**

* 引擎允许 `max_model_length` 超过物理 KV Cache 容量。([原始调度器初始化](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/scheduler.py#L6-L15))
* 请求可以设置高于引擎上限的上下文长度。([原始请求准入](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/scheduler.py#L18-L23))
* Prompt 已经占满上下文时仍能进入队列，没有为生成 Token 留出空间。([原始请求准入](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/scheduler.py#L18-L23))
* 超长请求可能永远无法被调度，而不是提前返回清晰错误。([原始 Prefill 调度](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/scheduler.py#L25-L41))

**示例**

如果引擎拥有 8 个 KV Block，且 `block_size = 16`，最多只能保存 128 个 Token。此时配置 `max_model_length = 256`，请求虽然能通过 API 校验，但执行时永远无法保存完整 Context。类似地，即使引擎上限正确设为 128，接受一个 128 Token 的 Prompt 也没有空间保存第一个生成 Token。这些请求会在内存压力下卡住或反复抢占，而不是在准入阶段立即失败。

**修复**

* 将引擎的 `max_model_length` 传入调度器。([引擎配置](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/llm_engine.py#L47-L56))
* 构造引擎时，将 Block 容量换算为 Token 数（`max_cached_blocks * block_size`），无法被物理 KV Cache 容纳的引擎 Context 上限会立即被拒绝。([容量验证](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/scheduler.py#L6-L26))
* 请求准入时只解析一次有效上限：未设置则继承引擎值；允许更小的正数请求上限；非正数或超过引擎的值会在进入 `waiting` 前被拒绝。([请求验证](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/scheduler.py#L34-L58))
* 强制满足 `num_prompt_tokens < request_max_length`，把解析后的上限写入 Sequence，并在每次采样 Token 后检查总长度，使生成在边界处结束并立即释放 Block。([Prompt 验证](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/scheduler.py#L48-L58)，[停止条件](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/scheduler.py#L113-L126))

### 2. Decode Block 分配边界

**问题**

`postprocess()` 会先将新采样的 Token 追加到 Sequence，然后才调用 `can_append()`。旧逻辑检查 `num_tokens % block_size == 0`，因此检查的是一个 Block 的末尾，而不是下一个 Block 的首个 Token，可能在没有所需 KV Cache 空间的情况下继续 Decode。([原始检查](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/block_manager.py#L157-L161))

**示例**

假设 `block_size = 4`，Sequence 当前有 4 个 Token，存放在 Block 0 中。`postprocess()` 追加 Token 5 后，`num_tokens = 5`，这个 Token 属于新的 Block 1。旧条件检查 `5 % 4 == 0`，结果为 false，因此认为不需要新 Block。下一轮 Decode 随后会为 Block 1 构造 Slot Mapping，但这个 Block 从未被分配。由于 Token 已经追加，正确边界应为 `5 % 4 == 1`。

**修复**

让 `can_append()` 与 `append()` 使用相同的“Token 已追加”状态约定：仅在 `num_tokens % block_size == 1` 时要求空闲 Block；其他情况下，新 Token 仍位于已经分配的最后一个 Block 内。这样容量准入与后续 Block Table 更新保持一致。([边界修复](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/engine/block_manager.py#L153-L163))

## 验证

调度器测试覆盖了继承和请求级上下文上限、无效上限、没有生成空间的 Prompt、KV Cache 容量溢出，以及在上下文边界准确停止。([测试](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/tests/test_scheduler.py#L190-L250))
