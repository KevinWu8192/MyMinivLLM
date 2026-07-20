# 摘要 — Commit ID 区间：[`ee950af`](https://github.com/KevinWu8192/MyMinivLLM/commit/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d)

**Tag：** [`release-5`](https://github.com/KevinWu8192/MyMinivLLM/tree/release-5)

此 Release 对完整推理路径进行加固，包括多 GPU Worker 协同、Prefix Cache 记账、调度公平性、数值稳定性、CUDA Graph Decode 以及引擎 API 边界行为。

## 主要修复

### 1. 多 GPU Worker 协同

**涉及文件**

* `src/myvllm/engine/sequence.py`
* `src/myvllm/engine/model_runner.py`
* `src/myvllm/engine/llm_engine.py`

**问题**

* Worker 反序列化时遗漏 `Sequence.block_size`，导致 Rank 0 之外的进程调用 `num_blocks` 等方法时失败。([原始序列化](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/sequence.py#L94-L110))
* 所有引擎使用同一个分布式 Rendezvous 端口和共享内存名称，并发引擎之间会发生冲突。([原始 Worker 配置](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/model_runner.py#L20-L31))
* 写入共享内存前没有检查序列化 Payload 大小。([原始共享内存写入](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/model_runner.py#L124-L143))
* Worker 初始化和清理顺序可能将异常表现为 NCCL 卡死，或者重复执行 Cleanup。([原始 Worker 生命周期](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/model_runner.py#L145-L169)，[原始引擎清理](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/llm_engine.py#L58-L62))

**示例**

同一主机上启动两个引擎时，它们会尝试绑定相同的 Rendezvous 端口并创建同名共享内存。第二个引擎可能在初始化时失败，也可能连接到第一个引擎拥有的资源。另一方面，Rank 0 序列化 Sequence 时遗漏了 `block_size`；Worker 恢复后计算 `num_blocks` 会抛出 `AttributeError: 'Sequence' object has no attribute 'block_size'`。此时 Rank 0 可能正在等待 NCCL Collective，使真正的 Python 异常看起来像分布式卡死。

**修复**

* 将 `block_size` 与其他 Sequence 状态一起序列化和恢复。([Sequence 状态](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/sequence.py#L95-L122))
* Spawn 进程前根据可用 CUDA 设备验证 `world_size`，让操作系统选择空闲 TCP 端口，并生成基于 UUID 的共享内存名称。两个值都通过 Worker Config 传递，使同一引擎内所有 Rank 保持一致，不同引擎之间相互隔离。([引擎初始化](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/llm_engine.py#L29-L59))
* Rank 0 创建并拥有共享内存，其他 Rank 按名称连接。每个命令先序列化，再将字节长度与配置容量比较，避免写入被截断的 Payload。([共享内存](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/model_runner.py#L119-L169))
* 使 Worker 和引擎 Shutdown 具备幂等性，并在清理时 Join Worker 进程。([Worker 清理](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/model_runner.py#L174-L196)，[引擎清理](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/llm_engine.py#L92-L99))

### 2. Prefix Cache 复用正确性

**涉及文件**

* `src/myvllm/engine/block_manager.py`
* `src/myvllm/models/qwen3.py`

**问题**

* 一个 Prefix Hash 只能映射到一个物理 Block。多个有效副本会互相覆盖，释放其中一个还可能删除另一个副本的映射。([原始标量 Hash Map](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/block_manager.py#L24-L35)，[原始替换逻辑](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/block_manager.py#L46-L63))
* Prefix Cache 命中后，Qwen3 RoPE Position 会从未缓存后缀的零位置重新开始，而不是从缓存 Context 后继续。([原始位置计算](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/models/qwen3.py#L209-L229))

**示例**

假设物理 Block 3 和 9 包含相同 Token Prefix，因此拥有同一个 Hash。标量 `hash -> block_id` 只能记住 Block 9；如果 Block 9 被回收，删除该映射也会使仍然有效的 Block 3 无法访问。对于 RoPE，如果缓存了 96 个 Token、后缀有 32 个 Token，其 Position 应为 96–127；若从 0–31 重新开始，Q/K 会像 Prefix 不存在一样被错误旋转。

**修复**

* 将标量映射改为 `hash -> set[block_id]`。注册时添加一个物理副本；回收时只移除对应 Block ID，仅当 Set 为空时才删除 Hash Key。([Hash 索引](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/block_manager.py#L30-L57))
* 对每个逻辑 Prefix Block 遍历预期 Hash 的所有物理候选项，通过 Token ID 校验防止 Hash Collision，选择有效 Block 时不破坏其他副本。([Prefix 选择](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/block_manager.py#L91-L116))
* 对变长 Batch 中每个 Sequence 计算 `cached_len = full_kv_len - query_len`，为每个新 Query 重复该 Offset 并加上本地后缀位置，在保留扁平 Batch 表示的同时生成绝对 RoPE Position。([位置计算](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/models/qwen3.py#L6-L20)，[使用位置](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/models/qwen3.py#L230-L235))

### 3. Prefill Budget 与调度公平性

**涉及文件**

* `src/myvllm/engine/scheduler.py`

**问题**

* Prefill Token Budget 会计算无需重新计算的缓存 Token。([原始 Budget 记账](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/scheduler.py#L25-L41))
* 单个未缓存 Prompt 大于 Batch Budget 时，由于尚未实现 Chunked Prefill，请求可能永远阻塞。([原始 Prefill 阻塞路径](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/scheduler.py#L28-L41))
* 持续到来的 Prefill 请求可能使已经进入 Decode 的请求饥饿。([原始 Prefill 优先调度](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/scheduler.py#L25-L43))

**示例**

* **缓存 Token Budget：** Prompt 有 128 个 Token，Prefix Cache Hit 为 96 个 Token，且 `max_num_batched_tokens = 32` 时，实际只需计算 32 个 Token。旧调度器却收取全部 128 个 Token 的 Budget，错误拒绝了一个恰好能够执行的请求。
* **Decode 饥饿：** 假设 Decode 请求 D 位于 `running`，而每轮调度前都有一个新 Prefill 请求到达。旧调度器总是先尝试 `waiting`，只要成功调度任何 Prefill 就立即返回，因此即使 Decode Work 已经就绪，D 仍可能无限等待。

**修复**

* 分配前先从 BlockManager 获取预期 Prefix Hit，在准入阶段只收取 `len(sequence) - cached_tokens`；分配后使用 Sequence 记录的 `num_cached_tokens` 累加当前 Batch Budget，使预测值与实际记账一致。([Budget 计算](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/scheduler.py#L61-L87))
* 对不支持的超长 Prefill 给出可操作的错误，而不是无限期留在队列中。([超长 Prefill](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/scheduler.py#L69-L77))
* 记录上一次成功调度是否为 Prefill。当两个队列都有任务时，Prefill Pass 后跳过一次 Prefill，执行 Decode，再清除状态，使下一轮可以继续 Prefill。这样既限制 Decode 等待时间，也不会在没有 Running Request 时阻止 Prefill。([交替状态](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/scheduler.py#L29-L31)，[状态转换](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/scheduler.py#L61-L114))
* 验证 Sampler 为每个已调度 Sequence 返回且仅返回一个 Token。([输出验证](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/scheduler.py#L125-L132))

### 4. 数值稳定性与模型配置

**涉及文件**

* `src/myvllm/layers/layernorm.py`
* `src/myvllm/layers/sampler.py`
* `src/myvllm/layers/linear.py`
* `src/myvllm/models/qwen3.py`
* `src/myvllm/models/llama.py`

**问题**

* RMSNorm 忽略模型配置的 Epsilon，并直接使用 BF16/FP16 计算方差。([原始 RMSNorm 计算](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/layers/layernorm.py#L16-L24)，[原始 Qwen3 Norm 构造](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/models/qwen3.py#L47-L75))
* Sampling Softmax 直接根据低精度 Logits 计算。([原始 Sampler](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/layers/sampler.py#L13-L19))
* 无效的 Attention Head 整除关系可能静默产生错误的 Tensor Parallel Shape。([原始 TP Head 切分](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/layers/linear.py#L151-L168)，[原始 Qwen3 Head 切分](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/models/qwen3.py#L25-L36))

**修复**

* 将 `rms_norm_epsilon` 传递给 Qwen3 和 Llama 的 Q/K Norm、Decoder Norm 以及 Final Norm。([Qwen3 Norm](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/models/qwen3.py#L71-L75)，[Llama Norm](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/models/llama.py#L156-L176))
* 使用 FP32 计算 RMSNorm 方差和缩放，再转换回输入数据类型。([RMSNorm](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/layers/layernorm.py#L16-L24))
* 使用 FP32 计算 Temperature Scaling 和 Softmax 概率。([Sampler](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/layers/sampler.py#L13-L21))
* 验证 Attention Head 和 KV Head 数量可以被 TP Size 整除。([TP 验证](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/layers/linear.py#L159-L169))
* 支持显式配置 `bfloat16`、`float16` 和 `float32` 模型数据类型，并为不支持的值给出清晰错误。([数据类型选择](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/model_runner.py#L34-L46))

### 5. CUDA Graph Decode

**涉及文件**

* `src/myvllm/engine/model_runner.py`

**问题**

* Graph Capture 使用错误的配置键；使用 `vocab_size` 设置 Hidden State Buffer；在 Capture 前访问 Graph Pool；并且 Static Buffer 数据类型不一致。([原始 CUDA Graph 配置](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/model_runner.py#L405-L444))
* 不支持的 Batch Size 可能被 Capture 或 Replay。([原始 Batch Size 选择](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/engine/model_runner.py#L422-L425))

**示例**

Decoder Layer 返回的 Hidden State Shape 是 `[batch_size, hidden_size]`，旧 Static Output Buffer 却按 `[batch_size, vocab_size]` 分配。`vocab_size` 通常远大于 `hidden_size`，Graph Capture 因此记录了语义不兼容的 Tensor；Replay 时可能在 `copy_` 阶段失败，或把错误 Shape 传给 LM Head。在成功 Capture 前访问 `graph.pool()` 又增加了另一条失败路径。

**修复**

* 按运行时 Dtype 和 Shape 分配所有 Static Input，以配置的模型 Dtype 将 Hidden State 分配为 `[max_batch_size, hidden_size]`，并使用规范的 `max_num_sequences` 上限。Replay 时只切分到实际 Batch，同时保持 Capture 地址不变。([Static Buffer](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/model_runner.py#L443-L462))
* Capture Size 列表只包含 Static Buffer 支持的 Batch Size；先在 Capture Stream 上 Warmup 模型，再 Capture Graph，成功后才保存 Graph 和 Pool 供后续 Capture/Replay 使用。([Capture 循环](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/model_runner.py#L464-L489))

### 6. 引擎 API 边界行为

**修复**

* 拒绝空 Prompt，以及无效的 `temperature`、`max_tokens` 或上下文限制。([Prompt 验证](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/llm_engine.py#L126-L131)，[采样验证](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/sampling_parameters.py#L11-L19))
* 从 Tokenizer 读取 EOS，并在最终 Decode 时移除特殊 Token。([Tokenizer 行为](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/llm_engine.py#L67-L73)，[Detokenization](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/llm_engine.py#L152-L159))
* 即使没有已调度 Batch，`LLMEngine.step()` 也返回一致的三个元素。([Step 结果](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/llm_engine.py#L105-L123))
* 从 `Sequence` 中移除可变的默认 `SamplingParams` 实例。([Sequence 构造](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/src/myvllm/engine/sequence.py#L14-L22))

## 验证

此 Release 新增边界测试，覆盖 Sequence 序列化、重复 Prefix Hash、缓存 Context 的 RoPE Position、无效采样参数、空调度结果、空 Prompt、幂等 Shutdown 和特殊 Token 移除。调度器测试还覆盖感知 Prefix 的 Budget 以及 Prefill/Decode 交替调度。([引擎边界测试](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/tests/test_engine_boundaries.py#L1-L101)，[调度器测试](https://github.com/KevinWu8192/MyMinivLLM/blob/ee950af0b35f5ac87fa6a495238cf6ebb57aa99d/tests/test_scheduler.py#L251-L300))

> 多 GPU 代码路径已经修复，但最终运行时验证仍需要至少配备两张 CUDA GPU 的环境。
