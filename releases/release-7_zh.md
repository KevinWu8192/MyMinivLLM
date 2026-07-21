# 摘要 — Commit ID 区间：[`7d1b0f1`](https://github.com/KevinWu8192/MinivLLM/commit/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf) → [`5608dc3`](https://github.com/KevinWu8192/MinivLLM/commit/5608dc33c3f59675cba7f0508f403f71bc0ad699)

**Tag：** [`release-7`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-7)

本次 Release 新增 Qwen3-32B 的端到端模型组装支持：提供独立且兼容 Hugging Face 的运行配置，将模型注册到现有张量并行 Runner，把其 Attention 层接入 Release 6 引入的 Large-scale Decode Kernel，并确保多 GPU 启动时只在 Tensor Parallel Worker 创建前下载一次 Checkpoint。

## 主要功能

### 1. 独立的 Qwen3-32B 运行配置

**涉及文件**

* `main_qwen32.py`

**功能**

* 新增独立的 Qwen3-32B 入口，模型架构配置为 `hidden_size = 5120`、`intermediate_size = 25600`、64 层、64 个 Query Head、8 个 KV Head、`head_dim = 128`、BF16 权重、40,960 Token 的位置容量，以及不共享的输入/输出 Embedding。运行配置默认使用八路张量并行，并显式启用 Large-scale Attention。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/main_qwen32.py#L13-L41))
* 该入口加载官方 `Qwen/Qwen3-32B` Tokenizer、应用其 Chat Template、初始化现有 `LLMEngine`，并通过其他已支持模型共用的调度、KV Cache、CUDA Graph、采样和 Checkpoint 加载链路执行生成。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/main_qwen32.py#L44-L64))

### 2. Qwen3-32B 模型注册与 Large-scale Attention 路由

**涉及文件**

* `src/myvllm/engine/model_runner.py`
* `src/myvllm/models/qwen3.py`

**功能**

* `ModelRunner` 现在同时识别 `Qwen3-0.6B` 和 `Qwen3-32B`，使用共享的 `Qwen3ForCausalLM` 实现构造两种 Checkpoint，并默认对 32B 模型启用 `use_large_scale_attention`。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/engine/model_runner.py#L48-L72))
* Attention Backend 选择从 `Qwen3ForCausalLM` 经过 `Qwen3Model` 传递到每一个 `Qwen3DecoderLayer`，因此全部 64 个 Transformer Layer 都能获得一致的 Backend 配置，同时无需修改现有 Layer 实现。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/models/qwen3.py#L190-L292))
* `Qwen3Attention` 在收到启用配置时选择 `attention_large_scale.Attention`。模型 API 边界上的该选项仍默认为 false，因此 Qwen3-0.6B 与现有调用方继续使用原 Attention 路径。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/models/qwen3.py#L30-L93))

最终模型组装路径如下：

```text
main_qwen32.py
    │ Qwen/Qwen3-32B 配置
    ▼
ModelRunner
    │ Qwen3-32B 注册 + TP Rank 本地模型构造
    ▼
Qwen3ForCausalLM → Qwen3Model → 64 × Qwen3DecoderLayer
    │ use_large_scale_attention = True
    ▼
attention_large_scale.Attention
    ├── Flash Attention Prefill
    └── Split-KV GQA Decode
```

### 3. Attention Backend 组装测试

**涉及文件**

* `tests/test_loader.py`

**功能**

* 新增聚焦模型组装的测试：使用 `use_large_scale_attention=True` 构造一个小型 Qwen3 模型，并验证每个 Decoder Layer 都包含 Large-scale Attention 实现。该测试无需分配 32B Checkpoint，即可检查完整的配置传递链路。([实现](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/tests/test_loader.py#L122-L140))

## 主要修复

### 1. 在创建 Tensor Parallel Worker 前只解析一次 Checkpoint

**涉及文件**

* `src/myvllm/engine/llm_engine.py`
* `src/myvllm/engine/model_runner.py`
* `src/myvllm/utils/loader.py`
* `tests/test_engine_boundaries.py`

**问题**

* `LLMEngine` 会先创建所有非零 TP Rank，再由 rank 0 构造自己的 `ModelRunner`，但创建进程前没有先把远程模型解析成共享的本地 Checkpoint 目录。因此每个进程都会独立进入模型初始化。([原始代码](https://github.com/KevinWu8192/MinivLLM/blob/c1d52dbf470b5595e39dbdb3bad196a80da0e900/src/myvllm/engine/llm_engine.py#L28-L67))
* 每个 `ModelRunner` 都把原始 Hugging Face 仓库 ID 直接传给 `load_weights_from_checkpoint()`。当 `world_size = 2` 时，rank 0 和 rank 1 会同时对同一组 17 个 Qwen3-32B Safetensors 分片调用 `snapshot_download()`，重复占用网络，并放大缓存、磁盘和认证失败。([原始代码](https://github.com/KevinWu8192/MinivLLM/blob/c1d52dbf470b5595e39dbdb3bad196a80da0e900/src/myvllm/engine/model_runner.py#L93-L100))

**示例**

当 `model_name_or_path = "Qwen/Qwen3-32B"`、`world_size = 2` 时，父进程先启动 rank 1，再初始化 rank 0。两个 Rank 都收到远程仓库 ID，因此都会开始下载从 `model-00001-of-00017.safetensors` 到 `model-00017-of-00017.safetensors` 的全部文件。控制台会出现两套下载进度；Xet CAS `401 Unauthorized`、磁盘容量不足或缓存写入中断可能分别导致两个 Rank 失败，并让分布式启动停留在不完整状态。

**修复**

* Checkpoint 路径解析被公开为 `resolve_checkpoint_path()`：若输入已经是本地目录则立即返回，否则只执行一次 Hugging Face Snapshot 下载。([修复](https://github.com/KevinWu8192/MinivLLM/blob/5608dc33c3f59675cba7f0508f403f71bc0ad699/src/myvllm/utils/loader.py#L108-L124))
* 父进程 `LLMEngine` 在创建任何 TP Process 之前调用幂等的 `resolve_checkpoint_once()`，并把解析出的目录写入 `config["checkpoint_path"]`。所有新 Rank 都会获得同一个本地路径；如果下载失败，也会在 NCCL Worker 启动前直接退出。([修复](https://github.com/KevinWu8192/MinivLLM/blob/5608dc33c3f59675cba7f0508f403f71bc0ad699/src/myvllm/engine/llm_engine.py#L29-L81))
* 每个 `ModelRunner` 继续使用原始模型 ID 判断架构，但在 `checkpoint_path` 存在时从该本地路径加载权重。Rank 本地的 Tensor Parallel 分片逻辑保持不变，仅集中处理远程 Snapshot 获取。([修复](https://github.com/KevinWu8192/MinivLLM/blob/5608dc33c3f59675cba7f0508f403f71bc0ad699/src/myvllm/engine/model_runner.py#L96-L102))
* 回归测试连续两次执行 Checkpoint 准备，验证 Resolver 只运行一次，并且两次调用都返回相同的本地目录。([修复](https://github.com/KevinWu8192/MinivLLM/blob/5608dc33c3f59675cba7f0508f403f71bc0ad699/tests/test_engine_boundaries.py#L104-L119))

## 验证

已提交测试覆盖了所有已构造 Qwen3 Decoder Layer 的 Large-scale Attention 选择，并验证重复执行 Checkpoint 准备时只进行一次路径解析。本次在 macOS ARM 环境中完成了语法编译、配置字段检查、Qwen3-32B Dispatch 检查和 `git diff --check`。由于 `vllm==0.15.0` 没有 macOS Wheel，且当前环境没有 NVIDIA CUDA Runtime 或多 GPU Checkpoint 容量，因此未在本地运行完整 Pytest 和 Qwen3-32B 端到端推理。该修复会避免 TP Rank 重复下载，但不会绕过 Hugging Face 认证：真实的 Hub 或 Xet `401 Unauthorized` 仍需要有效凭据或正确的 Hub 传输配置。所选 Split-KV Decode Kernel 的 CUDA 正确性与性能证据仍记录在 Release 6 中。
