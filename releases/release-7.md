# Summary — Commit ID Range: [`7d1b0f1`](https://github.com/KevinWu8192/MinivLLM/commit/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf) → [`5608dc3`](https://github.com/KevinWu8192/MinivLLM/commit/5608dc33c3f59675cba7f0508f403f71bc0ad699)

**Tag:** [`release-7`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-7)

This release adds end-to-end model assembly support for Qwen3-32B. It provides a dedicated Hugging Face-compatible runtime configuration, registers the model with the existing tensor-parallel runner, routes its Attention layers to the large-scale Decode Kernel introduced in Release 6, and ensures that a multi-GPU launch downloads the checkpoint only once before Tensor Parallel workers are spawned.

## Major Features

### 1. Dedicated Qwen3-32B Runtime Configuration

**Files**

* `main_qwen32.py`

**Features**

* A standalone Qwen3-32B entry point defines the model architecture with `hidden_size = 5120`, `intermediate_size = 25600`, 64 layers, 64 Query Heads, 8 KV Heads, `head_dim = 128`, BF16 weights, a 40,960-token position capacity, and untied input/output embeddings. The runtime defaults to eight-way Tensor Parallelism and explicitly enables large-scale Attention. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/main_qwen32.py#L13-L41))
* The entry point loads the official `Qwen/Qwen3-32B` tokenizer, applies its chat template, initializes the existing `LLMEngine`, and runs generation through the same scheduling, KV Cache, CUDA Graph, sampling, and checkpoint-loading pipeline used by other supported models. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/main_qwen32.py#L44-L64))

### 2. Qwen3-32B Model Registration and Large-Scale Attention Routing

**Files**

* `src/myvllm/engine/model_runner.py`
* `src/myvllm/models/qwen3.py`

**Features**

* `ModelRunner` now recognizes both `Qwen3-0.6B` and `Qwen3-32B`. It builds either checkpoint with the shared `Qwen3ForCausalLM` implementation and defaults `use_large_scale_attention` to true for the 32B model. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/engine/model_runner.py#L48-L72))
* The Attention-backend choice is propagated from `Qwen3ForCausalLM` through `Qwen3Model` and every `Qwen3DecoderLayer`, so all 64 transformer layers receive the same backend selection without changing the existing layer implementations. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/models/qwen3.py#L190-L292))
* `Qwen3Attention` selects `attention_large_scale.Attention` when requested. The option remains false by default at the model API boundary, preserving the original Attention path for Qwen3-0.6B and existing callers. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/src/myvllm/models/qwen3.py#L30-L93))

The resulting assembly path is:

```text
main_qwen32.py
    │ Qwen/Qwen3-32B configuration
    ▼
ModelRunner
    │ Qwen3-32B registration + TP-local model construction
    ▼
Qwen3ForCausalLM → Qwen3Model → 64 × Qwen3DecoderLayer
    │ use_large_scale_attention = True
    ▼
attention_large_scale.Attention
    ├── Flash Attention Prefill
    └── Split-KV GQA Decode
```

### 3. Attention Backend Assembly Coverage

**Files**

* `tests/test_loader.py`

**Features**

* A focused model-assembly test constructs a small Qwen3 model with `use_large_scale_attention=True` and verifies that every decoder layer contains the large-scale Attention implementation. This checks the complete configuration-propagation chain without allocating the 32B checkpoint. ([Implementation](https://github.com/KevinWu8192/MinivLLM/blob/7d1b0f1cb8c21f31e92241cfe7457fba769a2ddf/tests/test_loader.py#L122-L140))

## Major Fixes

### 1. Resolve the Checkpoint Once Before Spawning Tensor-Parallel Workers

**Files**

* `src/myvllm/engine/llm_engine.py`
* `src/myvllm/engine/model_runner.py`
* `src/myvllm/utils/loader.py`
* `tests/test_engine_boundaries.py`

**Bugs**

* `LLMEngine` spawned every nonzero TP Rank before rank 0 constructed its own `ModelRunner`, without first resolving the remote model into a shared local checkpoint directory. Each process therefore entered model initialization independently. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/c1d52dbf470b5595e39dbdb3bad196a80da0e900/src/myvllm/engine/llm_engine.py#L28-L67))
* Every `ModelRunner` passed the original Hugging Face repository ID directly to `load_weights_from_checkpoint()`. With `world_size = 2`, rank 0 and rank 1 consequently called `snapshot_download()` for the same 17 Qwen3-32B Safetensors shards at the same time, duplicating network work and amplifying cache, disk, and authentication failures. ([Original code](https://github.com/KevinWu8192/MinivLLM/blob/c1d52dbf470b5595e39dbdb3bad196a80da0e900/src/myvllm/engine/model_runner.py#L93-L100))

**Example**

For `model_name_or_path = "Qwen/Qwen3-32B"` and `world_size = 2`, the parent starts rank 1, then initializes rank 0. Both ranks receive the remote repository ID, so both begin fetching `model-00001-of-00017.safetensors` through `model-00017-of-00017.safetensors`. The console shows two download-progress sets; an Xet CAS `401 Unauthorized`, disk-capacity failure, or interrupted cache write can then fail both ranks independently and leave distributed startup in a partial state.

**Fixes**

* Checkpoint path resolution is exposed as `resolve_checkpoint_path()`, which returns an existing local directory immediately or performs the Hugging Face snapshot download once. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/5608dc33c3f59675cba7f0508f403f71bc0ad699/src/myvllm/utils/loader.py#L108-L124))
* The parent `LLMEngine` calls the idempotent `resolve_checkpoint_once()` before creating any TP process and stores the resolved directory in `config["checkpoint_path"]`. Every spawned Rank receives that same local path, and a download failure now occurs before NCCL workers are launched. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/5608dc33c3f59675cba7f0508f403f71bc0ad699/src/myvllm/engine/llm_engine.py#L29-L81))
* Each `ModelRunner` keeps the original model ID for architecture dispatch but loads weights from `checkpoint_path` when it is available. Rank-local Tensor Parallel sharding is unchanged; only remote snapshot acquisition is centralized. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/5608dc33c3f59675cba7f0508f403f71bc0ad699/src/myvllm/engine/model_runner.py#L96-L102))
* A regression test calls checkpoint preparation twice and verifies that the resolver runs exactly once and that both calls return the same local directory. ([Fix](https://github.com/KevinWu8192/MinivLLM/blob/5608dc33c3f59675cba7f0508f403f71bc0ad699/tests/test_engine_boundaries.py#L104-L119))

## Validation

The committed tests cover large-scale Attention selection across every constructed Qwen3 decoder layer and verify that repeated checkpoint preparation performs only one path resolution. In the current macOS ARM environment, syntax compilation, configuration-field checks, Qwen3-32B dispatch inspection, and `git diff --check` passed. The full Pytest suite and end-to-end Qwen3-32B execution were not run locally because `vllm==0.15.0` has no macOS wheel and this environment has no NVIDIA CUDA runtime or multi-GPU checkpoint capacity. This fix prevents duplicate TP downloads but does not bypass Hugging Face authentication: a genuine Hub or Xet `401 Unauthorized` still needs valid credentials or the appropriate Hub transport configuration. CUDA correctness and performance evidence for the selected Split-KV Decode Kernel remains documented in Release 6.
