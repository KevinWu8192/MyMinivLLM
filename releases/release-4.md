# Summary — Commit ID Range: [`4d1760d`](https://github.com/KevinWu8192/MyMinivLLM/commit/4d1760ded441988663c59f5bd9479a80de234731)

**Tag:** [`release-4`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-4)

This release replaces the Qwen3 checkpoint loader with a tensor-parallel-aware, streaming loader that correctly handles packed parameters and validates checkpoint completeness.

## Major Fixes

### 1. Correct Packed-Parameter Mapping

**Files**

* `src/myvllm/models/qwen3.py`
* `src/myvllm/utils/loader.py`

**Bugs**

* Hugging Face stores `q_proj`, `k_proj`, and `v_proj` separately, while MiniVLLM stores them in one `qkv_projection` parameter; the old loader concatenated the full source tensors before copying. ([Original QKV merge](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L75-L101))
* Hugging Face stores `gate_proj` and `up_proj` separately, while MiniVLLM stores them in one `gate_up` parameter; the old loader likewise concatenated both full source tensors. ([Original Gate/Up merge](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L102-L124))
* The old mapping used incorrect target names and assembled complete tensors before loading, bypassing rank-local sharding. ([Original loader](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L16-L150))

**Example**

For a small Qwen3 layer with hidden size 8, four Q Heads, two KV Heads, Head size 2, and TP size 2:

```text
Hugging Face q_proj: [8, 8] -> each rank needs [4, 8]
Hugging Face k_proj: [4, 8] -> each rank needs [2, 8]
Hugging Face v_proj: [4, 8] -> each rank needs [2, 8]
rank-local qkv_projection: [4 + 2 + 2, 8] = [8, 8]
```

The old loader concatenated full Q/K/V tensors to `[16, 8]` and tried to copy that into each Rank's `[8, 8]` parameter. Besides the Shape mismatch, blindly slicing the concatenated tensor would mix Q, K, and V region boundaries.

**Fixes**

* Define an explicit target-to-source mapping: `qkv_projection` receives `q_proj`, `k_proj`, and `v_proj` with shard IDs `q/k/v`; `gate_up` receives `gate_proj` and `up_proj` with packed-region IDs `0/1`. ([Qwen3 mapping](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/models/qwen3.py#L290-L301))
* `_map_weight_name()` replaces only the matching module path component, leaving the layer prefix and `.weight` suffix intact, then returns both the target parameter name and the source-specific shard ID. This avoids ambiguous substring mapping across layers. ([Name mapping](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L32-L48))

### 2. Tensor-Parallel-Aware Dispatch

**Bug**

The old loader copied full checkpoint tensors directly into rank-local parameters. This is incorrect for row-parallel, column-parallel, vocabulary-parallel, and packed QKV/MLP weights. ([Original direct copies](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L75-L165))

**Fixes**

* Build a lookup from `model.named_parameters(remove_duplicate=False)` so tied or aliased parameter names remain addressable, then dispatch every source Tensor through the matched target parameter's custom `weight_loader`. Replicated parameters fall back to strict Shape-checked copying. ([Loader dispatch](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L17-L31), [loading loop](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L50-L84))
* For ordinary TP parameters, the parameter loader selects the current Rank along its Row, Column, or Vocabulary sharding dimension. For Packed parameters, it first selects the source-specific Q/K/V or Gate/Up destination region and then writes only the current Rank's slice into that region.
* Wrap failures with both source and target parameter names so shape or sharding errors are traceable. ([Error context](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L67-L77))

### 3. Streaming and Strict Checkpoint Validation

**Bugs**

* The old loader materialized every Safetensors tensor in a Python dictionary before loading, increasing host memory usage. ([Original checkpoint collection](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L54-L67))
* Unexpected, missing, or duplicate tensors could silently produce a partially initialized model. ([Original permissive loading](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L68-L170))

**Example**

If two Safetensors shards both contain `model.layers.0.self_attn.q_proj.weight`, accepting the second value makes the loaded model depend on file ordering. If `k_proj.weight` is missing, the K region of `qkv_projection` remains initialized with unrelated memory while the loader may still appear successful. Both cases must fail deterministically.

**Fixes**

* Resolve a local directory or Hugging Face snapshot, sort Safetensors filenames for deterministic traversal, keep each file open only while iterating it, and yield one `(name, tensor)` pair at a time instead of accumulating the full checkpoint in RAM. ([Streaming iterator](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L127-L146))
* Track loaded parameter object IDs rather than names, so tied parameters count as initialized. After iteration, compare loaded IDs against all model parameters and collect unexpected source names before raising one actionable completeness error. ([Completeness checks](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L79-L110))
* Reject duplicate tensor names across Safetensors shards and report the number of loaded parameters. ([Checkpoint entry point](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L148-L160))

## Validation

New tests simulate two tensor-parallel ranks and verify vocabulary, Q/K/V, output projection, Gate/Up, Down projection, Norm, and LM Head loading. They also cover unexpected, missing, duplicate, and shape-mismatched tensors. ([Tests](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/tests/test_loader.py#L1-L118))
