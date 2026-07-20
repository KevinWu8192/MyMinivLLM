# 摘要 — Commit ID 区间：[`4d1760d`](https://github.com/KevinWu8192/MyMinivLLM/commit/4d1760ded441988663c59f5bd9479a80de234731)

**Tag：** [`release-4`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-4)

此 Release 使用支持张量并行的流式加载器替换原 Qwen3 Checkpoint Loader，正确处理 Packed Parameter，并严格验证 Checkpoint 完整性。

## 主要修复

### 1. 正确映射 Packed Parameter

**涉及文件**

* `src/myvllm/models/qwen3.py`
* `src/myvllm/utils/loader.py`

**问题**

* Hugging Face 分别存储 `q_proj`、`k_proj` 和 `v_proj`，MiniVLLM 则将它们存储在一个 `qkv_projection` 参数中；旧加载器会先拼接完整源 Tensor 再复制。([原始 QKV 合并](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L75-L101))
* Hugging Face 分别存储 `gate_proj` 和 `up_proj`，MiniVLLM 则将它们存储在一个 `gate_up` 参数中；旧加载器同样会拼接两个完整源 Tensor。([原始 Gate/Up 合并](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L102-L124))
* 旧映射使用了错误的目标名称，并在加载前拼接完整 Tensor，绕过了当前 Rank 的局部分片逻辑。([原始加载器](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L16-L150))

**示例**

对于 Hidden Size 为 8、四个 Q Head、两个 KV Head、Head Size 为 2、TP Size 为 2 的小型 Qwen3 Layer：

```text
Hugging Face q_proj: [8, 8] -> 每个 Rank 需要 [4, 8]
Hugging Face k_proj: [4, 8] -> 每个 Rank 需要 [2, 8]
Hugging Face v_proj: [4, 8] -> 每个 Rank 需要 [2, 8]
Rank-local qkv_projection: [4 + 2 + 2, 8] = [8, 8]
```

旧加载器先将完整 Q/K/V 拼接成 `[16, 8]`，再尝试复制到每个 Rank 的 `[8, 8]` 参数中。除了 Shape 不匹配，若直接切分拼接后的 Tensor，还会混淆 Q、K、V 各自的区域边界。

**修复**

* 定义显式的目标到源映射：`qkv_projection` 接收带 `q/k/v` Shard ID 的 `q_proj`、`k_proj`、`v_proj`；`gate_up` 接收 Packed Region ID 为 `0/1` 的 `gate_proj` 和 `up_proj`。([Qwen3 映射](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/models/qwen3.py#L290-L301))
* `_map_weight_name()` 只替换匹配的 Module Path 片段，保留 Layer Prefix 和 `.weight` Suffix，然后同时返回目标参数名与源专用 Shard ID，避免跨 Layer 的模糊字符串映射。([名称映射](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L32-L48))

### 2. 感知张量并行的加载分派

**问题**

旧加载器会将完整 Checkpoint Tensor 直接复制到当前 Rank 的局部参数中。对于 Row-Parallel、Column-Parallel、Vocabulary-Parallel 以及 Packed QKV/MLP 权重，这是错误的。([原始直接复制](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L75-L165))

**修复**

* 使用 `model.named_parameters(remove_duplicate=False)` 构造查找表，使 Tied 或 Alias 参数名称仍然可寻址，再将每个源 Tensor 分派给匹配参数的自定义 `weight_loader`；Replicated Parameter 使用严格 Shape 检查的默认复制逻辑。([加载器分派](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L17-L31)，[加载循环](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L50-L84))
* 普通 TP 参数由参数加载器沿 Row、Column 或 Vocabulary Sharding Dimension 选择当前 Rank 切片；Packed Parameter 则先定位 Q/K/V 或 Gate/Up 的目标区域，再只将当前 Rank 的切片写入该区域。
* 错误信息同时包含源参数和目标参数名称，便于定位 Shape 或 Sharding 问题。([错误上下文](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L67-L77))

### 3. 流式加载与严格 Checkpoint 验证

**问题**

* 旧加载器先将所有 Safetensors Tensor 放进 Python Dictionary，增加了主机内存占用。([原始 Checkpoint 收集](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L54-L67))
* 意外、缺失或重复的 Tensor 可能导致模型仅完成部分初始化，却没有明确报错。([原始宽松加载](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/utils/loader.py#L68-L170))

**示例**

如果两个 Safetensors Shard 都包含 `model.layers.0.self_attn.q_proj.weight`，接受第二个值会使模型结果依赖文件顺序。如果缺少 `k_proj.weight`，`qkv_projection` 的 K 区域会保留无关的初始化内存，但加载器仍可能显示成功。这两种情况都应该确定性失败。

**修复**

* 解析本地目录或 Hugging Face Snapshot，按文件名排序 Safetensors 以保证遍历确定性，只在迭代期间打开当前文件，并逐个 Yield `(name, tensor)`，不再将完整 Checkpoint 累积到内存。([流式迭代器](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L127-L146))
* 跟踪已加载参数对象 ID 而不是名称，使 Tied Parameter 能正确计为已初始化。迭代结束后，将已加载 ID 与全部模型参数比较，同时收集意外源名称，再通过一条可操作错误报告完整性问题。([完整性检查](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L79-L110))
* 拒绝 Safetensors Shard 之间重复的 Tensor 名称，并输出已加载参数数量。([Checkpoint 入口](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/src/myvllm/utils/loader.py#L148-L160))

## 验证

新增测试模拟两个 Tensor Parallel Rank，验证 Vocabulary、Q/K/V、Output Projection、Gate/Up、Down Projection、Norm 和 LM Head 的加载，同时覆盖意外、缺失、重复和 Shape 不匹配的 Tensor。([测试](https://github.com/KevinWu8192/MyMinivLLM/blob/4d1760ded441988663c59f5bd9479a80de234731/tests/test_loader.py#L1-L118))
