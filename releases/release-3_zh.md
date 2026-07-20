# 摘要 — Commit ID 区间：[`686b547`](https://github.com/KevinWu8192/MyMinivLLM/commit/686b547b798518fcbd558e51e3d0a387c1bdbe07)

**Tag：** [`release-3`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-3)

此 Release 修复 Prefix Cache 命中后的 Prefill Attention，使其读取缓存前缀与新 Token 组成的完整 KV Context，而不是只关注未缓存的后缀。

## 主要修复

### 1. 分离 Query 与 KV 序列长度

**涉及文件**

* `src/myvllm/layers/attention.py`

**问题**

Prefix Cache 命中后，ModelRunner 只为未缓存 Token 计算 Q/K/V。旧 Flash Attention Kernel 将缩短后的 Query 长度同时用作 Q 和 K/V 长度，导致新 Token 无法关注缓存前缀。([原始 Kernel](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/layers/attention.py#L112-L211))

**示例**

假设请求 B 有 128 个 Prompt Token，并复用请求 A 的前 96 个 Token。ModelRunner 正确地只提交 32 个未缓存 Token 作为 Q/K/V，因此 `seq_len_q = 32`；但 Attention Context 仍应包含 128 个 K/V Position。旧 Kernel 同样将 K/V 循环长度设置为 32，导致 B 的后缀只能关注本次调用计算的 Token 96–127，完全忽略 Prefix Cache 中的 Token 0–95。Cache Hit 本应与完整 Prefill 数学等价，此时模型输出却会发生变化。

**修复**

* 增加独立的 `cu_seqlens_q` 和 `cu_seqlens_k`：Q 表示本次新计算的 Token，K 表示包含缓存 Token 在内的完整 Context。([Kernel 输入](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L112-L151))
* 对变长 Batch 中的每个请求计算 `num_cached_tokens = seq_len_k - seq_len_q`。([长度计算](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L139-L151))
* 根据未缓存 Q 长度设置 Query Grid，同时遍历完整 Context 长度的 K/V。([Kernel 边界](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L153-L180)，[Launch Grid](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L300-L321))

### 2. 从 Paged Cache 读取 Prefix K/V

**问题**

连续 K/V Tensor 只包含本次 Prefill 调用中新计算的 Token。缓存 Prefix K/V 位于分页 KV Cache Block 中，无法通过旧的连续 Tensor 布局访问。([原始连续 K/V 寻址](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/layers/attention.py#L159-L195))

**示例**

假设 `block_size = 16`，逻辑 KV Position 37 位于逻辑 Block 2 的 Offset 5。如果 `block_table[2] = 11`，它的物理位置是 Cache Block 11 的 Offset 5，而不是当前连续 K Tensor 的第 37 行。Prefix 复用后，逻辑 Block 可以映射到任意物理 Block，因此必须进行这层间接寻址。

**修复**

* 将 K/V Cache Tensor、请求 Block Table、Block Size 和最大逻辑 Block 数传入 Triton Kernel。([Prefill 接口](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L251-L281))
* 对每个逻辑 K/V Offset，计算 `logical_block = offset // block_size` 和 `block_offset = offset % block_size`，从请求 Block Table 读取物理 Block ID，再结合 KV Head 和 Head Dimension Stride 得到缓存地址。([Paged K 寻址](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L182-L207)，[Paged V 寻址](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L228-L233))
* Attention 前先将当前后缀 K/V 写入 Paged Cache，使 Paged 路径能通过统一地址空间读取旧 Prefix 和新后缀。对于没有 Prefix Cache Hit 的请求，编译 `USE_PAGED_KV` 特化以保留连续内存快速路径；禁用分支接收的占位指针不会被解引用。([分派逻辑](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L306-L340))

### 3. 修正 Causal Mask 坐标

**问题**

Q Offset 相对于未缓存后缀，而 K/V Offset 相对于完整请求。直接比较会错误地屏蔽有效的 Prefix Cache Token，并产生不正确的位置关系。([原始 Causal Mask](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/layers/attention.py#L176-L182))

**示例**

命中 96 个缓存 Token 后，后缀 Query Offset 0 的绝对位置是 96，它可以关注 K/V Position 0–96。若直接比较原始 Offset，Kernel 会把它当成绝对位置 0，只允许关注 K/V Position 0。将 Query Offset 加上缓存长度后，才能得到正确的完整 Context 坐标 96。

**修复**

应用 Causal Mask 前，将 Q 坐标加上 `num_cached_tokens`，使 Q 与 K/V 都相对于完整 Context。([Causal Mask](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L213-L219))

## 结果

对于包含 96 个缓存前缀 Token 和 32 个新后缀 Token 的请求，Kernel 只需计算 32 行 Query，同时每个 Query 都能正确关注完整的 128 Token KV Context。
