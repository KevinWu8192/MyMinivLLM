# 摘要 — Commit ID 区间：[`23d95ae`](https://github.com/KevinWu8192/MyMinivLLM/commit/23d95aea25610777eb9f7249f26a74a651140e6d) → [`acdac94`](https://github.com/KevinWu8192/MyMinivLLM/commit/acdac942b3e918855125da7e03d63cba3feb8fc9)

**Tag：** [`release-1`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-1)

此 Release 建立了正确的跨请求 Prefix Cache Block 复用机制，并让 KV Cache 容量检查能够识别可复用的 Block。

## 主要修复

### 1. 跨请求保留 KV Block

**涉及文件**

* `src/myvllm/engine/block_manager.py`
* `src/myvllm/engine/sequence.py`

**问题**

* 请求释放时会清空 Block 的 Token 元数据，导致后续请求无法复用已经缓存的 Prefix。([原始释放逻辑](https://github.com/KevinWu8192/MyMinivLLM/blob/6e47fd07551321a62126d18e11060bdddaf4f67a/src/myvllm/engine/block_manager.py#L55-L60))
* Cache Hit 和新 Block 使用同一套分配流程，在复用命中结果前就会重置仍然有效的缓存元数据。([原始分配路径](https://github.com/KevinWu8192/MyMinivLLM/blob/6e47fd07551321a62126d18e11060bdddaf4f67a/src/myvllm/engine/block_manager.py#L46-L53)，[原始 Hit 处理](https://github.com/KevinWu8192/MyMinivLLM/blob/6e47fd07551321a62126d18e11060bdddaf4f67a/src/myvllm/engine/block_manager.py#L81-L90))
* Prefix Miss 后仍可能继续匹配后续 Block，但 Prefix Cache 只能复用连续前缀。([原始逐 Block 匹配](https://github.com/KevinWu8192/MyMinivLLM/blob/6e47fd07551321a62126d18e11060bdddaf4f67a/src/myvllm/engine/block_manager.py#L67-L97))
* 最后一个 Block 恰好填满时，其 Token 数量会被错误地计算为零。([原始 Sequence 逻辑](https://github.com/KevinWu8192/MyMinivLLM/blob/6e47fd07551321a62126d18e11060bdddaf4f67a/src/myvllm/engine/sequence.py#L67-L78))

**示例**

假设 `block_size = 4`，请求 A 的 Token 是 `[10, 11, 12, 13, 20]`，第一个 Block `[10, 11, 12, 13]` 可以作为 Prefix 复用。A 完成后，请求 B 携带 `[10, 11, 12, 13, 30]` 到达。旧释放逻辑会清空保存的 Token ID，因此 B 要么 Cache Miss，要么把残留的 Hash 映射判断为 Collision；即使找到了物理 Block，通用分配路径也会在复用前将其重置。

**修复**

* `_allocate_new_block()` 仅在旧 Hash 仍指向被回收 Block 时移除旧映射，然后重置元数据、从空闲队列取出 Block、写入新 Hash/Token 并初始化 `ref_count`。`_allocate_hit_block()` 只把未使用的缓存 Block 移回使用集合并增加引用计数。两条路径分离后，Cache Hit 不会重置有效数据。([分配路径](https://github.com/KevinWu8192/MyMinivLLM/blob/23d95aea25610777eb9f7249f26a74a651140e6d/src/myvllm/engine/block_manager.py#L44-L83))
* `_deallocate_block()` 现在只释放活跃所有权：物理 Block 回到空闲队列，但 Hash 和 Token ID 会被保留，直到该 Block 被回收并写入其他 Token。([释放修复](https://github.com/KevinWu8192/MyMinivLLM/blob/23d95aea25610777eb9f7249f26a74a651140e6d/src/myvllm/engine/block_manager.py#L76-L81))
* `_get_prefix_hit_blocks()` 从左到右计算 Hash，遇到第一个缺失 Hash 或 Token 不匹配立即停止。查找范围限制为 `(num_tokens - 1) // block_size`，保证 Prompt 最后一个 Token 仍会参与计算并生成第一个 Completion Logits。([Prefix 匹配](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/block_manager.py#L83-L106))
* 正确处理空序列、部分填充和恰好填满的最后一个 Block。([Sequence 修复](https://github.com/KevinWu8192/MyMinivLLM/blob/23d95aea25610777eb9f7249f26a74a651140e6d/src/myvllm/engine/sequence.py#L69-L78))

### 2. 感知 Prefix Cache 的容量检查

**问题**

`can_allocate()` 会为请求的每个逻辑 Block 要求一个空闲物理 Block。即使某些 Prefix Cache Block 正在被其他请求使用、只需增加引用计数即可共享，请求仍会被错误拒绝。([原始检查](https://github.com/KevinWu8192/MyMinivLLM/blob/23d95aea25610777eb9f7249f26a74a651140e6d/src/myvllm/engine/block_manager.py#L84-L85))

**示例**

一个请求需要三个逻辑 Block，其中第一个 Block 已经被另一个请求使用。如果只剩两个空闲物理 Block，旧逻辑比较 `2 >= 3` 后拒绝该请求。实际上只需分配两个新 Block，第一个 Block 只要增加 `ref_count` 即可共享，因此请求本应被接纳。

**修复**

* 使用 `_get_prefix_hit_blocks()` 统一发现 Cache Hit，保证容量检查与实际分配使用同一段连续 Prefix。([辅助函数](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/block_manager.py#L83-L106))
* 计算 `num_required_free_blocks = seq.num_blocks - num_shared_hit_blocks`。只有 `ref_count > 0` 的 Hit 才会被减去；引用计数为零的缓存 Block 仍位于空闲队列，实际分配时必须先从队列中领取。([容量计算](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/block_manager.py#L108-L118))
* 分配时将 Hit Block ID 写入请求 Block Table，将一个 Block 的 Token 数加入 `num_cached_tokens`，并根据 Block 是否活跃选择领取缓存 Block 或增加 `ref_count`；只有未匹配后缀会消耗新物理 Block。([分配逻辑](https://github.com/KevinWu8192/MyMinivLLM/blob/acdac942b3e918855125da7e03d63cba3feb8fc9/src/myvllm/engine/block_manager.py#L120-L145))

## 结果

具有相同完整 Block Prefix 的两个请求现在可以共享物理 KV Block。容量准入根据真正需要获取的物理 Block 数量判断，而不是根据请求的全部逻辑 Block 数量判断。
