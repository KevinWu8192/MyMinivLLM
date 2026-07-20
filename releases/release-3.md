# Summary — Commit ID Range: [`686b547`](https://github.com/KevinWu8192/MyMinivLLM/commit/686b547b798518fcbd558e51e3d0a387c1bdbe07)

**Tag:** [`release-3`](https://github.com/KevinWu8192/MinivLLM-Fixed/tree/release-3)

This release makes Prefill Attention consume the complete cached-and-new KV context after a Prefix Cache hit instead of attending only to the uncached suffix.

## Major Fixes

### 1. Separate Query and KV Sequence Lengths

**File**

* `src/myvllm/layers/attention.py`

**Bug**

After a Prefix Cache hit, the ModelRunner computes Q/K/V only for uncached tokens. The old Flash Attention kernel used that shortened query length for both Q and K/V, so new tokens could not attend to the cached prefix. ([Original kernel](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/layers/attention.py#L112-L211))

**Example**

Suppose request B has 128 Prompt Tokens and reuses the first 96 Tokens from request A. ModelRunner correctly submits only the 32 uncached Tokens as Q/K/V, so `seq_len_q = 32`; however, the attention context must still contain 128 K/V positions. The old kernel also set its K/V loop length to 32. Consequently, B's suffix attended only to Tokens 96–127 computed in the current call and completely ignored Tokens 0–95 from the Prefix Cache, changing the model output even though the cache hit should be mathematically equivalent to a full Prefill.

**Fixes**

* Add separate `cu_seqlens_q` and `cu_seqlens_k`: Q describes newly computed tokens, while K describes the full context including cached tokens. ([Kernel inputs](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L112-L151))
* Derive `num_cached_tokens = seq_len_k - seq_len_q` for every request in a variable-length batch. ([Length calculation](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L139-L151))
* Size the query grid from uncached Q lengths while iterating K/V across the complete context length. ([Kernel bounds](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L153-L180), [launch grid](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L300-L321))

### 2. Read Prefix K/V from the Paged Cache

**Bug**

The contiguous K/V tensors contain only tokens computed in the current Prefill call. Cached Prefix K/V lives in paged KV Cache blocks and therefore cannot be addressed through the old contiguous tensor layout. ([Original contiguous K/V addressing](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/layers/attention.py#L159-L195))

**Example**

With `block_size = 16`, logical KV position 37 belongs to logical Block 2 at offset 5. If `block_table[2] = 11`, its physical cache location is Block 11, offset 5—not row 37 of the current contiguous K tensor. Prefix reuse makes this indirection mandatory because logical Blocks can map to arbitrary physical Blocks.

**Fixes**

* Pass K/V Cache tensors, request Block Tables, Block size, and the maximum logical Block count into the Triton kernel. ([Prefill interface](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L251-L281))
* For every logical K/V offset, compute `logical_block = offset // block_size` and `block_offset = offset % block_size`, load the physical Block ID from the request's Block Table, and combine it with KV Head and Head Dimension strides to obtain the cache address. ([Paged K addressing](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L182-L207), [Paged V addressing](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L228-L233))
* Store the current suffix K/V into the paged cache before attention, allowing the paged path to read both the old Prefix and new suffix through one address space. Keep the contiguous fast path for requests without a Prefix Cache hit by compiling a `USE_PAGED_KV` specialization; placeholder pointers are passed but never dereferenced in the disabled branch. ([Dispatch](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L306-L340))

### 3. Correct Causal Mask Coordinates

**Bug**

Q offsets are relative to the uncached suffix, whereas K/V offsets are relative to the full request. Comparing them directly masks valid Prefix Cache tokens and permits incorrect positions. ([Original causal mask](https://github.com/KevinWu8192/MyMinivLLM/blob/798b4552a1401f1eebbfff865816a95169aada5f/src/myvllm/layers/attention.py#L176-L182))

**Example**

After a 96-Token cache hit, suffix Query offset 0 is absolute position 96. It may attend to K/V positions 0 through 96. Comparing the raw offsets instead treats it as absolute position 0, so the causal mask allows only K/V position 0. Adding the cached length converts Query offset 0 to the correct full-context coordinate 96.

**Fix**

Shift Q coordinates by `num_cached_tokens` before applying the causal comparison, making Q and K/V positions relative to the same full context. ([Causal mask](https://github.com/KevinWu8192/MyMinivLLM/blob/686b547b798518fcbd558e51e3d0a387c1bdbe07/src/myvllm/layers/attention.py#L213-L219))

## Result

For a request with a 96-token cached prefix and a 32-token suffix, the kernel computes only 32 query rows while each query can correctly attend over the full 128-token KV context.
