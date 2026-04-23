Implement an efficient KV-cache system for autoregressive transformer inference from scratch.

Requirements:
1. Support incremental decoding (one token at a time).
2. Avoid recomputing attention for past tokens.
3. Handle:
   - multi-head attention
   - batching with variable sequence lengths
4. Provide:
   - data structure layout (memory format)
   - update logic per step
   - attention computation using cached keys/values

Additionally:
- Analyze memory growth over long sequences.
- Propose at least two optimizations (e.g., paged attention, chunking, compression).
- Explain how this would map to GPU execution.

Do not use any frameworks.