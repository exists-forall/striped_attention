# Striped Attention: Faster Ring Attention for Causal Transformers

William Brandon, Aniruddha Nrusimha, Kevin Qian, Zachary Ankner, Tian Jin, Zhiye Song, Jonathan Ragan-Kelley

Based on _Ring Attention with Blockwise Transformers for Near-Infinite Context_ by Hao Liu et al. (https://arxiv.org/abs/2310.01889)

---

This is an implementation of Striped Attention built as an extension to Hao Liu et al.'s implementation of Ring Attention in JAX (https://github.com/lhao499/llm_large_context).

The implementation adds a new model configuration option, `attention_type='striped'`, which enables Striped Attention.

Information about the underlying Ring Attention implementation, and instructions on how to run it, are included in `README_RING_ATTN.md`.
