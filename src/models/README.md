# Llama 3 Architecture

### RMSNorm
RMSNorm is a simplification of Layernorm

### RoPE Positional Embeddings


### KV-Cache
The idea for KV Cache is that during inference, there are a lot of repeated computations.

For instance, lets say Q is 4 x 4096, K^T is 4 x 4096. We are interested in generating the next token (token number 4). However, during the computation of the 2nd and 3rd token, entries of QK^T did not change, but we need to recompute this dot product if we dont do any caching. 

<a href="https://ibb.co/spyp8CKs"><img src="https://i.ibb.co/spyp8CKs/Screenshot-2025-05-24-at-1-10-13-PM.png" alt="Screenshot-2025-05-24-at-1-10-13-PM"></a>

- In this example, we already generated the upper left triangle of entries. If we dont do any caching, we would have to recompute this 4 x 4 with those already computed entries
- The model also dont need to compute upper right triangle (cuz model is casual, tokens can only attend to previous tokens)

With KV Caching, you cache the **keys** and **values**. So the attention scores are always the dot product btw Q (the current token) with K (prev tokens that are cached + current token), with V being same as K.

### Grouped Multi-Query Attention
