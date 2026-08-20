# Lab｜从零手写 Multi-Head Attention

## 目标

不用 `nn.MultiheadAttention`，独立实现：

- `[B, T, D] → [B, H, T, D_h]` 的 reshape；
- scaled dot-product attention；
- causal mask；
- padding mask；
- stable softmax；
- head 拼接和 output projection。

实现依据 PyTorch 官方 Transformer building blocks 教程以及
`torch.nn.functional.scaled_dot_product_attention` 的定义：

- https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

## 运行

在已安装 PyTorch 的隔离环境中执行：

```bash
python -m unittest labs/attention/test_mha.py -v
```

当前仓库只提供实验代码和测试，不把依赖直接装进用户环境。

## 验收标准

1. 输出 shape 正确；
2. 无 dropout 时，与官方 SDPA 在小样本上数值对齐；
3. causal mask 不允许看到未来 token；
4. padding token 不参与 key/value 聚合；
5. 修改 padded value 不影响非 padded query 的输出；
6. 能解释 `sqrt(d_head)`、mask 广播和 `contiguous()` 的原因。

## Transformer Block 扩展

[`transformer_block.py`](./transformer_block.py) 在 MHA 之上加入：

- pre-norm residual；
- RMSNorm；
- RoPE；
- GQA 的 K/V head repeat；
- SwiGLU FFN；
- causal future-isolation 测试。

运行：

```bash
python -m unittest labs/attention/test_transformer_block.py -v
```

这是 `RUNNABLE_CPU` 的 reference implementation，不代表 FlashAttention、PagedAttention 或 CUDA kernel 性能。
