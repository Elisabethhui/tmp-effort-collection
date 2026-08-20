# R2–R3 学习资源

执行时重新核对版本；这里保存长期入口，不把 `latest/main` 当成永久可复现版本。

## 算法、数学与 PyTorch

- [MIT 6.006 Introduction to Algorithms](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/)
- [MIT 6.006 Problem Sets](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/problem-sets/)
- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- [Dive into Deep Learning](https://d2l.ai/)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)

## Transformer 主干

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
- [PyTorch Transformer Building Blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Hugging Face Llama model notes](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/llama.md)
- [Hugging Face RoPE utilities](https://huggingface.co/docs/transformers/internal/rope_utils)
- [Hugging Face attention backends](https://huggingface.co/docs/transformers/attention_interface)

## GPU/推理源码阅读

- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Hugging Face Cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation)
- [Transformers cache_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py)
- [vLLM V1 KVCacheManager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)
- [vLLM V1 BlockPool](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)
- [vLLM prefix caching design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)

## R4 训练闭环

- [PyTorch Autograd](https://docs.pytorch.org/docs/stable/autograd.html)
- [PyTorch Optimizers](https://docs.pytorch.org/docs/stable/optim.html)
- [PyTorch saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch serialization notes](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [PyTorch Transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)

## 使用规则

1. 论文用于推导，官方文档用于 API，源码用于调用链和边界；
2. 每个单元只读一个小节，随后必须写公式、跑 Lab 或做口述；
3. 面经只能证明“有人问过”，不能替代技术来源；
4. 未验证的框架参数和 GPU 性能结论标记 `unverified`。
