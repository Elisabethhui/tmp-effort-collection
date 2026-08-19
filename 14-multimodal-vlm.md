# 14｜多模态大模型：CLIP / ViT / BLIP / LLaVA / Qwen-VL 面试体系

> 2026 信号：字节抖音电商多模态面经直接问 CLIP、ViT、Swin、PPO/DPO/GRPO、LoRA；淘天多模态面经问 BLIP、BLIP2、Q-Former、LLaVA MLP adaptor、Qwen-VL 训练流程，并要求手写 MHA。

---

# 1. 多模态问题的核心

要把不同模态映射到可以联合推理的表示空间。

主要问题：

- representation alignment；
- granularity alignment；
- temporal/spatial structure；
- modality gap；
- data noise；
- catastrophic forgetting；
- hallucination；
- token/compute explosion。

---

# 2. ViT `S`

## 2.1 Patchify

把图像切成 patch：

```text
Image H×W×C
 → N patches
 → flatten + linear projection
 → visual tokens
```

再加 positional information，送入 Transformer Encoder。

## 2.2 为什么 ViT 可行

Attention 可以直接建模全局 patch interaction，不依赖 CNN 固定 local inductive bias。

### 高频追问

- patch size 变小会怎样？token 更多、计算更贵、细节更丰富。
- ViT 为什么数据需求大？inductive bias 相对弱。
- CLS token vs pooling？

---

# 3. Swin Transformer `A`

核心：window attention + shifted windows。

目的：

- 降低全局 attention 对高分辨率图像的二次复杂度；
- 引入层次化 feature map；
- shifted window 让跨窗口信息交互。

2026 字节多模态面经直接问原理。

---

# 4. CLIP `S`

2026 字节/腾讯持续出现。

## 4.1 Training

图片和文本各自编码：

```text
image → image encoder → vi
text  → text encoder  → vt
```

在 batch 内构造 similarity matrix，用 symmetric contrastive loss：

- image-to-text CE；
- text-to-image CE。

## 4.2 Zero-shot

把类别名写成 text prompts，得到 class text embeddings，再与 image embedding 做 similarity。

**Zero-shot 不是“模型天然知道所有类别”，而是语言监督让 visual representation 与文本语义对齐。**

---

# 5. BLIP `A`

需要理解三类目标的设计动机，而不是只背缩写：

- image-text contrastive：global alignment；
- image-text matching：fine-grained matching；
- language modeling：generation。

BLIP 同时兼顾理解和生成。

---

# 6. BLIP-2 / Q-Former `A`

关键思想：冻结强视觉编码器和冻结 LLM，中间用轻量 Q-Former bridging。

learnable query tokens 从视觉特征提取对语言模型有用的信息。

### 为什么不直接把所有 image patches 喂给 LLM？

- token 多；
- modality gap；
- 训练成本；
- Q-Former 提供信息压缩/对齐层。

---

# 7. LLaVA：简单 Projector 路线 `S/A`

典型：vision encoder → MLP projector → LLM token space。

优势：

- 简单；
- end-to-end；
- 工程实现容易。

相对 Q-Former：

- 缺少显式 learned query bottleneck；
- 但复杂度低、训练路线简洁。

### 淘天 2026 直接问

> Q-Former 复杂 adaptor 与 LLaVA 简单 MLP，哪个好？

**标准答案不能选绝对赢家。** 应比较数据规模、token 压缩需求、训练成本、信息瓶颈、冻结策略和任务类型。

---

# 8. Qwen-VL / 现代 VLM 的训练阶段 `A`

不同版本会有差异，因此不要背一个永恒“三阶段”。理解通用目的：

1. vision-language alignment；
2. multimodal instruction tuning；
3. 更高质量对话/任务/后训练（视模型版本）。

面试如果问具体 Qwen-VL 版本，必须先确认版本，再依据官方报告回答。

---

# 9. Position Encoding in VLM `A/B`

视觉除了序列顺序，还包含 2D/temporal structure。

需要理解：

- 1D token position；
- 2D spatial position；
- video temporal position；
- multimodal rotary variants。

不同模型实现差异很大，面试不要把所有 VLM 都说成普通 RoPE。

---

# 10. Image Resolution 与 Token Cost `S/A`

分辨率变高 → patch/token 数增加 → attention/LLM context cost 上升。

常见方案：

- adaptive resolution；
- tiling；
- token merging/resampler；
- Q-Former/perceiver；
- visual token pruning/compression。

---

# 11. Multimodal Data `S/A`

数据质量问题：

- image-text mismatch；
- OCR noise；
- duplicate；
- unsafe content；
- templated caption；
- modality imbalance；
- resolution/aspect distribution。

### 清洗

- perceptual hash / duplicate detection；
- CLIP score；
- OCR/metadata consistency；
- language quality；
- safety filter；
- curriculum / data mix。

---

# 12. Multimodal SFT `S/A`

训练需要保证：

- image token placeholder 与 encoded features 对齐；
- loss 通常主要落在 assistant output token；
- visual backbone/projector/LLM 哪些参数 trainable 明确；
- 多图/视频的 position 与 packing 正确。

---

# 13. Multimodal RL / Preference `A/↗`

2026 字节多模态面经直接把 PPO/DPO/GRPO 和多模态放在一起。

Reward 可能来自：

- human preference；
- correctness verifier；
- grounding；
- OCR/math/task-specific evaluator；
- safety。

主要风险：reward hacking、视觉证据被语言 prior 覆盖。

---

# 14. Hallucination / Grounding `S/A`

VLM 常见幻觉：描述图里不存在的对象/属性。

诊断：

- visual encoder 是否保留信息；
- projector bottleneck；
- training data bias；
- language prior 过强；
- prompt；
- decoding；
- evaluation blind spot。

不能只靠“加更多图片”解决。

---

# 15. OCR / Document VLM `A`

核心：

- high resolution；
- layout；
- text recognition；
- reading order；
- table/chart structure。

如果业务是文档/电商图片，面试可能追问 OCR+Text pipeline 为什么换 VLM，以及收益来自哪里。

---

# 16. Video / Temporal `B/A`

视频引入：

- frame sampling；
- temporal ordering；
- token explosion；
- long-range event relation；
- audio/video alignment。

需要在帧数、分辨率、时长与 token budget 之间权衡。

---

# 17. Multimodal Evaluation `A`

不要只报一个 benchmark score。

分维度：

- perception；
- OCR；
- grounding；
- reasoning；
- hallucination；
- long image/video；
- safety；
- latency/cost。

项目中还要有 task-specific eval 和 bad-case set。

---

# 18. 2026 字节抖音电商多模态面经

公开问题包括：

- SFT 模型/数据/框架；
- PPO 数据、reward model、loss、GPU 数；
- DPO 流程与 PPO 对比；
- 如何评估 finetune 提升；
- 消融实验；
- CLIP；
- optimizer；
- MHA；
- GRPO；
- LoRA rank；
- ViT；
- Swin Transformer；
- Qwen3 fast/slow thinking；
- coding。

**信号：** 多模态岗位同样要求 LLM 后训练与基础架构，而不是只会 CV。

---

# 19. 2026 淘天多模态面经

公开问题包括：

- CLIP；
- LoRA；
- 常见 VLM；
- BLIP 三个 loss；
- BLIP2/BLIP3；
- Qwen-VL 训练流程；
- Q-Former vs LLaVA MLP；
- 手写 MHA。

**信号：** “模型设计动机”非常重要：为什么要 Q-Former、为什么用 projector、为什么有多个训练 objective。

---

# 20. 高频等级

## S

- ViT
- CLIP
- contrastive learning
- vision encoder + projector + LLM
- multimodal SFT
- LoRA
- MHA

## A

- Swin
- BLIP/BLIP2
- Q-Former vs MLP
- visual token compression
- hallucination/grounding
- multimodal eval

## B

- video VLM
- multimodal RL details
- advanced spatial/temporal position encoding
- document-specific architecture

---

# 21. Labs

1. CLIP-like mini contrastive training；
2. ViT from scratch；
3. frozen image encoder + MLP projector + tiny LLM；
4. compare MLP projector vs query-based resampler；
5. high-resolution token cost benchmark；
6. hallucination bad-case eval；
7. VLM SFT data collator / label masking。

---

# 22. 自测

- [ ] 能推导 CLIP contrastive objective
- [ ] 能解释 CLIP zero-shot
- [ ] 能解释 ViT patch token
- [ ] 能解释 Swin shifted window
- [ ] 能比较 Q-Former 与 MLP projector
- [ ] 能设计多模态数据清洗
- [ ] 能解释 VLM hallucination 的多层原因
- [ ] 被问具体 Qwen-VL 版本时会先确认版本再回答
