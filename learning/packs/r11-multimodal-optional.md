# R11（可选）：多模态扩展

> 状态：`DRAFT / BACKLOG`
> 默认车道：`SOURCE_READ` / `RUNNABLE_CPU`；多模态大模型训练和推理标为 `REMOTE_GPU`。
> 触发条件：岗位明确要求视觉、图像 RAG、语音或多模态 Agent 时再启动。

## 1. Mission

R11 不是 R4–R10 的先决条件。它负责把已有的 Transformer、RAG、Agent 和评测能力迁移到图像/音频等非文本模态，重点回答“模态如何进入语言模型、对齐如何评估、系统如何处理大输入”。

## 2. Prerequisites

- R3：attention、位置编码、Transformer block；
- R6：token budget、缓存和服务指标；
- R7/R8/R9：检索、工具、轨迹评测；
- 基础卷积/patch embedding、采样率和图像张量 shape。

## 3. Learning outcomes

完成本包后能够：

1. 解释 image patch/token、vision encoder、projector、cross-attention 与 multimodal fusion；
2. 追踪图像 `[C,H,W]` 到视觉 token 再到语言上下文的 shape 和预算；
3. 设计图像 RAG 的索引、OCR/region evidence、引用和权限；
4. 识别多模态 hallucination、OCR 错误、分辨率/压缩偏差和跨模态对齐失败；
5. 在 CPU 小图或源码阅读中验证 shape，明确不把大模型训练写成本地完成。

## 4. Knowledge units

| 单元 | What | Why | Mechanism | 验证方式 |
| --- | --- | --- | --- | --- |
| U1 Visual representation | resize、patch、CNN/ViT、positional encoding | 图像如何变成模型可读 token | `[C,H,W] → [N_patch,D]` | 手算 patch 数和 shape |
| U2 Fusion | projector、cross-attention、early/late fusion | 模态间信息如何交换 | 视觉 token 对齐到 LM hidden size | 画一层融合流程 |
| U3 Multimodal data | image-text pair、instruction、OCR/region labels | 数据决定对齐和偏差 | filtering、dedup、license、quality | 找出一个 bad pair |
| U4 Image RAG | OCR、caption、region/vector retrieval | 图像证据需要可定位 | page/region/span metadata、rerank | 设计带区域引用的 schema |
| U5 Eval/safety | VQA、grounding、OCR、bias、refusal | 文本 BLEU/accuracy 不足以覆盖视觉风险 | modality-specific rubric + adversarial set | 写 5 个视觉失败样例 |

## 5. Mac validation lane

- `RUNNABLE_CPU`：用少量本地图像实现 patchify、位置编码、简单投影和 shape 断言；
- `SOURCE_READ`：阅读所选 vision-language model 的 config/modeling 入口和 processor；
- `MPS_OPTIONAL`：若依赖可用，只跑小图前向烟测；不做大模型训练或速度排名；
- `REMOTE_GPU`：视觉语言预训练、长视频、量化和高分辨率 benchmark 另立项目。

## 6. Planned labs（待 R10 后启动）

1. `labs/multimodal/patchify.py`：图像 patch shape 和位置编码；
2. `labs/multimodal/projector_reference.py`：视觉 embedding 到 LM hidden size 的投影；
3. `labs/multimodal/image_rag_schema.py`：region/OCR evidence、ACL 和引用；
4. `labs/multimodal/multimodal_eval.py`：OCR、grounding、hallucination 和拒答分桶。

## 7. Failure modes

1. **patch 数或位置编码错**：图像尺寸、resize/crop 和 patch stride 没有写进 processor 配置；
2. **视觉信息在投影中丢失**：projector 维度、归一化或 token 顺序不一致；
3. **OCR/grounding 错误被语言模型掩盖**：只看流畅文本，没有区域证据和拒答指标；
4. **数据/许可证风险**：图像、字幕、OCR 文本和衍生 embedding 没有记录来源与使用边界；
5. **硬件结论越界**：CPU/MPS smoke test 不能代表高分辨率、多图或视频模型的 GPU 性能。

## 8. Interview rehearsal

- **30 秒**：说明图像如何变成语言模型可用的 token，以及一个对齐风险。
- **3 分钟**：比较 patch embedding、projector、cross-attention 和 image RAG 的职责。
- **15 分钟白板/代码**：追踪 `[C,H,W]` 到语言上下文的 shape，设计区域引用和多模态评测。

## 9. Acceptance gate

- [ ] 能画出图像到语言 token 的 shape；
- [ ] CPU patchify/projector reference 通过断言；
- [ ] 设计一套带区域证据的 image RAG schema；
- [ ] 解释至少三个多模态失败模式和评测方式；
- [ ] 只有岗位需要且通过 R10 治理门禁后，才把本包转为 active。

## 10. Primary sources

- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [CLIP](https://arxiv.org/abs/2103.00020)
- [Hugging Face Transformers multimodal documentation](https://huggingface.co/docs/transformers/main/en/tasks/image_text_to_text)
- [Hugging Face image-text-to-text task guide](https://huggingface.co/docs/transformers/main/en/tasks/image_text_to_text)
