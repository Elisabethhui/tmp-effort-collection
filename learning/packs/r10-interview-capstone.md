# R10：面试治理与综合项目

> 状态：`DRAFT`
> 默认车道：`RUNNABLE_CPU` / `SOURCE_READ`；项目证据必须区分本地实测、源码阅读和远端 GPU。
> 目标：把 R2–R9 的知识组织成可检索、可演示、可追问的面试能力。

## 1. Mission

学习项目的终点不是收藏更多资料，而是面对一道陌生题时能快速建模、写最小验证、说明取舍并诚实报告证据。R10 建立面试治理层：

```text
question registry → concept graph → 30s/3min/15min answer
                  → code/source evidence → mock interview
                  → error log → spaced review → knowledge promotion
```

综合项目建议串起一条可解释链：`tiny decoder LM → SFT/LoRA toy → CPU KV cache → grounded RAG → durable Agent → eval/trace`。不要求把所有系统做成生产服务。

## 2. Prerequisites

- R2–R3：数学、算法、PyTorch、Transformer；
- R4–R9：至少完成每包 acceptance gate 的核心部分；
- 已有 `learning/lessons/`、`learning/reference/`、`learning/packs/` 和 labs；
- 能维护 Git 提交、实验日志、来源链接和硬件标签。

## 3. Learning outcomes

完成本包后能够：

1. 把面经问题按概念、代码、系统、权衡和行为证据分类，而不是按日期堆积；
2. 对每个高频问题准备 30 秒、3 分钟、15 分钟三档回答，并能根据追问展开；
3. 用最小代码、公式、shape 图或源码调用链支撑答案；
4. 在 Mac 约束下诚实说明“可运行、可选 MPS、源码阅读、远端 GPU”四类证据；
5. 通过模拟面试后的错误日志和间隔复习，决定哪些内容进入长期 `knowledge/`。

## 4. Knowledge units

| 单元 | What | Why | Mechanism | 验证方式 |
| --- | --- | --- | --- | --- |
| U1 Question registry | question、tags、difficulty、source、status | 便于按岗位和薄弱点复习 | YAML/Markdown schema、去重、版本 | 将 20 道题归入概念图 |
| U2 Answer ladder | 30s/3min/15min | 面试时间和追问深度不固定 | 结论→机制→证据→trade-off→failure | 同一题录三段回答 |
| U3 Evidence portfolio | lab、test、source map、benchmark | 让项目证据可审计 | commit、环境、输入输出、限制 | 为三个项目写 evidence card |
| U4 Mock interview | warm-up、coding、system、follow-up | 暴露检索失败而非阅读幻觉 | 计时、随机题、评分 rubric | 完成两轮并记录错误 |
| U5 Governance | learning record、knowledge promotion、退役 | 防止过时/未验证内容污染知识库 | 状态、owner、review date、source | 审核一条知识条目 |
| U6 Capstone | end-to-end mini system | 证明跨阶段整合能力 | data → model → retrieval/tool → eval | 10 分钟演示 + 15 分钟追问 |

### 三档回答模板

- **30 秒**：先给结论，再给一个机制和一个边界；
- **3 分钟**：补 shape/公式、最小流程、关键 trade-off 和一个失败模式；
- **15 分钟**：白板或代码验证，说明数据、实验、指标、硬件、版本和未验证部分。

不要把“我看过某论文”当作证据；证据应能被复跑、复述或从官方源码定位。

## 5. Mac validation lane

- `RUNNABLE_CPU`：所有 capstone 组件都提供 tiny/offline path；
- `SOURCE_READ`：GPU/分布式/服务专题保留调用链、公式、账本和限制；
- `MPS_OPTIONAL`：只记录可选 smoke test，不与 CUDA 结果混表；
- `REMOTE_GPU`：如果以后使用远端 GPU，单独保存 run manifest、硬件、版本、数据规模和成本。

## 6. Planned labs（本包后续实现）

1. `learning/question-registry.yaml`：题目、标签、阶段、难度、来源和复习状态；
2. `labs/capstone/evidence_card.md`：统一记录输入、输出、测试、commit、硬件和限制；
3. `labs/capstone/mock_interview.py`：按阶段随机抽题、计时、记录评分和 follow-up；
4. `labs/capstone/error_log.md`：把错误归类为 recall、concept、implementation、system 或 communication；
5. `labs/capstone/capstone_checklist.md`：端到端演示和发布前检查。

## 7. Failure modes

1. **题库越大越安心**：题目数量替代不了间隔检索和错误闭环；
2. **回答只背结论**：追问到 shape、代码或失败模式就断裂；
3. **项目描述夸大**：把源码阅读、toy CPU 验证写成生产 GPU 经验；
4. **知识库复制未验证资料**：没有来源、日期、硬件和 review owner；
5. **模拟面试只做一次**：没有错误日志和二次复测，无法判断是否真正改善；
6. **综合项目范围失控**：先固定 tiny/offline acceptance，再增量扩展真实模型或服务。

## 8. Interview rehearsal

- **30 秒**：介绍你的学习系统如何从原理到代码再到面试证据。
- **3 分钟**：任选一个 Transformer/推理/RAG/Agent 题，展示三档回答的切换。
- **15 分钟**：演示 capstone，接受 shape、故障、扩展、成本、硬件和治理追问。

推荐 retrieval questions：

1. 你做过的实验哪些是 CPU correctness，哪些只是 source reading？
2. 如果线上 latency 变差，你如何从 trace、cache、queue 和模型逐层排查？
3. 你如何决定一条笔记是否进入长期 knowledge？

## 9. Acceptance gate

- [ ] 题库至少覆盖 R2–R9，每类有来源、难度和复习状态；
- [ ] 至少 20 道题完成三档回答，其中 5 道能现场写代码/画图；
- [ ] 至少 3 张 evidence card，明确 test、commit、硬件和限制；
- [ ] 完成两轮模拟面试，并对错误做二次检索；
- [ ] 完成一次端到端 capstone 演示和一次故障追问；
- [ ] 只有通过上述门禁、用户确认且来源仍有效，才把条目晋级到 `knowledge/`。

## 10. Primary sources

- [MIT 6.006 Introduction to Algorithms](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html)
- [Hugging Face documentation](https://huggingface.co/docs)
- 本项目各包的 `Primary sources` 和实验提交记录
