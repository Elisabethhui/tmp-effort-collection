# Job Interview 学习入口

> 这是一份“控制面板”，不是新的教材。每次学习只从这里选一个入口，不再从根目录的几十个 Markdown 文件随机开始。
> 当前主线：`R2 基础 → R3 Transformer → R4 训练闭环`。Agent、RAG、Serving 和分布式资料先保留，不在第一章同时展开。

## 1. 先记住这条规则

```text
学习入口：learning/lessons + learning/reference
动手证据：labs/
技术权威：sources/ + 每个课件里的 Primary sources
面试信号：09-2026-real-interviews.md
广度教材：01～14 根目录 Markdown
Agent 原始资料：reference/GenAI_Agents/（只读）
Agent 改写资料：modernized/（选定后再运行）
```

不要把“资料存在”当成“现在要学习”。当前只激活一个学习流：第一章 Transformer/Attention。

## 2. 当前内容到底是什么

| 目录/文件 | 来源性质 | 用途 | 当前动作 |
| --- | --- | --- | --- |
| `01-transformer-attention.md` ～ `14-multimodal-vlm.md` | 用户提供的面试教材/ChatGPT 整理资料；初始导入时一次性进入仓库 | 做广度地图、检索关键词、发现面试追问 | 只读相关小节，不从头通读 |
| `09-2026-real-interviews.md` | 面经证据 | 判断“最近问什么” | 只用于选题，不作为答案权威 |
| `sources/2026-08-19-sources.md` | 面经来源 + Authority 索引 | 区分面经证据与论文/官方资料 | 查出处和日期 |
| `sources/learning-resources-research.md` | 已核验的官方课程、论文、官方代码清单 | 选择主教材和实验来源 | 每次只读一个资源小节 |
| `learning/` | 已整理的正式教学层 | 当前唯一的连续学习入口 | 按 Lesson 顺序执行 |
| `labs/` | 可运行代码和测试 | 证明“我真的会写、会测、会诊断” | 每学一个单元跑对应测试 |
| `reference/GenAI_Agents/` | 外部 Agent 仓库的只读原始副本，固定 revision | 迁移对照、查原始意图 | 第一章不打开 |
| `modernized/` | 对 Agent 原 Notebook 的版本化改写副本 | R8 Agent 阶段的实践材料 | 第一章不运行；以后只选 5 个主干 Notebook |
| `tasks/` | 迁移/实现过程的工作计划 | 管理工作，不是教材 | 不作为学习入口 |
| `learning/learning-records/` | 学习者实际掌握后的记录 | 记录已验证理解 | 现在为空是正确状态 |

### 来源优先级

```text
论文 / 官方文档 / 官方代码
        ↓
learning/ 里的已引用教学页
        ↓
根目录专题 Markdown 与面经索引
        ↓
Agent 应用 Notebook 示例
```

如果根目录 Markdown 和官方资料冲突，保留冲突记录，以官方资料为答案依据；如果面经和教材冲突，面经只说明题目可能出现过。

## 3. 第一章到底从哪里开始

### 第一章 = Transformer / Attention 主干

根目录的 [01-transformer-attention.md](./01-transformer-attention.md) 是 926 行的广度教材，不适合第一次顺读。第一章的正式入口是下面三节短课：

1. [Lesson 0001：R2 基础](./learning/lessons/0001-r2-foundations.html)
2. [Lesson 0002：Attention Shapes](./learning/lessons/0002-r3-attention-shapes.html)
3. [Lesson 0003：Transformer Block](./learning/lessons/0003-r3-transformer-block.html)
4. [R3 Transformer 速查](./learning/reference/transformer.html)
5. [第一章对应 Labs](./labs/README.md)：foundations + attention

`01-transformer-attention.md` 只做三件事：

- 课前用标题定位 Q/K/V、scale、mask、MHA/GQA、RoPE、Norm、FFN；
- 课后查漏和准备面试追问；
- 不把其中的每一个工程专题都提前展开到 R6/R9。

## 4. 第一章执行步骤

每次只执行一个 60–90 分钟单元。固定顺序是：

```text
回忆 → 读一个小节 → 写 shape/公式 → 跑 Lab → 做 retrieval questions
→ 30 秒回答 → 记录 unknown → 再进入下一单元
```

### Session 0：环境和范围（10 分钟）

只做一次：

```bash
cd /Users/huguoqing/Documents/ChatGPT/Learn_project/job_interview/tmp-effort-collection-v2
.venv/bin/python -m unittest labs/foundations/test_algorithms.py labs/foundations/test_math_ops.py -v
.venv/bin/python -m unittest labs/attention/test_mha.py labs/attention/test_transformer_block.py -v
```

目标不是刷测试，而是确认 CPU 基线可用。不要安装 Agent 依赖，不要打开 Notebook，不要下载模型。

### Session 1：R2 基础诊断（60–90 分钟）

- 阅读 [Lesson 0001](./learning/lessons/0001-r2-foundations.html)；
- 回忆 stable softmax、cross entropy、梯度、Adam、tensor/device；
- 跑 `labs/foundations/` 对应测试；
- 写一张纸：`logits → loss → gradient → optimizer`；
- 输出一个 30 秒回答：“loss 有数值但参数不更新，我怎么查？”

通过标准：能解释 shape/dtype/device、mask 和 loss 的关系。不会的部分标为 `unknown`，不要跳去看 RAG/Agent。

### Session 2：Attention Shapes（60–90 分钟）

- 阅读 [Lesson 0002](./learning/lessons/0002-r3-attention-shapes.html)；
- 对照 `01-transformer-attention.md` 中的 Q/K/V、scaled dot-product 和 causal mask 小节；
- 阅读 [MHA Lab](./labs/attention/mha.py)；
- 跑 `labs/attention/test_mha.py`；
- 手写并口述：`[B,T,D] → [B,H,T,Dh] → scores [B,H,T,T] → context`。

通过标准：能说明为什么除以 `sqrt(Dh)`、causal mask 屏蔽谁、padding mask 屏蔽谁。

### Session 3：Transformer Block（60–90 分钟）

- 阅读 [Lesson 0003](./learning/lessons/0003-r3-transformer-block.html)；
- 对照 [Transformer Lab](./labs/attention/transformer_block.py)；
- 跑 `labs/attention/test_transformer_block.py`；
- 解释 Pre-Norm、RMSNorm、RoPE、SwiGLU、GQA；
- 画出：`x → norm → attention → residual → norm → FFN → residual`。

通过标准：把一个未来 token 改掉时，能解释为什么过去位置的输出不应改变。

### Session 4：第一章代码整合（60–90 分钟）

```bash
.venv/bin/python -m unittest discover -s labs -p 'test_*.py' -v
```

只关注 `foundations` 和 `attention` 的测试结果，暂时不要求理解所有后续 Lab。然后完成：

- 一张 Transformer shape 总图；
- 两道 30 秒回答；
- 一道 3 分钟回答；
- 一个你主动制造并解释的 mask/shape 错误。

### Session 5：第一章验收（30–45 分钟）

不看资料回答：

1. 为什么 Attention 要缩放？
2. MHA、MQA、GQA 的 K/V 存储差异是什么？
3. Pre-Norm 和 Post-Norm 的训练取舍是什么？
4. RoPE 作用在 Q/K 还是 V？为什么？
5. 如何验证 causal mask 没有信息泄漏？

验收分三档：

- 30 秒：定义 + 机制 + 一个边界；
- 3 分钟：补 shape、公式、trade-off 和失败模式；
- 15 分钟：现场写最小 attention 或读代码调用链。

只有这五题和对应 Lab 都能讲清，才进入 [R4 训练闭环](./learning/lessons/0004-r4-causal-loss.html)。

## 5. 第一章之后的顺序

```text
第一章：R2/R3 基础 + Transformer
        ↓ gate
R4：训练闭环 / Decoder LM
        ↓
R5：SFT / LoRA / DPO / GRPO
        ↓
R6：推理 / KV Cache / Serving 源码
        ↓
R7：RAG / 搜索
        ↓
R8：Agent Runtime / MCP / Memory
        ↓
R9：评测 / 分布式
        ↓
R10：项目证据 / 模拟面试
```

Agent 资料不是当前第一章的补充阅读。到 R8 时，只先选现代化副本 Phase 1 的五个 Notebook：

1. `langgraph-tutorial.ipynb`
2. `simple_conversational_agent.ipynb`
3. `mcp-tutorial.ipynb`
4. `agent_hackathon_genAI_career_assistant.ipynb`
5. `HR_AI-Assistant.ipynb`

其余 46 个应用案例先当案例库，3 个 AutoGen/PydanticAI 文件保持 `review`，不进入主线。

## 6. 每次学习结束要留下什么

每次只留下四项：

1. 一个公式或 shape 图；
2. 一个测试结果和一个失败样例；
3. 一段 30 秒/3 分钟回答；
4. 一条 `unknown` 或下一步。

不要现在创建 learning record；只有你实际回答、被追问、纠正误解后，才写入 [learning-records](./learning/learning-records/README.md)。
