# Job Interview｜第 2 轮及后续任务清单

> 状态：待用户审核；本清单未开始执行。
> 详细说明、依赖、验收和文件范围见 [`plan-rounds-2-plus.md`](./plan-rounds-2-plus.md)。

## 审核前：只确认范围，不写代码

- [ ] 确认目标岗位权重：算法 / LLM / Agent / RAG / Infra
- [ ] 确认每日时间预算：30 / 60–120 / 120+ 分钟
- [ ] 确认第二轮先做基础 + Transformer（R2–R3）还是 Agent（R8）
- [ ] 确认是否暂缓建立独立 `knowledge/`

## Mac 双轨规则

- [ ] `RUNNABLE_CPU`：tiny tensor、reference、单元测试
- [ ] `MPS_OPTIONAL`：只有 MPS 可用时才运行
- [ ] `SOURCE_READ`：官方源码调用链、shape/状态/内存账本、CPU toy
- [ ] `REMOTE_GPU`：Linux/CUDA 上的真实 kernel、vLLM、FSDP 或 benchmark
- [ ] 未有对应硬件时不把理论或源码阅读标成性能验证

## Phase 0：资料与治理

- [ ] Task 0.1：登记第二轮来源、Notebook 和状态
- [ ] Task 0.2：建立 Question Registry、Update Ledger 和 Knowledge Promotion Gate

### Checkpoint 0

- [ ] 54 个 Notebook 有状态，未迁移项未标为完成
- [ ] 3 道现有面经题完成来源/专题/Lab 映射

## Phase 1A：R2–R3 基础与 Transformer

- [ ] Task 1A.1：Python/算法/数学/ML-DL 基础 CPU Lab
- [ ] Task 1A.2：Transformer 结构地图与逐层 shape
- [ ] Task 1A.3：MHA → Transformer Block（RoPE/RMSNorm/SwiGLU/GQA）
- [ ] Task 1A.4：KV Cache、SDPA/Flash、PagedAttention、vLLM 源码阅读包

### Checkpoint T

- [ ] Python/数据结构、数学/优化、ML/DL 各有最小代码证据
- [ ] Transformer 的 shape、mask、RoPE、Norm、FFN、MHA/MQA/GQA 可口述和手写
- [ ] GPU 内容完成 CPU reference 或源码阅读，不要求本机 CUDA

## Phase 1B：R4 训练闭环

- [ ] Task 1.1：PyTorch train loop + checkpoint Lab
- [ ] Task 1.2：tokenizer/label shift/padding 实验

### Checkpoint A

- [ ] 训练 smoke、恢复测试和边界测试通过
- [ ] 5 道训练题完成 30 秒/3 分钟回答
- [ ] 用户审核是否进入 R5 后训练

## Phase 2：R5 后训练

- [ ] Task 2.1：LoRA/PEFT toy 实验
- [ ] Task 2.2：DPO/GRPO 完整数据流与 reward hacking

### Checkpoint B

- [ ] 能解释 SFT/LoRA/DPO/GRPO 的目标函数和边界
- [ ] 至少一个 reward hacking 反例和修复记录

## Phase 3：R6 推理与 Serving（源码优先）

- [ ] Task 3.1：推理显存与延迟账本
- [ ] Task 3.2：vLLM benchmark（硬件允许时）

## Phase 4：R7 RAG 与搜索

- [ ] Task 4.1：Hybrid Retrieval、RRF、rerank 和失败分类
- [ ] Task 4.2：Versioned index、rollback、ACL

### Checkpoint C

- [ ] Serving 结果或硬件阻塞记录完成
- [ ] RAG ablation、失败分类和权限边界完成

## Phase 5：R8 Agent Runtime

- [ ] Task 5.1：迁移 5 个 Agent Phase 1 Notebook
- [ ] Task 5.2：Memory、durability、MCP、trace 与现有 Lab 对齐

### Checkpoint D

- [ ] 静态检查通过，至少 3 个 offline smoke 通过
- [ ] retry/idempotency/checkpoint/trace 有故障注入记录

## Phase 6：R9 Eval 与 Distributed

- [ ] Task 6.1：Interview/Agent evaluation harness
- [ ] Task 6.2：DDP/FSDP2/ZeRO 理论账本与 tiny smoke

## Phase 7：R10 面试转化

- [ ] Task 7.1：20 道高频面经结构化去重
- [ ] Task 7.2：3 个项目三档回答包和两场模拟面试
- [ ] Task 7.3：首批 10 个稳定概念进入 knowledge（需单独批准）

### Checkpoint E

- [ ] 20 道题可回归评分
- [ ] 项目事实有证据、指标、bad case 和个人贡献
- [ ] knowledge 晋升项有 primary source、验证 commit 和反例

## 停止/回退条件

- [ ] 资料重复度高：停止迁移，保留索引和淘汰理由
- [ ] 版本/硬件无法验证：标为 `theory-only` 或 `blocked`，不伪造结果
- [ ] 学习负担过高：停在最近 checkpoint，先学习再继续建设
