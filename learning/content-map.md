# Job Interview 内容地图与维护规则

## 1. 内容分层

本项目把“教材、证据、实践、外部资料”分开维护。文件可以继续保持现有位置，先通过索引治理，不做高风险搬迁。

### A. 面试教材层：根目录 Markdown

文件：`01-transformer-attention.md` 到 `14-multimodal-vlm.md`、`07-project-deep-dive.md`、`08-algorithm-coding.md`。

- 这是用户提供/ChatGPT 整理的广度资料和面试题解释；
- 适合查主题、查问法、查遗漏；
- 不作为唯一答案来源；
- 长文中的版本、参数、性能结论必须回到 `sources/` 的官方来源核对；
- 不把整篇长文复制到 `learning/lessons/`，只提炼当前要学的小单元。

### B. 面经证据层：真实面试

文件：`09-2026-real-interviews.md`、`sources/2026-08-19-sources.md` 中的面经部分。

- 作用是决定优先级和追问方向；
- 一道题出现过，不等于技术结论正确；
- 同一道题跨公司重复出现，才提升为高优先级；
- 每道面经题应链接到一个专题、一个 Lab 或一个官方来源；
- 面经原文不应该被改写成“权威答案”。

### C. 教学层：`learning/`

这里是当前唯一的连续课程：

```text
MISSION / NOTES / RESOURCES
        ↓
lessons/          单个短课，实际学习入口
reference/        速查和压缩知识
packs/            阶段课程地图和验收门禁
learning-records/ 通过演示后才写的掌握记录
```

任何新知识要进入主线，必须先有：What、Why、Mechanism、Code/Source、Trade-off、Interview、Validation。

### D. 实践证据层：`labs/`

- `labs/*/README.md` 是实验入口；
- `*.py` 是最小可读实现；
- `test_*.py` 是明确的 correctness 边界；
- 实验没有测试或运行输出时，只能标为 planned/unverified；
- GPU 专题默认拆成 `RUNNABLE_CPU`、`MPS_OPTIONAL`、`SOURCE_READ`、`REMOTE_GPU`。

### E. 权威来源层：`sources/`

- `sources/learning-resources-research.md`：官方课程、论文、代码和实验入口；
- `sources/2026-08-19-sources.md`：面经证据和 Authority 列表；
- 新来源进入前必须记录 URL、来源类型、日期/版本、适用主题和最小练习；
- `latest/main` 页面不能直接当作永久可复现版本。

### F. Agent 原始与现代化层

```text
reference/GenAI_Agents/  原始仓库，只读，固定 revision
          ↓ 迁移/版本隔离
modernized/              现代化副本、manifest、离线 smoke、独立环境
```

- 原始仓库包含 54 个 Notebook、数据、图片和音频，体量大且有自定义许可证；
- 现代化副本当前 51 个 `migrated`、3 个 `review`；
- Agent 主线只从 Phase 1 五个 Notebook 开始；
- `modernized/` 的 Notebook 不是第一章教材，也不是全部必须运行的作业；
- 真实 provider、网络、API key 和付费服务默认关闭。

### G. 操作层：`tasks/`

`tasks/plan.md`、`tasks/todo.md` 和阶段计划是施工记录，不是学习材料。它们可以告诉我们迁移是否完成，但不能替代课程、Lab 或学习记录。

## 2. 现在读什么，先不读什么

| 优先级 | 现在动作 | 暂停内容 |
| --- | --- | --- |
| P0 | `learning/lessons/0001-0003`、R2/R3 reference、foundations/attention Labs | 51 个 Agent Notebook、GraphRAG、vLLM benchmark |
| P1 | R4 `0004-0006` 与 `labs/training` | DPO/GRPO 全套应用和真实模型微调 |
| P2 | R5/R6/R7 | 大规模应用 Notebook、外部服务接入 |
| P3 | R8 Agent 五个 Phase 1 Notebook | 其余 Agent 应用案例 |
| P4 | R9/R10 综合评测和面试 | R11 多模态，除非岗位明确要求 |

## 3. 新资料进入规则

### ChatGPT/用户下载资料

1. 先放入根目录 `inbox/`，不要直接增加根目录编号 Markdown；
2. 写下来源日期、原始文件名、主题和是否包含个人隐私；
3. 去重后决定：面经证据 → `09`/`sources`，稳定知识 → `learning`，只是线索 → 留在 inbox；
4. 通过官方来源核验后，才提炼到课程或 knowledge；
5. 原始文件不改写，改写版本注明来源和日期。

### 网上官方资料

1. 先进入 `sources/learning-resources-research.md` 的资源表；
2. 记录 primary URL、版本/commit、检索日期、先修和最小练习；
3. 课件引用官方文档，Lab 记录实际环境和运行结果；
4. 未验证的 API、参数、性能和 GPU 结果标成 `unverified`。

### Agent 资料

1. 原始仓库只放 `reference/`；
2. 修改和版本升级只放 `modernized/`；
3. 每个迁移必须有 manifest source/target/track/phase/status；
4. 先做 offline smoke，再开启 live provider；
5. 应用案例不能反过来定义 Agent 基础知识，基础知识仍来自 R8 教学包和官方文档。

## 4. 进入长期 knowledge 的门禁

一条内容只有同时满足以下条件，才允许晋级：

- 能不看资料说出定义、机制和一个反例；
- 有 CPU Lab 测试、源码调用链或可追溯官方证据；
- 能回答一个“为什么”和一个“失败怎么办”；
- 已标记来源、版本/日期和硬件证据等级；
- 经过用户确认，且不是仅仅“读过”。

面经、版本号、个人错题和未验证性能留在本项目，不直接复制到长期 knowledge。
