# 08｜算法、手撕代码与 2026 AI Coding：从 LeetCode 到模型训练代码

> 目标：算法岗/大模型岗的“手撕”已经分成三类：传统 DSA、ML/LLM 手写、AI Coding/业务建模。三类都要准备。

---

## 0. 2026 真实信号

近期公开题包括：

- DeepSeek：手写完整 MHA。
- 百度：Transformers + PyTorch 实现 Qwen2 SFT。
- 字节：torch 写 SFT loss（shift-right）。
- 美团：二叉树层序、K 个一组翻转链表、最长有效括号、路径总和 III、两数之和。
- 小红书：快速排序、InfoNCE 矩阵表达与代码思路。
- 美团 2026-05-16 算法机考：HAC 聚类。
- 美团 2026-08-18：电商搜索排序 AI Coding，评价 NDCG@10。

所以不能只刷 Hot100。

---

# Part A｜传统算法基础地图

## 1. 数组 / 双指针 ★★★★★

必须会：

- Two Sum；
- 3Sum；
- remove duplicates；
- sliding window；
- prefix sum。

核心能力：把 `O(n²)` 搜索转成 hash / two-pointer / prefix structure。

---

## 2. 链表 ★★★★★

必须会：

- reverse list；
- merge lists；
- cycle；
- LRU 思想；
- K-group reversal。

### K 个一组翻转

2026 美团北斗二面出现。

要点：

1. 检查剩余是否够 K；
2. 原地 reverse K 个；
3. 正确连接上一组 tail 与下一组 head。

这题面试容易在 pointer 边界挂。

---

## 3. 树 ★★★★★

高频：

- level order；
- DFS/BFS；
- LCA；
- path sum；
- serialize。

### 层序遍历

2026 美团多次出现。核心不是难度，而是检查 BFS 基础是否稳定。

```python
from collections import deque

def level_order(root):
    if not root:
        return []
    q = deque([root])
    ans = []
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        ans.append(level)
    return ans
```

---

## 4. Stack ★★★★★

### Longest Valid Parentheses

近期美团大模型应用面经出现。

可用：

- stack of indices；
- DP。

Stack 法核心维护最后一个无法匹配的位置作为长度基准。

---

## 5. Heap / Top-K ★★★★★

大模型/推荐/检索系统非常常见，因为系统里大量 top-k：

- retrieval；
- beam candidates；
- ranking；
- frequent items。

必须熟悉 `heapq` 和 `O(n log k)`。

---

## 6. Graph ★★★★☆

- BFS/DFS；
- topological sort；
- shortest path；
- union find。

Agent workflow / dependency scheduling 也能自然追到 DAG/topological sort。

---

## 7. Dynamic Programming ★★★★☆

必须建立状态定义习惯：

```text
dp[i] 表示什么？
transition？
base case？
iteration order？
space optimization？
```

不要背代码。

---

# Part B｜ML / LLM 手写

## 8. Stable Softmax ★★★★★

见 Transformer 章节。要求能解释 max subtraction。

## 9. Cross Entropy ★★★★★

理解 logits → log-softmax → negative log likelihood。

## 10. SFT Causal Loss ★★★★★

必须会 shift-right + ignore_index。

## 11. Multi-Head Attention ★★★★★

必须会 shape/mask。

## 12. InfoNCE ★★★★☆

```python
scores = q @ d.T / temperature
labels = torch.arange(B, device=q.device)
loss = F.cross_entropy(scores, labels)
```

前提：batch 对角是正样本。

## 13. LoRA 核心 ★★★★☆

手写一个线性 LoRA：

```text
W' x = W x + scale * B(Ax)
```

理解 A/B shape、rank、初始化，而不是只调用 PEFT。

---

# Part C｜训练 Coding

## 14. Qwen SFT 面试写什么？ ★★★★★

不需要从零实现 Transformer。面试通常期待：

1. tokenizer/chat template；
2. dataset map；
3. input_ids/attention_mask/labels；
4. prompt mask；
5. model forward；
6. loss/backward/optimizer；
7. gradient accumulation；
8. eval/save。

### 最小训练循环骨架

```python
model.train()
for batch in loader:
    out = model(**batch)
    loss = out.loss / grad_acc_steps
    loss.backward()
    if step % grad_acc_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

面试还会追：mixed precision、OOM、DDP/FSDP/DeepSpeed。

---

# Part D｜AI Coding / 业务算法

## 15. 2026-08-18 美团搜索排序 AI Coding

公开题信息显示：训练数据包含 query、商品标题、品牌、价格、评分、点击/加购等，评价 NDCG@10。

这类题不是传统 LeetCode，而是在考：

- feature/data inspection；
- group-aware ranking；
- leakage；
- train/validation split；
- ranking metric；
- baseline；
- robust submission pipeline。

### 第一反应不应该是“上大模型”

先：

1. 确认 label；
2. 按 query group split；
3. 建 baseline；
4. 检查 category/numeric/text features；
5. 选择 pointwise/pairwise/listwise；
6. NDCG@10 offline validation；
7. error analysis。

---

## 16. NDCG 是什么 ★★★★☆

DCG 对排名靠前的高 relevance item 给更大权重，并对位置做 log discount。

NDCG 用 ideal DCG 归一化：

```text
NDCG@K = DCG@K / IDCG@K
```

搜索/推荐面试非常常见。

---

## 17. HAC 聚类题

美团 2026-05-16 算法机考出现 single-link hierarchical agglomerative clustering。

核心：

- 初始每点一簇；
- 距离 = 两簇最近点距离；
- 每次合并最小；
- tie-break；
- 到 C 簇停止。

朴素可很慢。面试/机考要按 n 约束选择：

- 预计算 point distance；
- priority queue；
- union/cluster membership；
- stale heap entry 处理。

---

# Part E｜代码面试流程

## 18. 先说再写 ★★★★★

顺序：

1. clarify input/output；
2. example；
3. brute force；
4. optimize；
5. complexity；
6. code；
7. edge cases；
8. test。

大模型岗也会因此判断工程沟通能力。

---

# Part F｜必须主动测试的边界

- empty；
- one element；
- duplicates；
- negative；
- very large；
- sorted/reverse sorted；
- all same；
- overflow；
- Unicode/text；
- NaN/Inf（ML code）。

---

# Part G｜学习清单

## S 级 DSA ★★★★★

- hash
- two pointers
- sliding window
- binary search
- linked list
- tree DFS/BFS
- stack
- heap/top-k
- prefix sum

## A 级 DSA ★★★★☆

- graph
- topological sort
- union find
- DP
- trie
- interval

## S 级 LLM Coding ★★★★★

- softmax
- CE loss
- SFT shift
- MHA
- KV Cache
- InfoNCE
- Qwen/HF SFT pipeline

## A 级工程 Coding ★★★★☆

- LoRA
- RRF
- simple vector retrieval
- metric computation
- async tool executor
- retry/idempotency

---

# 本章验收

- [ ] Hot100 核心类型能在 20-30 分钟内稳定写。
- [ ] MHA/SFT loss 不依赖复制代码。
- [ ] 能写一个最小 Transformers SFT。
- [ ] 能实现 NDCG/Recall/MRR。
- [ ] AI Coding 先做数据/metric/baseline，而不是盲目调模型。
