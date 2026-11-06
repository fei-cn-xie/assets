# LWD PD 分离：代码层「不合理耗时点」审计

> 前提：**不讨论拉起脚本/配置**。本文只从代码路径上找"这一步为什么非要花这么久"，并尽量用日志里现成的打点做量化。
> 标注约定：**[实测]** = 日志里直接读到的数；**[推算]** = 由实测数交叉推导；**[待 profile]** = 代码上可疑但现有日志无法定量。

---

## 0. 先把"时间花在哪"的账建起来

三份日志里其实已经埋了不少 per-step 打点，把它们拼成每步预算表：

| 打点位置 | 含义 | 实测值 |
|---|---|---|
| `lwd_cloud_worker.py:802` `cloud-step sample= send= total=` | decode 每步：采样 + DOWN 提交 | **sample p50 = 21.15 ms**、send p50 = 0.48 ms **[实测]** |
| `lwd_cloud_model_runner.py:1248` `collect rank= pinned= total=` | hidden D2H 采集 | 1.28 ms（rank 1.09 + pinned 0.19）**[实测]** |
| `lwd_cloud_model_runner.py:554` `up-recv-wait dur=` | decode 等 UP 窗口输入 | avg 0.92 ms / max 52.3 ms **[实测]** |
| `lwd_edge_worker.py:353` `embed forward= submit_send=` | 边侧单个 chunk 的 embed + 发送 | forward 0.18 ms、submit 0.26~0.30 ms **[实测]** |
| `lwd_cloud_engine.py:324` `poll_gap_ms=` | prefill 单 chunk 步时 | p50 460.3 ms（n=413）**[实测]** |

关键交叉验证 **[实测]**：

```
并发 ≈ 1  时  sample = 20.3 ms
并发 = 32  时  sample = 20.4 ms
```

**采样耗时与批量完全无关。** 这说明这 21 ms 不是算力，而是**固定的 host 侧开销** —— 这是本文最重要的一条线索。

---

## 1. decode 每步固定 21 ms 的 `sample_tokens`（单点最大）

### 1.1 实测分布

```
n = 2068   min = 17.71 ms   p50 = 21.15 ms   p90 = 22.05 ms   max = 107.55 ms
```

（min 在并发 1 时也成立，max 的 107.55 是图 capture/warmup 的头三步）

### 1.2 这 21 ms 花在哪：LWD 自己只占 1.3 ms

```python
# vllm-ascend/vllm_ascend/worker/lwd_cloud/lwd_cloud_model_runner.py:994-1025
@torch.inference_mode()
def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
    ...
    captured = (state[5], state[1], state[2], state[0],
                list(self.input_batch.req_ids),
                self.discard_request_mask.np.copy())      # ← 每步 numpy 拷贝 + 列表拷贝
    self._lwd_captured_sampler_output = None
    output = super().sample_tokens(grammar_output)         # ← 这 19.8 ms 全在这里
    if (...):
        self._lwd_pending_down_payload = self._lwd_collect_down_payload(...)   # ← 1.28 ms
    return output
```

`collect total=1.28ms` 是 LWD 自己的 DOWN 采集；**剩下约 19.8 ms 全在基座 `NPUModelRunner.sample_tokens()` 里**。

### 1.3 为什么说它"不合理"

- 按 vLLM 常规量级，128 个 token 的采样 host 开销在 **1~3 ms**；这里是 21 ms，**高了 7~20 倍**；
- 且它与批量无关 —— 如果是算子/带宽成本，32 个请求一定比 1 个慢；恒定就意味着**固定次数的、和批量无关的 host 往返**。

### 1.4 代码上的可疑点（按嫌疑排序）

**(a) MTP 逐位置拒绝采样没有融合（嫌疑最大）**

```python
# vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2622-2651
def _sample(self, logits, spec_decode_metadata):
    ...
    if spec_decode_metadata is None:
        return self.sampler(...)
    ...
    sampler_output = self.rejection_sampler(spec_decode_metadata, None, logits, sampling_metadata)
    return sampler_output
```

`num_speculative_tokens=3` → 每步要跑 1+3 个位置的接受/拒绝。若 rejection sampler 内部按位置做 **Python 循环 + 每轮一次 D2H/同步**，就是 4 次固定往返 —— 与批量无关，与观测完全吻合。**[待 profile]**

**(b) `_bookkeeping_sync` 里的隐式同步**

```python
# vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2652+
def _bookkeeping_sync(self, ...):
    discard_sampled_tokens_req_indices = self.discard_request_indices.np[: self.num_discarded_requests]
    for i in discard_sampled_tokens_req_indices:        # ← 逐请求 Python 循环
        gen = self.input_batch.generators.get(int(i))
        ...
    req_ids_output_copy = self.input_batch.req_ids.copy()
```

`_to_list()` 内有 `event.synchronize()`（注释里明确写了 "``_to_list`` does ``event.synchronize()``"）。这个同步是**每步一次、与批量无关的固定成本**。**[待 profile]**

**(c) LWD 自己加的每步固定拷贝**

```python
lwd_cloud_model_runner.py:1012   self.discard_request_mask.np.copy()
lwd_cloud_model_runner.py:1011   list(self.input_batch.req_ids)
```

小，但属于"每步都要交的税"，且注释说明它是为了并发正确性（batch queue 重叠时防止读错批序）。更好的做法是**只拷脏区间**或换成 ring buffer。**[推算 ~0.1 ms]**

### 1.5 对 TPOT 的影响

```
TPOT = 步时 / 3.46
```

- 6 并发：步时 64 ms，其中采样 21 ms → **占 33%**
- 22 并发：步时 102 ms，采样 21 ms → 占 21%
- **即便把 ACL 图修好、前向压到 20 ms，这一步也会被 21 ms 采样顶住** → TPOT 硬下限 ≈ `21 / 3.46 = 6.1 ms`

所以这条不是"可选优化"，而是**通往低 TPOT 的必经之路**。

---

## 2. 边侧 token 回程的串行阻塞（交付链路上的结构性缺陷）

### 2.1 收割在派发之前，且"弹空全队列再逐个处理"

```python
# vllm/vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_engine.py:798-833
while self._lwd_batch_queue:                       # ① 先全弹空
    kind, payload, t_dispatch, future = self._lwd_batch_queue.popleft()
    if kind == "unembed":
        self._lwd_deliver_unembed(payload, future, outputs, finished_reqs)
    elif kind == "pd_tail":
        self.scheduler.lwd_pd_tail_harvested(payload)
    else:
        self.scheduler.lwd_edge_update_progress(dict(payload.num_scheduled_tokens))   # ② 不等 future
    ...
d_emb = 0
while (len(self._lwd_batch_queue) < LWD_EDGE_BATCH_QUEUE_DEPTH   # ③ 之后才派发
       and self.scheduler.has_requests()):
```

- 收割（①）在派发（③）**之前** → **一个慢 unembed 会把本步的 embed 派发整段挡住**；
- 而 embed（UP，喂 P 段）和 unembed（DOWN 回程，出 token）**共用同一条队列和同一个深度预算** `LWD_EDGE_BATCH_QUEUE_DEPTH = 4`（`lwd_edge_engine.py:94`）→ **P 段喂料与 D 段 token 交付互相踩（head-of-line blocking）**。

### 2.2 unembed 同步等 NPU 前向，embed 却"假装"完成了

```python
# lwd_edge_engine.py:1039-1056
def _lwd_deliver_unembed(self, notify, future, outputs, finished_reqs):
    if future is None:
        ...
    else:
        result = future.result()        # ← 同步阻塞等 worker 的 NPU 前向
```

对照 embed 分支只调 `lwd_edge_update_progress`、**从不等 future**。

**语义不一致的代价是双向的**：
- unembed 该异步的却串行了（token 交付被一次完整 NPU 前向卡住）；
- embed 该等的却没等（进度乐观推进 → chunk 静默丢失时边侧毫无感知，这正是故障分析里 413/414 丢批不可见的直接原因）。

**正确做法**：两条独立队列；unembed 只登记 future、下一步收割（天然流水化）；embed 的 future 至少要挂 `done_callback` 收失败。

### 2.3 现有日志没能定量（且为什么）

`edge harvest kind=unembed seqno= wait=Xms` 只在日志里留下 embed 的样本（unembed 的 3 条被压缩掉了）。
**这块需要补一条打点**：`_lwd_deliver_unembed` 入口/出口各记一次时间，把 `future.result()` 的等待时长单独打出来。

---

## 3. DOWN 发送挂在采样线程的同步路径上，且失败不可见

```python
# vllm-ascend/vllm_ascend/worker/lwd_cloud/lwd_cloud_worker.py:734-780
@torch.inference_mode()
def sample_tokens(self, grammar_output):
    output = self.model_runner.sample_tokens(grammar_output)     # 21 ms
    ...
    for (edge_id, hidden, meta_dev, seg_lens_host, ev, req_ids) in payloads:
        seqno = self._lwd_next_down_seqno(edge_id)
        if lwd_wire.is_lwd_channel_endpoint(...):                 # ← 每步每 rank 的 Python 判定
            logger.info("[Lwd][cloud-worker] DOWN send ...")       # ← 每步一条 INFO
            get_lwd_comm_service().submit_send(LwdCommRequest(...))   # ← 返回值直接丢弃
```

三个问题：

1. **`submit_send` 是同步调用，且可能长时间阻塞**：
   ```python
   # vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py:436-448
   limit = self.runtime.limits.pending_per_channel          # = 8
   if len(self._entries) < limit:
       break
   self._capacity.wait(remaining)                            # remaining 来自 operation_timeout_s = 120s
   ```
   **最长可在 decode 的热路径上阻塞 120 s**。实测 `send=0.48ms` 是常态，但这颗雷一直在。
2. **future 被丢弃** → 发送失败静默（与故障分析里"两条批静默丢失"同源）。
3. **每步一条 INFO + 每个 rank 一次端点判定**。

**建议**：`submit_send` 换成无锁队列入队（后台线程发），future 挂 `done_callback` 打 ERROR；端点判定结果按 (edge_id, engine_index) 缓存。

---

## 4. 数据面每批的额外整形

### 4.1 payload scramble：每批两个设备算子 + 一次事件重记

```python
# sidecar_channel.py:152-160（发送侧）
seed = _scramble_seed_for(request)
... get_payload_scrambler().scramble(owned, seed=seed)      # view(int16).bitwise_xor_()
# sidecar_channel.py:330-342（接收侧）
... get_payload_scrambler().descramble(tensor, seed=seed)   # 而且"完成事件在 descramble 之后才重记"
```

实现本身是设备侧 in-place XOR（`payload_scramble.py:185`），量级不大；但它是**每批固定两次**，UP 每批 1536×5120×2B = **15.7 MB**、DOWN 每批 128×5120×2B = 1.31 MB。属于可省的纯开销（若安全域允许按配置关闭）。

### 4.2 DOWN 固定按 padded 128 行采集

```
collect edges=1 rows=128 ...        # 实测
```

而实际产出的 token 数是 `32 × 3.46 = 111` → **多采 15%**，边侧 lm_head 也多算 15%。padding 是 spec-decode 的常规手法（凑 4 的倍数），但这里凑到了 128 而不是 112，属于采集侧没有按 `n_seqs × query_len` 精确取整。

---

## 5. 提前收（pre_recv）被 1 ms 睡眠 + 配额 4 卡成串行

```python
# vllm-ascend/vllm_ascend/worker/lwd_cloud/lwd_cloud_worker.py:317-318
self._early_recv_max_inflight = int(os.environ.get("VLLM_ASCEND_LWD_EARLY_RECV_MAX_INFLIGHT", "4"))

# :511-514  单边时 = 4
def _early_recv_quota(self, edge_id):
    e_live = max(1, len(self._lwd_early_recv_edges))
    return max(1, -(-self._early_recv_max_inflight // e_live))

# :427-429  通信线程每轮睡 1 ms
time.sleep(0.001)
```

三个叠加效应：

1. **配额 4 vs 一个 prompt 要 3 个 chunk** → 同时只装得下 **1.33 个请求**的提前收；
2. 而目录里配置的是 `rx_slots: 8`（`lwd_sidecar_directory.json:42`）→ **配置允许 8 个在飞，代码只放 4**，一半配额白给；
3. `time.sleep(0.001)` 让水位推进带 1 ms 量化延迟，配额满时提示进 FIFO（`_early_recv_pending`），只能按到达序补挂 —— **单请求的 3 个 chunk 无法一次性挂满**。

日志证据 **[实测]**：

```
[Lwd][pre-recv] queue stuck: oldest queued (edge,seqno)=(0, 258),
4 slot(s) held by (0,254)..(0,257)   (downstream dispatch is the pacer)
```

**建议**：配额提到 `min(rx_slots, max_inflight)`；按"请求的 chunk 数"整组配额（要么 3 个一起挂、要么不挂）；`sleep(0.001)` 换 `Condition.wait(timeout)`。

---

## 6. 每步 O(n) 的 Python 遍历与队列重建

| 位置 | 内容 | 量级 |
|---|---|---|
| `lwd_cloud_mixed_scheduler.py:311 / 421 / 452` | `for rid, record in list(self.lwd_pd_d_records.items())` ×3/步 | records 峰值 126 → **378 次 dict 拷贝+遍历/步** |
| `lwd_base_scheduler.py:145-189` | `[req for req in self.running if req not in picked]` 三队列剔除+拼回，O(n·m) | decode 每个 window 步都走 |
| `lwd_cloud_mixed_scheduler.py:545-560` | `token_ids=[[0] * item.num_tokens for item in all_items]` | tail window=1 时很小；`max_window_tokens` 一放大就是每步几万次 int 分配 |

单看都不致命，但它们**和 244 ms 的步时是叠加关系**，是 jitter 的来源之一。改法：三处遍历合并成一次；`list()` 快照改成"脏标记 + 增量"。

---

## 7. 热路径 INFO 日志（空转步的主要 CPU 来源）

| 位置 | 频次 |
|---|---|
| `lwd_edge_engine.py:852` `edge step harvest_emb=...` | **每步 1 条（8 个参数）** —— 边侧空转 680 步/s |
| `lwd_edge_engine.py:815` `edge harvest kind=...` | 每批 1 条 |
| `lwd_cloud_worker.py:765` `DOWN send ...` | 每步 1 条 |
| `lwd_cloud_worker.py:802` `cloud-step ...` | 每步 1 条（建议改 10 s 聚合） |
| `lwd_cloud_engine.py:324 / 344` | 每步（gap 超阈值）/ 每 200 次（可接受） |

边侧空转步只有 **1.5~2.3 ms**（680 步/s），而一条带 8 个参数的 INFO 要走 `logging` 全局锁 + `%` 格式化 + 写盘，按 10~40 μs 计 → **占空转步的 5~15%**。

这条对 TTFT/TPOT 的**直接**影响不大，但它是"空转 680 步/s 抢 CPU"的主要来源，而 P/D 与边侧同机时这会间接把对方主循环打抖（prefill `poll_gap_ms` 峰值 **742.1 ms**、decode `poll_gap_ms=228.9 ms` —— 都远超正常步时）。

---

## 8. 汇总：按"可省时间"排序

| # | 耗时点 | 单位成本 | 折算影响 | 性质 |
|---|---|---|---|---|
| **1** | `sample_tokens` 固定 21 ms | 21.15 ms/步 | 6 并发时占步时 **33%**；TPOT 硬下限 6.1 ms | **纯固定开销，与批量无关** |
| **2** | unembed 串行 + 与 embed 共队列（head-of-line） | 待打点 | token 交付与 P 段喂料互相阻塞 | 结构性 |
| **3** | `submit_send` 同步可阻塞 120 s + future 丢弃 | 0.48 ms 常态 / 最坏 120 s | 卡死 + 静默丢批 | 正确性风险 |
| **4** | pre_recv 配额 4（配置给了 8） | 结构性 1.33 请求并发 | 直接放大 TTFT 排队 | 配置未用满 |
| **5** | DOWN padded 128 行 | +15% 传输与 lm_head | 每步 ~0.2 ms | 可精确取整 |
| **6** | scramble/descramble | 2 设备算子/批 | 每批 ~0.1 ms | 可配置关闭 |
| **7** | 每步 O(n) 遍历 ×3 | 378 次 dict 拷贝/步 | jitter | 可合并 |
| **8** | 每步 INFO 日志 | D 侧 3 条/步、edge 680 条/s | CPU 争用 → 主循环打抖 | 可降频 |

---

## 9. 要变成定论还需要补的 4 个打点

现有日志能定位到"哪一段慢"，但下面 4 处只能定性，建议补上（都是几行代码）：

```python
# ① sample_tokens 内部拆分：把 21 ms 拆成 base-sampler / rejection / bookkeeping
t0 = time.monotonic(); sampler_output = self.rejection_sampler(...);  t1 = time.monotonic()
# 打 [Lwd][perf] sample_split base=%.2f rejection=%.2f bookkeeping=%.2f

# ② unembed 收割等待（边侧）
t0 = time.monotonic(); result = future.result(); t1 = time.monotonic()
# 打 [Lwd][perf] unembed_wait=%.2fms rows=%d

# ③ submit_send 的阻塞时长（decode 侧）
t0 = time.monotonic(); fut = ...submit_send(...); t1 = time.monotonic()
# 打 [Lwd][perf] down_submit=%.2fms（超过 5ms 才打）

# ④ pre_recv 每请求的挂收跨度
# 打 (rid, 首 chunk 挂收时刻, 末 chunk 挂收时刻)，验证 §5 的 1.33 请求并发
```

有了这 4 条，§1.4 / §2.3 / §3 / §5 的推断就能从"强相关"升级成"定论"，同时也能直接量出每条修复的实际收益。
