# 请求abort · 需求

## 背景与需求

边云协同（LWD prefill_only）推理把一套大模型切成两段：边侧只跑 embedding 与 unembed / lm_head，云侧跑全部 transformer 层。一次请求的完整生命周期要跨节点往返多次：prefill 阶段边侧把整段 prompt 的 embeddings **一次性**发往云侧（**每个请求恰好占 1 个 UP 传输**，见"关键要素"§1），decode 阶段每步一次"云算完 hidden → 边做 unembed 还原 token"。

请求有**两种终止方式**，在引擎侧汇入同一个 `abort_requests` 入口，本需求对两者统一处理：

1. 用户在推理未结束时主动发起终止（abort）；
2. 客户端断开连接（关页签 / 网络中断），服务端感知后同样终止该请求。

当请求被终止时，服务必须：

1. **立刻停止对该请求的推理**，不再为它调度任何新计算；
2. **停止向用户输出该请求的结果**，不再流式返回任何 token；
3. **释放该请求占用的资源**（KV cache 等）；
4. **保证这次终止不阻塞、不破坏同一服务上其他仍在推理的请求。**

第 4 点是本需求的核心难点：两条数据面通道（UP / DOWN）都是严格按 seqno 顺序配对的 FIFO（HCCL P2P 无 tag），而一个请求在 UP 上占用**恰好 1 个** seqno、在 DOWN 上每步占用 1 个。若请求被终止时，边侧已发出它的 UP 载荷、而云侧尚未消费对应的 Notify，通道上就留下 seqno 空洞；空洞不处理会卡死该通道上它之后的所有请求。

因此本需求的核心是：**当请求被终止时，终止信号必须同时传导到边、云两侧，使两侧一致地排空该请求在两条通道上已占用但不会再完成的传输，并清理两侧簿记——既终止该请求本身，又不影响其他请求。**

## 关键要素

### 1. 两条数据面通道（方向各自独立）

| 通道 | 方向 | 数据面载荷 | 一次传输粒度 |
|---|---|---|---|
| `UP` | 边 → 云 | prompt embeddings（`[N, H]` bf16） | **一个请求的整段 prompt**（单请求批、恰好 1 块） |
| `DOWN` | 云 → 边 | 本步 hidden（`[R, H]` bf16） | 一个 decode 步（可含多个请求的行） |

- 每条通道有**独立的 HCCL communicator 与独立的 NPU stream**；`UP` / `DOWN` 各自维护独立的 seqno 计数器。
- `DOWN` 的每请求行信息（rank / `top_id_ths` / `req_ids` / `finish_reasons` / `down_seqno`）不随数据面，走控制面（`LwdC2eNotify`）。
- **边侧无法分块（关键事实）**：边侧是 0 层拓扑（无 attention 层），`get_kv_cache_spec()` 返回空，引擎初始化因 `kv_cache_groups` 为空而**强制关闭 chunked prefill**（`vllm/v1/engine/core.py`）⇒ 每个请求的整段 prompt 恰好占 1 个 UP seqno、只发一次；`prompt > max_num_batched_tokens` 的请求**不会被边侧接纳**。
- 推论：UP 洞的触发窗口 = "边侧发出载荷"到"云侧 pop 到该请求的 Notify"之间的**一个云侧调度步**——窗口窄但存在；且一旦成洞，卡死的同样是其后**所有请求**（seqno 是全局共享的）。

### 2. 每通道 seqno 与严格 FIFO

- 每条通道 seqno 从 0 起、每次传输递增 1，**两侧必须对"同一 seqno = 同一次传输"达成一致**。
- HCCL P2P **无 tag**，收发配对完全依赖"发送顺序 == 对端接收投递顺序"。
- 接收侧若某 seqno 缺失，其后所有 seqno 都要在**重排缓冲**里被"持有"，直到缺失 seqno 到达（正常收发，或对已终止请求**挂空排空**补齐）；否则形成队头阻塞，卡死整条通道。
- `UP` seqno 由边侧在"发布 `LwdRangeNotify` 成功"时原子分配（peek-then-advance），保证号只分配给真正上 wire 的传输、号连续无空洞；每个请求恰好分配 1 个号（整段 prompt 一次发出，见 §1）。
- `DOWN` seqno 由云侧在**实际发送 DOWN 载荷时**分配（无载荷步不占号），经 `LwdC2eNotify.down_seqno` 回传边侧，边侧据此预挂精确尺寸 recv。

### 3. 控制面协议（四个通知）

| 通知 | 方向 | 内容 | 与终止的关系 |
|---|---|---|---|
| `LwdRequestNotify` | 边 → 云 | 请求元数据（prompt 长度 / 采样参数 / 哈希链） | 准入时发 Notify |
| `LwdRangeNotify` | 边 → 云 | 每个请求整段 prompt 的范围（offset=0 / num_tokens / seqno） | 每请求一条 Notify，seqno 随之分配 |
| `LwdAbortNotify` | 边 → 云 | `request_id` | **终止信号** |
| `LwdC2eNotify` | 云 → 边 | 本步 DOWN 元数据（`hidden_num_elements` / `top_id_ths` / `req_ids` / `finish_reasons` / `down_seqno`） | 携带逐请求 finish 码（含 ABORT） |

### 4. 终止的两个触发源与信号流

- **触发源**：显式 abort、客户端断连，两者都汇入 `abort_requests`。
- **边侧** `abort_requests`：先发 `LwdAbortNotify` 出云，再走原生 `finish_requests(FINISHED_ABORTED)` 做本地清理（移出队列、释放 KV）。
- **云侧**收到 `LwdAbortNotify`：触发原生 abort，把请求移出调度器；后续调度步产生 `finished_req_ids` 随 `SchedulerOutput` 下发云 worker，驱动 collector / embeds 缓冲清理。
- 数据面**没有 FIN 报文**：请求终结完全由控制面宣告。

### 5. 请求生命周期与终止命中点

| 命中点 | 数据面状态 | 处理要点 |
|---|---|---|
| waiting（尚未调度） | 无任何传输 | 普通清理，无 seqno 需要处理 |
| prefill 已发出、云侧未消费 | 整段 prompt 的 embeds 已发（edge 已 send） | 已发出的载荷存在**在途 send**；若云侧尚未消费对应 Notify，必须按原尺寸挂空排空（recv-and-drop） |
| decode 中途 | DOWN 载荷可能已组包 / 已发送 | 边侧按 C2e 预挂 recv；已发载荷需按原尺寸接收 |
| 结果在途（竞态窗口） | 云已发 DOWN、边未 recv | 迟到结果幂等丢弃，不复活请求 |

### 6. 两侧簿记

| 簿记 | 侧 | 终止时需清理的含义 |
|---|---|---|
| `_lwd_awaiting` | 边调度器 | embed 完结、等云结果的台账；abort 时摘除，防迟到结果误认领 |
| collector 注册表 | 云 worker | 请求是否仍参与 DOWN 组包；终止后剔除，否则其行继续被打进 DOWN 载荷 |
| `_lwd_gate_pending`（门池） | 云 engine | 已收 `LwdRequestNotify`、尚未准入的请求；abort 时清除 |
| prompt-embeds 组装缓冲 | 云 worker | prefill 中途终止的兜底释放 |
| 重排缓冲 held 条目 / recv future | 通道 / worker | 乱序持有或已投递未消费的传输，等待方不得挂到超时 |

## 核心难点：终止产生的"洞"

### 洞一（UP 方向）：边已发送、云未接收

边侧"发布 `LwdRangeNotify(seqno=N)`"与"挂 `lwd_batch(seqno=N)` 派发"是同步完成的，即边侧该请求（整段 prompt）的 send **必然已在途**。若此时请求被终止、且 abort 先于该陈旧 Notify 被消费（竞态窗口），云侧 `LwdAbortNotify` 把请求移出调度器，随后 `prefill_notify_queue` 里的陈旧 `LwdRangeNotify(seqno=N)` 被丢弃、不再 post recv——于是 UP 通道在 seqno=N 处留下洞，卡死其后所有请求（它们的 seqno 都排在 N 之后）。

**必须处理**：云侧丢弃陈旧 Notify 时，仍须按原尺寸挂一个空 recv 排空（recv-and-drop）该在途 send，保持 UP 通道配对连续。这是 v0.23.0_lwd 采用的"排空、绝不跳过"思路，也是本需求唯一需要新增数据面处理的点。

### DOWN 方向为什么没有对称的洞（需守住的语义）

DOWN 方向天然排空：边侧**永远按云侧实际发送的完整尺寸 recv**（`recv_num_elements = hidden_num_elements`），被终止请求的迟到结果由 `lwd_edge_deliver_tokens` 幂等丢弃（不在 awaiting → 丢弃告警）。需要守住两个语义前提：

- 云侧把"被终止请求的行"从**后续** DOWN 组包中剔除（collector drop），其终止码经 `LwdC2eNotify.finish_reasons` 回传边侧；
- 一个 DOWN 步里"部分请求已终止、部分健康"时，健康请求的行仍要正常流动，不能因剔除行而错位。

## 注意事项

1. **每个已分配的 seqno 都必须在两侧有配对动作**：同一 seqno 在两侧指向同一次传输，发送方按原尺寸发送、接收方按原尺寸接收（被终止请求的载荷同样按原尺寸收发后丢弃）；任一侧缺收发动作、或两侧尺寸不一致，都是无 tag 通道上的配对错误。
2. **绝不因终止卡死后续请求**：被排空的 seqno 必须能让通道 next seqno 推进，否则一个终止堵死整条通道。
3. **在途载荷不能"收回"**：已投递/已发出的传输只能按原尺寸接收后丢弃（排空）；尚未派发的传输则根本不分配 seqno（peek-then-advance）。因此终止的处理是"排空在途 + 不分配未来号"，没有"跳过已分配号"的路径。
4. **终止与发送的竞态必须兜底**：对端可能先于终止把数据发出（边侧"Notify+派发"同步），本端已排空的 seqno 仍可能收到数据——不得崩溃，应可识别地快速失败而非静默超时。
5. **UP seqno 必须连续**：号只分配给真正上 wire 的传输；被终止请求不再占用任何后续号（每个请求本来就只有 1 个号）。
6. **迟到结果幂等丢弃**：被终止请求的云结果到达时，边侧不得复活该请求、不得误认领（awaiting 摘除 + deliver 校验）。
7. **清理依据控制面信号，无 FIN 报文**：云侧清理依赖 `LwdAbortNotify`（触发 abort）+ `finished_req_ids`（下发 worker），不引入数据面结束标记。
8. **不改变健康请求语义**：未被终止的请求的 seqno 顺序、收发配对、数据内容完全无感。
9. **断连与显式 abort 同路径**：两者在 `abort_requests` 汇合，无需区分处理；断连到达时刻的任意性由同一竞态兜底覆盖。

## 需求边界

| 必须做到 | 不负责 |
|---|---|
| 终止该请求的推理与输出（边云两侧） | 不处理正常 finish 的语义差异（尽量复用同一清理路径） |
| 释放 KV cache 等引擎资源 | 不改变调度 / 抢占 / 投机解码的既有策略 |
| 传导终止到两侧，排空该请求的在途传输 | 不引入新的控制面消息（复用 `LwdAbortNotify` / `finished_req_ids`） |
| 清理两侧簿记并兜底竞态 | 不负责数据面之外的加密 / 传输介质问题 |
| 保证其他请求完全无感 | 不要求终止同步即时生效（允许一个调度步内收敛） |
