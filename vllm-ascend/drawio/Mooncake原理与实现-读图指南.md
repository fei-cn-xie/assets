# Mooncake 原理与实现 · 读图指南（入门版）

配套文件：`Mooncake原理与实现.drawio`（10 页）。
读者定位：**会 Python 基础、学过 attention，但没读过推理框架的大一新生**。
每页的编排都是同一条：**一句话结论 → 大白话类比 → 才落到代码锚点**。

> 锚点按本仓**当前检出分支**写（`vllm` / `vllm-ascend` 均为 `fix/lwd-multidp-dense-pd`）。
> 若切回 `feat/pd-sep-lwd-v0.30.0`，行号可能有偏移，用每页给的 grep 重新定位。

---

## 1. 页面清单与建议顺序

| 页 | 讲什么 | 一句话结论 |
| --- | --- | --- |
| ① | 为什么需要搬 KV | K/V 算一次就不变 → 存起来 → 占显存 → 分机器 → 必须跨机搬 |
| ② | Mooncake 是什么 | = 只会搬字节的传输引擎（在用） + 可选的 KV 大仓库（不用） |
| ③ | KV 在显存里长什么样 | 显存按「块」切，块表记「逻辑块 → 物理块」，所以按块搬 |
| ④ | 传输引擎三步曲 | 注册内存 → 认路（session） → 下单（三张平行数组） |
| ⑤ | vLLM 的钩子表 | 固定时刻调固定方法；三个 layerwise 钩子在本项目是 pass |
| ⑥ | connector 分层结构 | 门面 → 调度侧 / worker 侧 → 后台线程 → 传输引擎 |
| ⑦ | 一次交接的完整时序 | P 钉住 → D 登记派活 → D 主动读 → 回执解钉 |
| ⑧ | 什么时候算成功 | 块 / 请求 / 链路三层判定；失败兜底四类 |
| ⑨ | 五个误解 + 性能账 | D 拉不推、整请求批量、点对点不进池 |
| ⑩ | 术语表 + 阅读路径 | 卡词查表；按目的选路径 A/B/C |

三条阅读路径（在 ⑩ 里也画了）：

- **只想懂原理**：① → ② → ③ → ⑦
- **想改 connector**：⑤ → ⑥ → ⑦ → ⑧（先读 `base.py` 的钩子 docstring）
- **只想查/调性能**：④ → ⑨ → ⑦

---

## 2. 一次交接的四段（背下这四段就够了）

1. **P 侧算完 + 钉住**：`request_finished` 返回 `(delay_free=True, kv_transfer_params{块号, IP, 端口, 规模})`
   —— 块先不回收，等 D 来拿。（`mooncake_connector.py:2016,2051-2078`）
2. **D 侧登记 + 派活**：`get_num_new_matched_tokens → (n, True)`（异步，不阻塞调度）→
   `update_state_after_alloc` 记本地块 → `build_connector_meta` 打包 → `start_load_kv` 入队。
3. **D 主动读（pull）**：先 `GET_META_MSG` 换回 P 的地址表，再算
   `src / dst / length` 三个平行数组，一次 `batch_transfer_sync_read` 搬完整请求。
   **P 从不 push**——这是本套实现最容易被误解的一点。
4. **回执与释放**：D 发 `DONE_RECVING`（**成功失败都发**，不发就漏显存）→ P 累计回执到齐才回
   `ACK` → `get_finished()` 报给核心 → P 的块才真正释放；480 s 不来拿则强制释放并打 error。

---

## 3. 常数与端口速查

| 项 | 值 | 出处 |
| --- | --- | --- |
| 控制面基端口 | P `14570`、D `14580`（`kv_port`） | `cloud_p.sh:37` / `cloud_d.sh:33` |
| 每设备握手端口 | `kv_port + dp_index*tp*pp*pcp + device_index` | `mooncake_connector.py:2196-2204` |
| 单次批量 | 该请求**所有层 × 所有块** | `:1042`（一次 `batch_transfer_sync_read`） |
| 搬运线程池 | 32 线程；每对端每轮最多 5 个请求 | `:97, :487-495` |
| 显存注册段上限 | 256 段（相邻 4 KiB 内先合并） | `utils/utils.py:15,16` |
| 对齐要求 | 某些路径要求起始地址 **2 MiB** 对齐 | `:2556, :2610` |
| 对端不来拿的兜底 | 480 s 强制释放 | `vllm/envs.py:260`、`:232-253` |
| 后台线程 ready 超时 | 5 分钟 | `:2760-2765` |
| 传输协议 | `initialize(ip, "P2PHANDSHAKE", "ascend", device)`——**不是** rdma/tcp | `utils/mooncake_transfer_engine.py:24-26` |

---

## 4. 六个容易踩的坑（都在图里标了）

1. **箭头方向 ≠ 数据方向**：P 侧线程只回元数据和 ACK，KV 数据是 D 端 `batch_transfer_sync_read` 读过去的。
2. **没有 `KV_READY` 这个字面量**（在 `mooncake_connector.py` 里）：完成判定是 `DONE_RECVING` + `ACK` + 计数器；
   `KV_READY` 是 **LWD 调度层**的说法（`lwd_control/pd/cloud_scheduler.py:121-130`）。
3. **没有 `pin` API**：「钉住」= `request_finished` 返回的 `delay_free_blocks=True`，由 vLLM 核心延迟释放。
4. **不是逐层的**：三个 layerwise 钩子是 `pass`；整请求块级批量。逐层版在
   `mooncake_layerwise_connector.py`（另一套）。
5. **点对点，不进池**：KV 不写进任何公共存储；Store 模式是另一个连接器
   （`AscendStoreConnector`），本 PD 场景不用。
6. **`up_buffer_size` / `down_buffer_size` 与 KV 无关**：那是 host 数据面（UCX 帧块）的参数。

补充一条部署事实：**edge 不参与 Mooncake**——`edge.sh` 里没有 `--kv-transfer-config`，
只有云端 P/D 两侧传 `kv_connector: MooncakeConnectorV1`。

---

## 5. 自查与重生成

```bash
cd assets/vllm-ascend/drawio/tools
python mooncake/build.py                 # 重建 ../Mooncake原理与实现.drawio
python checkdrawio.py ../Mooncake原理与实现.drawio
python render_check.py ../Mooncake原理与实现.drawio -o /tmp/r --scale 0.8
python color_audit.py ../Mooncake原理与实现.drawio      # 配色 + 图例 + 文案↔颜色一致性
```

配色遵守同目录 `README.md` 的规范（沙色=中立/通用，蓝=边端，橙=云 P，绿=云 D，黄=数据面，
红=失败；**不允许白色填充**）。本套图里：**橙=P 侧步骤，绿=D 侧步骤，黄=真正搬数据的通道**。

---

## 6. 图的来历（便于复核）

三份走读并行产出，全部要求带 `文件:行号`、不确定就写 `(未确认)`：

1. `mooncake_connector.py` 内部机制（类/线程/块级 pull/pin/回执/失败路径/常量）；
2. 本仓对 Mooncake（Transfer Engine / Store）的**使用面**（初始化参数、注册、传输 API、session）；
3. vLLM 侧 **connector 契约**与 LWD PD 编排（钩子表、`KVTransferConfig` 实际取值、谁等 KV、谁判超时）。

三份走读一致纠正了以下预设（写进图里，避免以讹传讹）：
`KV_READY` / `pin` 在 connector 里不存在；P 不 push；不是逐层；LWD PD 是 **fail-closed**
（`recompute_kv_load_failures=False`，导入失败直接终止，不重算也不换机器）。
