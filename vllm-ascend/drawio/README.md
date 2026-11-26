# drawio · 画图规范与工具链（给以后的会话用）

本目录存放 **vllm / vllm-ascend 相关的端到端流程图**，以及生成、校验这些图的工具。
目标是让下一个会话（人或 AI）不必重新踩坑：照着这里的规范与脚手架，直接产出**能打开、
能读、代码锚点可追溯**的多页 `.drawio`。

| 文件 | 作用 |
| --- | --- |
| `vLLM生命周期-端到端.drawio` | 9 页成品图（部署/启动/模型加载×2/KV cache/PD·KV/单请求/控制面/Step 循环） |
| `vLLM生命周期-读图指南.md` | 那 9 页的阅读顺序、三分侧对照表、常数速查、易踩的坑 |
| `README.md`（本文件） | 绘制规范 + 工具用法 + 新图 SOP |
| `tools/` | 画布引擎、泳道引擎、格式器、两个校验器 |
| `tools/pages/` | 那 9 页的**页面源码**（`p1.py … p9.py` + `build.py`） |

---

## 1. 三条纪律（先记住这三条，其余的都能查）

1. **一个颜色只有一个含义。** 填色＝角色/归属（白=中立，蓝=边端，橙=云 P，绿=云 D，
   青=PD Proxy，浅灰=客户端，红=**只**表示失败/取消/超时）；线条＝平面（紫=控制面 ZMQ，
   黄=数据面 UCX·Mooncake，灰=HTTP/调用）。阶段分组、分叉点**不占用颜色**（用带编号的
   灰带 / 虚线框表达）。每页底部都必须有同一个 `legend_bar()`。
   **"按进程切泳道"的页面（P02/P05/P09）用灰阶梯度 `proc1…proc5`**——灰阶不是颜色，
   不会和角色色冲突；这些页的图例额外多一项说明"灰阶 = 进程泳道 / 分列"。
   `legend_bar()` 内部会临时清空该页的 `fill_map`，**保证图例在任何页面上都是规范色**。
2. **每个节点都要带 `文件:行号`。** 图是给人顺着源码读的，没有锚点的框等于没写。
   锚点用 9px 灰字（`<font style="font-size:9px" color="#666666">`）。
3. **交付前必须过三道校验**（见 §4）。只看"图能打开"不够：缺 `parent="1"` 时
   文件能打开、页数正确，但整页几乎全空——这个坑已经踩过一次。

---

## 2. 版式：先选骨架，再填内容

| 版式 | 用在哪 | 工具入口 | 成品页 |
| --- | --- | --- | --- |
| **泳道时序** | 跨角色/跨进程的时间线（启动、请求、KV 交接） | `seq.Seq` | P02/P03/P05/P06/P07 |
| **分组调用链** | 一条自上而下的调用栈 + 右栏注释 | `tools/pages/p4.py` 的 `group()` | P04 |
| **部署/自由布局** | 谁在哪台机、几个进程、几条通道 | `page.box/edge/table` | P01 |
| **表格 + 状态机** | 消息清单 + 若干个对象的状态迁移 | `dg.table()` + `p8._sm()` | P08 |
| **多列对照** | 通用机制 vs LWD 改动 | `page.box` 三列 | P09 |

泳道时序写法（最常用）：

```python
from dg import Doc, legend_bar
from seq import Seq

p = doc.page("10 新的一页", pw=2200)          # 页面尺寸后续会自动贴合内容
s = Seq(p, lanes=[("cl", "客户端", "client"),
                  ("e",  "边端", "edge"),
                  ("c",  "云 P", "cloudp")],
        lane_w=340, title="⑩ · 标题", subtitle="一句话说明这张图回答什么")

s.phase("① 阶段名")                           # 横跨所有泳道的灰色阶段带
s.msg("cl", "e", "HTTP 请求", mode="sync", role="http")
s.step("e", "做什么", "文件/路径.py:行号", "补充说明（棕色小字）", role="edge")
s.steps([("e", "左", "a.py:1", None, "edge", "box", True, None),
         ("c", "右", "b.py:2", None, "cloudp", "box", True, None)])   # 同一时刻多个泳道
s.note(("e", "c"), "跨泳道的解释文字", role="neutral")                # 便签
s.gap(18)
legend_bar(p, s._layout_rows(dry=True) + 44)   # 统一图例，按实际内容底部定位
```

`Seq` 会自动算高度并让页面贴合内容；**不要手写 x/y 去摆时序图的框**。

---

## 3. draw.io 文件格式的硬约束（都是踩过的）

| 约束 | 症状 | 本工具链怎么保证 |
| --- | --- | --- |
| 每个 cell 都要 `parent="1"` | 文件能打开、页数对，但**整页几乎空白** | `dg.py` 的 `_cell_xml()` 强制写 parent |
| `id` 页内唯一，不要用 `0`/`1` | 后者是 root 与 layer | `Page.nid()` 自增 |
| vertex 必须有 `<mxGeometry x y width height/>` | 不出图 | `Page.box()` 统一生成 |
| 边必须有 `source/target` 或 `sourcePoint/targetPoint` | 悬空箭头 / 不画 | `Page.edge()`；校验器会断言 |
| 中文必须在 `<font>` 里用 9px，不要塞进标题 | 标题一长就挤掉锚点 | `box_for()` 自动拼 label |
| 文件要给 `<?xml version="1.0" encoding="UTF-8"?>`、UTF-8 无 BOM | 个别工具按 ANSI 读会乱码 | `format.normalized()` 补声明 |
| **一行 300KB 的 XML 不可 review** | diff 全是 `+` 一整行 | `format.normalized()`：一 tag 一行 |
| draw.io 桌面版会留 `.$xxx.dtmp`、`.bkp` | `git status` 噪声 | `.gitignore` 里屏蔽 |

### 与 next-ai-drawio MCP 的分工

- 预览/截图/导出 PNG 用它：`start_session` → `load_diagram` → `screenshot_diagram` → `export_diagram`。
- **MCP 导出的文件会被重新序列化**：属性变字典序、cell 顺序可能变化（会改变 z-order！）、
  写成单行。所以：**只用它看图，不用它定稿**；定稿走 `pages/build.py`。
- 预览 URL 里的 session id **必须带 `mcp-` 前缀**，否则页面报 `Invalid session id`。
- MCP 的 `load_diagram(path=…)` 会替换整个文档，手改内容前先 `get_diagram` 备份。

---

## 4. 三道校验（缺一不可）

```bash
cd assets/vllm-ascend/drawio/tools
PY=python            # 需要 Pillow；字体用 C:\Windows\Fonts\msyh.ttc，缺了会退化

# 0) 生成（同时跑版式断言：盒子重叠 / 边穿框 / 超出页面）
$PY pages/build.py                       # 写回 ../vLLM生命周期-端到端.drawio

# 1) 结构层：像不像 draw.io 自己存的文件
$PY checkdrawio.py ../vLLM生命周期-端到端.drawio

# 2) 语义层：反解 .drawio 自己渲染 PNG/SVG，并断言
#    不可见 cell / 文字溢出 / 边端点悬空
$PY render_check.py ../vLLM生命周期-端到端.drawio -o /tmp/render --scale 0.8 --svg
```

三道都过 = 文件几何、可见性、文字都正确。**要判断视觉效果，最后再用 MCP 截图看 1~2 页**
（`screenshot_diagram`），不要靠猜。

`render_check.py` 的两个设计要点：

- 它**从 `.drawio` 文件反解**再画，不是读生成器的内存模型——所以能抓"文件写坏了"这类问题；
- 溢出的判定**在真实尺寸（scale=1）下算**，只把绘制缩放，所以 `--scale` 不影响结论。

---

## 5. 工具 API 速查

`dg.py`

| 东西 | 说明 |
| --- | --- |
| `C` / `LEGEND_ITEMS` / `LEGEND_LINES` | 唯一配色表与图例文案（改颜色只改这里） |
| `Doc()`、`doc.page(name, pw, ph)` | 多页文档；`doc.xml()` → 完整 mxfile |
| `Page.box/edge/text` | 图元；`extra_style=` 可追加 `dashed=1;exitX=…` 等 |
| `Page.fill_map` | **页内别名映射**：把本页的临时角色名映射到规范色（P02/P09 用它把"进程"类方框统一成中立） |
| `box_for(page, x, y, w, title, anchor, extra)` | 自动按字体度量算高度并拼 label |
| `measure_h(title, anchor, extra, w)` | 与 `render_check.py` **同一套字体度量**，保证两边不打架 |
| `table(page, x, y, cols, rows)` | 简易表格（首行表头） |
| `legend_bar(page, y, extra_fill=, extra_line=)` | 全文件统一图例，**每页都要调** |
| `validate(page)` | 版式断言（重叠 / 穿框 / 越界），`build.py` 自动跑 |

`seq.py`

| 东西 | 说明 |
| --- | --- |
| `Seq(page, lanes, lane_w=, title=, subtitle=)` | 泳道时序；`lanes=[(key, 标签, role)]` |
| `s.phase / s.step / s.steps / s.msg / s.selfmsg / s.note / s.gap` | 行构造器（按调用顺序自上而下） |
| `s._layout_rows(dry=True)` | 取内容底部 y，用来放图例（不产生 cell） |
| `chain(...)` / `aligned(...)` | 垂直调用链 / 双列对齐（P04 这类页面可直接用） |

---

## 6. 新增一张图的 SOP

1. **先定版式**（§2 的表），再动手；跨角色时间线一律泳道时序。
2. 复制一页源码（`p3.py` 之类）→ 改名 → 在 `pages/build.py` 的 `PAGES` 里登记。
3. 内容从代码来：`grep -n` 找锚点，**每个框至少一个 `文件:行号`**；不确定的写 `(未确认)`，
   不要编。
4. 跑 §4 三道校验，红了就改到绿。
5. 用 MCP 截图看 1~2 页（排版目视），必要时只调 `lane_w` / `gap` / 文案长度。
6. 更新 `vLLM生命周期-读图指南.md` 的页面清单（或新建一份同名 `*-读图指南.md`）。
7. 提交：`docs: <一句话>`（单行、无 body），然后推送（`origin` 配了双推地址，推之前确认）。

**不要**用生成器去覆盖一张已经被手工改过的 `.drawio`：先 `git log -p` 看最后是生成器还是
手改，再决定"重建"还是"接着手改"。

---

## 7. 下一批图的建议素材（省得重新找入口）

| 想画的图 | 先读这些 |
| --- | --- |
| **vLLM 架构总览**（进程/线程/引擎分层） | `vllm/v1/engine/core.py`（EngineCore / EngineCoreProc）、`vllm/v1/executor/*`、`vllm/v1/engine/core_client.py`、`vllm/distributed/parallel_state.py` |
| **MTP / 投机解码实现原理** | `vllm/config/speculative.py`、`vllm/v1/spec_decode/*`；昇腾侧 `vllm_ascend/worker/lwd_cloud_model/lwd_mtp_proposer.py`、`vllm_ascend/spec_decode/*`；`--speculative-config` 三份启动脚本 |
| **Mooncake KV 传输原理** | `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`（4300+ 行，块级 pull，非 layerwise）、`vllm/distributed/kv_transfer/kv_connector/v1/base.py` 的钩子契约、`cloud_p.sh/cloud_d.sh` 的 `--kv-transfer-config` |
| **PD 分离数据面（UCX/UHC1 帧）** | `vllm/v1/lwd_comm/data_comm/{ucx_channel,ucx_link,ucx_pool,ucx_daemon,ucx_shm}.py`、`vllm_ascend/distributed/lwd_data_plane/host/{ucx_io,lwd_host_channels}.py` |
| **前缀缓存 / hash 路由** | `vllm/v1/core/kv_cache_utils.py`、`vllm/v1/lwd_control/control_cloud_scheduler/lwd_prefix_cache.py`、`lwd_hash/lwd_hash_routing.py` |

需要跨会话的深度走读笔记时，可以让子代理按"每个结论带 `文件:行号`、末尾给
`## 建议流程图节点清单`"的模板产出——本套 9 页就是这么来的（5 个子代理并行走读）。

---

## 8. 已知限制

- 工具只依赖 Pillow 与 Windows 字体；在 Linux 上能跑，但中文字体缺失时高度估算会回退到
  字符宽度近似（`dg._char_w`），此时**必须**把 `render_check.py` 的溢出告警当准。
- `render_check.py` 是"几何忠实"的近似渲染：不实现 draw.io 的正交路由、箭头样式细节，
  所以它能判"有没有、在哪、放不放得下"，**判不了线条绕得漂不漂亮**——那要靠 MCP 截图。
- 页面里所有坐标都是算出来的；手工在 draw.io 里挪动后，生成器再跑就会覆盖。
