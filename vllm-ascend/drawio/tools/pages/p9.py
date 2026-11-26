"""P9 · 经典 vLLM v1 引擎 step 循环（学习页）。"""
from dg import Doc, box_for, legend_bar

X1, W1 = 40, 560      # 左列：同步 step 链
X2, W2 = 700, 620     # 中列：step 内部
X3, W3 = 1400, 520    # 右列：异步调度流水 + 数据结构


def build(doc: Doc) -> None:
    p = doc.page("09 vLLM v1 Step 循环", pw=2020, ph=1900)
    # 本页是通用 vLLM：全部中立色，只有"LWD 边侧改造"那一处取边端蓝。
    p.fill_map = {"ctrl": "neutral", "proxy": "neutral", "cloudp": "neutral",
                  "grey": "neutral", "ok": "neutral"}
    p.text(36, 16, 1700, 26, "⑨ · 经典 vLLM v1 引擎：一个 step 里到底发生了什么（学习页）",
           "title")
    p.text(36, 44, 1800, 20,
           "这张图与 LWD 无关，是理解 P3~P7 的地基：EngineCore 主线程只做「收包 → 调度 → 执行 → 回包」。"
           "LWD 的边/云引擎只是把 scheduler 与 runner 换成自己的实现。", "subtitle")

    # ---------------- 左：busy loop ----------------
    p.text(X1, 88, W1, 22, "<b>① 进程骨架：EngineCore 主线程</b>", "text_b")
    loop = [
        ("`run_busy_loop()`：只要 has_work 就转", "vllm/v1/engine/core.py:1465-1476", "grey"),
        ("`_process_input_queue()`：从 input_queue 取命令", "core.py:1491-1520", "ctrl"),
        ("命令分发：ADD / ABORT / WAKEUP / UTILITY", "core.py:1594-1628", "ctrl"),
        ("`_process_engine_step()`：跑一步", "core.py:1522-1540", "ctrl"),
        ("把 EngineCoreOutputs 交 output socket 线程", "core.py:1851-1918", "ctrl"),
    ]
    cy = 116
    ids = []
    for t, a, r in loop:
        cid, h = box_for(p, X1, cy, W1, t, a, None, role=r)
        ids.append((cid, h))
        cy += h + 30
    for i in range(len(ids) - 1):
        p.edge(ids[i][0], ids[i + 1][0], st="arrow", role="ctrl")
    # 回边（左侧绕回）
    p.edge(ids[-1][0], ids[0][0], st="arrow_dash_open", role="grey",
           points=[(X1 - 24, cy - h - 15), (X1 - 24, 116 + 20)],
           label="下一轮")

    # ---------------- 中：step 内部 ----------------
    p.text(X2, 88, W2, 22, "<b>② `EngineCore.step()` 内部四件事</b>", "text_b")
    inner = [
        ("`scheduler.schedule()` → SchedulerOutput",
         "core.py:589-620 · vllm/v1/core/sched/scheduler.py",
         "决定这一步跑哪些请求、各多少 token；**顺带在 KV 块池里分配块**", "proxy"),
        ("`kv_cache_manager.allocate_slots(...)`",
         "vllm/v1/core/kv_cache_manager.py:402",
         "新请求 first-fit 连续块；已有请求 append；前缀命中复用块", "proxy"),
        ("`executor.execute_model(scheduler_output)`",
         "vllm/v1/executor/*.py",
         "广播到各 worker：unicast RPC（mp）或直接调用（uni）", "cloudp"),
        ("worker：`execute_model` → `_prepare_inputs` → forward → `sample_tokens`",
         "vllm/v1/worker/gpu_model_runner.py · vllm_ascend/worker/model_runner_v1.py:2227",
         "slot_mapping 决定 KV 写到哪一行；采样得到 token", "cloudp"),
        ("`scheduler.update_from_output(outputs)`",
         "scheduler.py · vllm/v1/engine/core.py:589-620",
         "追加 token、判定 finished、把完成请求从 running 摘除", "proxy"),
        ("`_free_request` → 归还 KV 块 / 触发 connector",
         "scheduler.py:2625-2652 · block_pool.py:789",
         "没有 connector 就立即 free；有就按 delay_free_blocks 决定", "grey"),
        ("产出 `EngineCoreOutputs` → 前端", "core.py:589-620 → process_output_sockets:1851",
         "前端再做 detokenize / stop / SSE", "ok"),
    ]
    cy = 116
    m = []
    for t, a, e, r in inner:
        cid, h = box_for(p, X2, cy, W2, t, a, e, role=r)
        m.append((cid, h))
        cy += h + 30
    for i in range(len(m) - 1):
        p.edge(m[i][0], m[i + 1][0], st="arrow", role="ctrl")
    # 反馈（右侧绕回）
    p.edge(m[-1][0], m[0][0], st="arrow_dash_open", role="grey",
           points=[(X2 + W2 + 26, cy - h - 15), (X2 + W2 + 26, 116 + 20)],
           label="下一步")

    # ---------------- 右：异步流水 + 关键结构 ----------------
    p.text(X3, 88, W3, 22, "<b>③ --async-scheduling：scheduler 与执行重叠</b>", "text_b")
    pipe = [
        ("`step_with_batch_queue()`：step_fn 换成它",
         "core.py:630-745 · 本部署三端都用", "ctrl"),
        ("`batch_queue: deque[maxlen=max_concurrent_batches]`",
         "core.py:210-216", "ctrl"),
        ("scheduler 先算第 N+1 步，同时等第 N 步执行 future",
         "core.py:686-742", "ctrl"),
        ("执行完 → `update_from_output` 第 N 步 → pop 队尾",
         "core.py:701-745", "ctrl"),
        ("边侧把 batch_queue 深度改成按实例上限",
         "lwd_edge_engine.py:168-177", "edge"),
    ]
    cy = 116
    q = []
    for t, a, r in pipe:
        cid, h = box_for(p, X3, cy, W3, t, a, None, role=r)
        q.append((cid, h))
        cy += h + 30
    for i in range(len(q) - 1):
        p.edge(q[i][0], q[i + 1][0], st="arrow", role="ctrl")

    p.text(X3, cy + 10, W3, 22, "<b>④ 记住这五个结构就不会迷路</b>", "text_b")
    kn = [
        ("`SchedulerOutput`：这一步的「执行计划」", "vllm/v1/core/sched/output.py", "grey"),
        ("`ModelRunnerOutput`：这一步的「事实回报」", "vllm/v1/outputs.py", "grey"),
        ("`KVCacheManager` + `BlockPool`：块账本",
         "vllm/v1/core/kv_cache_manager.py · block_pool.py", "grey"),
        ("`EngineCoreRequest` / `Request`：跨进程身份 vs 内部可变态",
         "vllm/v1/engine/__init__.py · vllm/v1/request.py", "grey"),
        ("`EngineCoreOutputs`：回前端的增量包", "vllm/v1/engine/__init__.py", "grey"),
    ]
    cy += 40
    for t, a, r in kn:
        cid, h = box_for(p, X3, cy, W3, t, a, None, role=r)
        cy += h + 12

    p.text(X1, max(cy, 700) + 30, 1300, 44,
           "<b>为什么这张图对读 P3~P7 有用</b>：LWD 做的全部改动，本质是把 ② 里的 "
           "`scheduler.schedule()` / `executor.execute_model()` / `update_from_output()` "
           "三个方法换成「边侧只发 embedding、云侧只吃 embedding」的版本，"
           "而 ① 的进程骨架与 ③ 的流水深度完全不变。", "subtitle")
    legend_bar(p, max(cy, 700) + 90,
               extra_fill=("边侧改造点", "edge"))
