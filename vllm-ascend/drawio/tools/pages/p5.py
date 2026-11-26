"""P5 · KV cache 生命周期。"""
from dg import Doc, legend_bar
from seq import Seq


def build(doc: Doc) -> None:
    p = doc.page("05 KV Cache 生命周期", pw=2200, ph=2700)
    # 本页全部发生在"一个云实例"内部：进程泳道一律中立色，
    # 只有承载 KV 搬运的那条泳道用数据面色；边端对照注释用边端蓝。
    p.fill_map = {"ctrl": "neutral", "proxy": "neutral", "cloudp": "neutral",
                  "cloudd": "data", "grey": "neutral"}
    s = Seq(
        p,
        lanes=[("ec", "EngineCore<br><font style='font-size:9px'>core.py 驱动</font>", "ctrl"),
               ("sch", "Scheduler / KVCacheManager<br><font style='font-size:9px'>引擎进程内</font>", "proxy"),
               ("wk", "Worker<br><font style='font-size:9px'>张量分配与写入</font>", "cloudp"),
               ("kv", "Mooncake / 对端<br><font style='font-size:9px'>P → D 块级搬运</font>", "cloudd")],
        lane_w=390, title="⑤ · KV Cache 生命周期：从显存画像到跨机交接",
        subtitle="六段：规格收集 → 显存画像 → 块池建立 → 请求级分配/写入 → 释放/回收 → PD 交接。"
                   "边端在第一步就断了（spec 为空），所以它既没有块池也没有 native 步进。")

    s.phase("① 规格收集与显存画像（启动期，worker 全部 ready 之后）")
    s.step("ec", "`_initialize_kv_caches` 入口：注册 spec → 收集",
           "vllm/v1/engine/core.py:254-261",
           "register_all_kvcache_specs → model_executor.get_kv_cache_specs()", role="ctrl")
    s.step("wk", "`get_kv_cache_spec()`：逐层算 spec",
           "vllm/model_executor/layers/attention/attention.py:604 · "
           "vllm_ascend/worker/model_runner_v1.py:6314",
           "FullAttentionSpec(block_size, num_kv_heads, head_size, dtype…)："
           "block 字节数 = 2 × kv_heads × head_size × BS × dtype", role="cloudp")
    s.note(("ec", "ec"),
           "<b>边端在这里就分叉</b>：LwdEdgeModelRunner.get_kv_cache_spec() 直接 return {} "
           "（lwd_edge_model_runner.py:138-140）—— 边侧没有 attention/transformer 层，不需要 KV。",
           role="edge")
    s.step("ec", "`has_kv_cache` 判定 + resolve_kv_cache_layout",
           "core.py:289-312",
           "没有 spec → available_gpu_memory = [0]×N，**连访存画像都不调用**", role="ctrl")
    s.step("wk", "`determine_available_memory`：request_memory = 整卡 × 0.90",
           "vllm_ascend/worker/worker.py:567 · vllm/v1/worker/utils.py:534-554",
           "--gpu-memory-utilization 0.90 的乘法只在这里；基数是整卡总显存", role="cloudp")
    s.step("wk", "`profile_run()` → `_dummy_run(max_num_tokens=8192)`",
           "vllm_ascend/worker/model_runner_v1.py:4328 · gpu_model_runner.py:6537",
           "按 max_num_batched_tokens/max_num_seqs 造一次真实形状的峰值", role="cloudp")
    s.step("wk", "峰值回收：del + gc.collect() + empty_cache()",
           "gpu_model_runner.py:6548-6550 · vllm/utils/mem_utils.py:298-301",
           "画像跑完必须把临时激活还回去，否则块池会算小", role="cloudp")
    s.step("ec", "available = requested − non_kv_cache_memory",
           "vllm_ascend/worker/worker.py:615-634",
           "日志锚点 `Available KV cache memory: … GiB`", role="ctrl")

    s.phase("② 块池建立（一次，启动期）")
    s.step("ec", "`get_kv_cache_configs`：num_blocks = available // bytes_per_block",
           "vllm/v1/core/kv_cache_utils.py:2617 → :1747-1752",
           "余数丢弃；写回 cache_config.num_gpu_blocks / block_size / capacity", role="ctrl")
    s.step("sch", "`KVCacheManager` → `KVCacheCoordinator` → `BlockPool(n)`",
           "vllm/v1/core/sched/scheduler.py:300 · kv_cache_manager.py:171 · "
           "kv_cache_coordinator.py:98",
           "manager 管策略（前缀命中/引用），pool 管空闲队列（O(1) 分配）", role="proxy")
    s.step("wk", "`initialize_from_config` → `ensure_kv_transfer_initialized`",
           "vllm/v1/worker/gpu_worker.py:735-759（:752 connector 先建）",
           "connector 必须先于张量分配建好，否则 register_kv_caches 无处注册", role="cloudp")
    s.step("wk", "`initialize_kv_cache_tensors`：单块 int8 backing + per-layer view",
           "vllm_ascend/worker/model_runner_v1.py:4583 · :5006 · :4928-4949",
           "有 kv_transfer_config 时多分配 alignment 再按 2MiB 对齐（RDMA 注册需要）", role="cloudp")
    s.step("wk", "`bind_kv_cache` + `register_kv_caches(connector)`",
           "vllm/v1/worker/utils.py:591 · model_runner_v1.py:4683",
           "forward_context[layer].bind_kv_cache 之后，每层前向才能拿到自己的 KV 张量",
           role="cloudp")
    s.note(("ec", "sch"),
           "边端对照：spec 为空 → kv_cache_tensors=[] → num_blocks=1（只剩 null_block）"
           "（kv_cache_utils.py:1664-1674 · block_pool.py:193-194）。边侧「块池」是个空壳。",
           role="edge")

    s.phase("③ 请求级：分配 → 块表 → 写入")
    s.step("sch", "`allocate_slots`（新请求与 append 同一入口）",
           "vllm/v1/core/kv_cache_manager.py:402 · 调用点 scheduler.py:1217",
           "三段式：腾空间 → 算 need + 准入 → 分配 + cache_blocks（前缀命中在此复用块）",
           role="proxy")
    s.step("sch", "block 表 → `compute_slot_mapping`：slot = blk×BS + off",
           "vllm/v1/worker/block_table.py:474 · 调用 gpu_model_runner.py:2205",
           "slot 是「每 token 写到哪一行」，物理块可以离散", role="proxy")
    s.step("wk", "forward 写 `attn_layer.kv_cache[slot]`",
           "vllm/model_executor/layers/attention/attention.py:716-737",
           "写 KV 与取 KV 都通过 reshape_and_cache / paged attention kernel", role="cloudp")

    s.phase("④ 释放与回收")
    s.step("sch", "`request_finished` → `_free_request` → `free_blocks`",
           "vllm/v1/core/sched/scheduler.py:2625-2652 · block_pool.py:789",
           "connector 可以返回 delay_free_blocks=True 来延迟释放（PD 就靠这个 pin 块）",
           role="proxy")
    s.step("sch", "抢占（preemption）：num_computed_tokens=0 + 回 waiting 队头",
           "scheduler.py:1541-1582 · LWD 最小重算 lwd_cloud_model_runner.py:575",
           "PD 抢占一律 fail-fast：`PD preemption recovery is unsupported`"
           "（pd/cloud_scheduler.py:115-117）", role="proxy")

    s.phase("⑤ PD 交接：P 侧 pin → D 侧 pull → 回执解 pin")
    s.step("ec", "P：`request_finished` 产出 kv_transfer_params + **pin 块**",
           "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:2014-2076",
           "delay_free_blocks=True（:2056-2059）→ 调度器跳过 _free_blocks（scheduler.py:2648）"
           "→ KVCacheTaskTracker.delayed_free_requests 记账", role="ctrl")
    s.msg("ec", "kv", "P 导出：块级 pull 源地址 + lwd_handoff(p_end / i_end / r_pos / hit)", mode="data",
          role="data")
    s.step("kv", "D 发起块级 pull（RDMA / memcpy），非 layerwise",
           "mooncake_connector.py:3831-3932（recv 线程 :420 · :568）",
           "三个 layerwise 钩子 wait_for_layer_load/save_kv_layer/wait_for_save 全是 pass"
           "（:1691/:1695/:1701）", role="cloudd")
    s.step("ec", "D：`get_num_new_matched_tokens` → (i_end, True) → 进 "
                 "`WAITING_FOR_REMOTE_KVS`",
           "mooncake_connector.py:1914-1983 · scheduler.py:1273-1303（:1276）",
           "本轮该请求 forward token 数 = 0；allocate_slots 带 delay_cache_blocks=True",
           role="ctrl")
    s.step("ec", "D：KV_READY 判定 = `num_computed_tokens == r_pos`",
           "vllm/v1/lwd_control/pd/cloud_scheduler.py:121-130",
           "不满足直接 PD fail（`PD remote KV import failed`），不重算、不降级", role="ctrl")
    s.msg("ec", "kv", "D → P：DONE_RECVING_MSG（读完了）", mode="return", role="cloudd")
    s.step("ec", "P：`finished_sending` → `_free_blocks`（真正释放）",
           "vllm/v1/core/sched/scheduler.py:3086-3089 · mooncake_connector.py:213-224",
           "超时兜底：`Force freed expired request`（:232-253）", role="ctrl")

    s.gap(8)
    s.note(("ec", "kv"),
           "<b>三方 KV 归属</b>：edge = 无块池（num_blocks=1 空壳，但保留 hash 链给云端命中）；"
           "P = 真实块池 + producer（写自己算的 KV，pin 到 D 读完）；"
           "D = 真实块池 + consumer（导入 P 的 KV 作为 num_computed 前缀，自己只写 1 行 tail）。"
           "易混点：拓扑里的 up/down_buffer_size 是 host UCX 帧块（MB），与 KV 无关；"
           "KV 走 kv_port 14579/14580 的 Mooncake 通道。")

    # ---- 统一图例（全文件同义）----
    legend_bar(p, s._layout_rows(dry=True) + 44)
