"""P3 · 模型加载：公共骨架 + 边/P/D 三分侧。"""
from dg import Doc, legend_bar
from seq import Seq


def build(doc: Doc) -> None:
    p = doc.page("03 模型加载·三分侧", pw=2200, ph=2600)
    s = Seq(
        p,
        lanes=[("all", "公共（vllm 仓，三端同序）", "grey"),
               ("e", "edge (.194, NPU4, TP=1)", "edge"),
               ("p", "cloud-P (.195, NPU0,1, TP=2)", "cloudp"),
               ("d", "cloud-D (.195, NPU2,3, TP=2)", "cloudd")],
        lane_w=380, title="③ · 模型加载生命周期（公共骨架 + 边/P/D 三分侧）",
        subtitle="六段：配置期 → Worker 初始化 → 模型构造 → 权重装载 → 装载后处理 → KV 规格。"
                   "同一段代码在两个角色上跑出不同结果，靠的是 role 传入的模块归属表。")

    s.phase("① 配置期：三端完全同序，只有 role / instance_id / TP 不同")
    s.step("all", "`python -m vllm.entrypoints.cli.main serve` → ServeSubcommand.cmd",
           "vllm/entrypoints/cli/main.py:111 · vllm/entrypoints/cli/serve.py:52",
           "LWD 钩子 serve.py:81 lwd_dispatch_attach_frontend；本形态无 --api-server-* → 返回 False",
           role="grey")
    s.step("all", "AsyncEngineArgs.create_engine_config()",
           "vllm/engine/arg_utils.py:2044 · LWD 解析 :2056-2060",
           "--additional-config 的 lwd_config={path,role,instance_id} → resolve_lwd_config → "
           "LwdConfig.from_dict（entry.py:43）", role="grey")
    s.step("all", "LwdTopology.from_file → validate_layout() → 算 digest",
           "vllm/config/lwd/topology.py:330-368",
           "digest = sha256(canonical json) 前 16 hex；三端必须一模一样，否则日志对不上",
           role="grey")
    s.step("all", "VllmConfig.__post_init__ → log_lwd_config → lwd_check_parallel_config",
           "vllm/config/vllm.py:1261-1299",
           "拒绝 PP&gt;1 / PCP / DCP / elastic EP；host 模式在 lwd_parallel_groups.py:240-251 早退",
           role="grey")
    s.step("all", "platform.check_and_update_config → 二分 worker_cls",
           "vllm-ascend/vllm_ascend/platform.py:536 → :1363-1384",
           "role=edge → LwdEdgeWorker；role=cloud → LwdCloudWorker（都继承 NPUWorker）",
           role="grey")
    s.steps([("e", "执行器：world_size=1 → \"uni\"", "vllm/config/vllm.py:1306-1316",
              "LwdHostUniProcExecutor：worker 与 EngineCore 同进程", "edge", "box", True, None),
             ("p", "执行器：TP=2 → \"mp\"", "vllm/config/parallel.py:966-1011",
              "MultiprocExecutor → 2 × VllmWorker 子进程", "cloudp", "box", True, None),
             ("d", "执行器：TP=2 → \"mp\"", "vllm/v1/executor/multiproc_executor.py:641",
              "同 P；worker 用 spawn 起", "cloudd", "box", True, None)])

    s.phase("② Worker 初始化（edge 的 worker 就在 EngineCore 进程里跑）")
    s.step("all", "WorkerProc.__init__ → init_worker → 按 worker_cls 实例化",
           "multiproc_executor.py:641-673 · uniproc_executor.py:52-74",
           "edge 走 UniProcExecutor 的 driver_worker，同序：init_worker → init_device → load_model",
           role="grey")
    s.step("all", "init_device → _init_device：NPU 绑定 + 显存快照",
           "vllm_ascend/worker/worker.py:498 → :391-473",
           "ASCEND_RT_VISIBLE_DEVICES 生效点；之后禁止换卡", role="grey")
    s.step("all", "_init_worker_distributed_environment：hccl world + 并行组",
           "worker.py:490 → :1283-1304 · vllm/distributed/parallel_state.py:2038-2044",
           "data_plane_comm=host → 走 native：只建实例内 TP / MC2 组，不建 LWD 组", role="grey")
    s.steps([("e", "换 runner：LwdEdgeModelRunner", "lwd_edge_worker.py:39",
              "再 _init_lwd_data_plane()（:44）", "edge", "box", True, None),
             ("p", "换 runner：LwdCloudModelRunner", "lwd_cloud_worker.py:83",
              "+ enable_prompt_embeds=True（:82），必须能吃外部 embedding", "cloudp", "box", True, None),
             ("d", "同 P，外加 _lwd_pd_decode=True", "lwd_cloud_worker.py:234",
              "P/D 双阶段总开关：决定 stager 还是 tail buffer", "cloudd", "box", True, None)])
    s.steps([("e", "数据面：own 全部连接，唯一阻塞侧", "host/lwd_host_channels.py:219 · :391-419",
              "lwd_init_host_channels → open_transport(daemon) → wait_ready（605s/条）", "edge", "box", True, None),
             ("p", "数据面：own 本 DP 的 UP recv/DOWN send", "lwd_host_channels.py:318-344",
              "+ LwdHostStager（全 prompt pinned，双水位）", "cloudp", "box", True, None),
             ("d", "数据面：own 本 DP 的 UP recv/DOWN send", "lwd_cloud_worker.py:142-150",
              "+ LwdTailBuffer（只留真实 tail 行）", "cloudd", "box", True, None)])

    s.phase("③ 模型构造：同一份 initialize_model，按角色建/不建")
    s.step("all", "NPUWorker.load_model → runner.load_model → NPUModelRunner.load_model",
           "worker.py:854-865 · model_runner_v1.py:4372",
           "先起 nz warm / early kernel warmup 线程，再进 DeviceMemoryProfiler", role="grey")
    s.step("all", "get_model → DefaultModelLoader.load_model → create_model",
           "vllm/model_executor/model_loader/__init__.py:130 · base_loader.py:56-72",
           "set_default_torch_dtype → target_device → initialize_model", role="grey")
    s.step("all", "LWD_MODULE_OWNERSHIP + lwd_skips_module：决定建不建",
           "vllm/model_executor/models/utils.py:864-896",
           "不建的位置用 PPMissingLayer 占位，保住 forward 签名与 named_parameters 契约",
           role="grey")
    s.steps([("e", "建：embed_tokens + lm_head + visual", "models/utils.py:864-875",
              "不建：model.layers（0 层）、model.norm、model.mtp", "edge", "box", True, None),
             ("p", "建：全部 layers + norm + MTP", "models/utils.py:864-875",
              "不建：visual（视觉塔留在边侧）", "cloudp", "box", True, None),
             ("d", "同 P", "同上", "P/D 权重完全相同，只是 stage_role 不同", "cloudd", "box", True, None)])

    s.phase("④ 权重装载：读盘前按角色丢张量")
    s.step("all", "get_all_weights → lwd_should_skip_weight(name, role)",
           "default_loader.py:341-374 · lwd_weight_filter.py:60",
           "skip 集合由 role 推导；命中即在迭代器里跳过，**不读盘**（省 IO 与显存）",
           role="grey")
    s.step("all", "model.load_weights(...) + track_weights_loading 严格未加载检查",
           "default_loader.py:458-513",
           "漏覆盖参数直接报错：这是「裁剪漏一层」的三层防漏之一", role="grey")
    s.step("all", "process_weights_after_loading（量化/重排/融合）",
           "vllm/model_executor/model_loader/loader/utils.py:97", role="grey")

    s.phase("⑤ 装载后处理：三端真正分叉的地方")
    s.steps([("e", "装 tid2eid 表 + 固定 UP layout + 跳过 warmup/capture",
              "lwd_edge_model_runner.py:84-105 · lwd_edge_worker.py:62-71",
              "0 层拓扑跑完整前向会在 final norm 解包失败，故整段跳过", "edge", "box", True, None),
             ("p", "层数/首末层日志 + eager（--enforce-eager）",
              "lwd_cloud_worker.py:161-182 · cloud_p.sh:26",
              "MTP drafter 就位；无 KV 传输导出前的状态", "cloudp", "box", True, None),
             ("d", "同 P + FULL_DECODE_ONLY 图捕获", "cloud_d.sh:31",
              "decode 段用 cudagraph，prefill 段不用", "cloudd", "box", True, None)])

    s.phase("⑥ KV 规格：边端在源头就断了")
    s.step("all", "get_kv_cache_spec() 汇总 → has_kv_cache 判定",
           "vllm/v1/engine/core.py:261-312 · model_runner_v1.py:6314",
           "spec 为空的角色连 determine_available_memory 都不会调用", role="grey")
    s.steps([("e", "返回 {} → num_gpu_blocks=0", "lwd_edge_model_runner.py:138-140",
              "无块池、无图、无 native 步进（详见 P5）", "edge", "box", True, None),
             ("p", "native spec（FullAttentionSpec/MLA/SWA…）", "attention.py:604",
              "→ 显存画像 + 块池（P5 展开）", "cloudp", "box", True, None),
             ("d", "同 P", "同上", "P/D 各自一套独立块池（同机不同卡）", "cloudd", "box", True, None)])

    s.gap(10)
    s.note(("all", "d"),
           "<b>读这张图的三个要点</b>：① ①~④ 三端跑的是同一份代码，差异全部由 "
           "lwd_config.role 注入；② 真正的分叉只有三处：worker/runner 类、模块归属表、KV spec；"
           "③ 边端的「少」不是裁剪掉，而是构造期用 PPMissingLayer 占位 + 装载期跳读盘，"
           "所以 forward 签名与参数契约一致。")

    # ---- 统一图例（全文件同义）----
    legend_bar(p, s._layout_rows(dry=True) + 44)
