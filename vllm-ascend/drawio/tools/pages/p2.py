"""P2 · 启动时序：从 bash 脚本到 ready（进程级）。"""
from dg import Doc, legend_bar
from seq import Seq


def build(doc: Doc) -> None:
    p = doc.page("02 启动时序", pw=2100, ph=2450)
    # 本页讲"进程骨架"，与角色无关：泳道用**灰阶梯度**（灰阶不是角色色，不冲突），
    # 只有真正角色专属的步骤用 hi_* 取角色色。箭头仍按平面着色。
    p.fill_map = {
        "sh": "proc1", "fe": "proc2", "ec": "proc3", "wk": "proc4",
        "sd": "proc5", "ot": "neutral",
        "grey": "proc1", "edge": "proc2", "ctrl": "proc3", "cloudp": "proc4",
        "proxy": "proc5", "cloudd": "neutral", "ok": "neutral",
        "hi_edge": "edge", "hi_p": "cloudp", "hi_d": "cloudd",
        "hi_proxy": "proxy", "hi_fail": "fail",
    }
    s = Seq(
        p,
        lanes=[("sh", "启动脚本 / 环境", "grey"),
               ("fe", "前端进程<br><font style='font-size:9px'>API Server</font>", "edge"),
               ("ec", "EngineCore 进程", "ctrl"),
               ("wk", "Worker 进程<br><font style='font-size:9px'>× TP rank</font>", "cloudp"),
               ("sd", "sidecar daemon", "proxy"),
               ("ot", "对端（P / D / 边）", "cloudd")],
        lane_w=290, title="② · 启动时序：从 bash 到 ready（进程级）",
        subtitle="三条脚本共用 env.sh；四段：配置与校验 → 进程树 → Worker 初始化 → 控制面/数据面握手 → KV 画像。"
                   "模型加载细节见 P3/P4，KV 画像细节见 P5。")

    s.phase("① 脚本与环境（bash env.sh，三端共用）")
    s.step("sh", "env.sh：导出运行时变量",
           "assets/c_home/bash/pds/30rc_1124/sidecar/env.sh:42-80",
           "HCCL_SOCKET_IFNAME=lo · UCX_TLS=rc_verbs,tcp,self · LD_PRELOAD=jemalloc"
           " · PYTHONPATH=vllm-ascend:vllm · VLLM_ENGINE_READY_TIMEOUT_S=1800", role="grey")
    s.step("sh", "lwd_require_ucx_runtime：预检 libucp",
           "env.sh:86-105", "缺 libucp.so.0 → 直接 exit 3；否则会伪装成 605s 握手超时", role="grey")
    s.msg("sh", "fe", "lwd_run → exec python -m vllm.entrypoints.cli.main serve",
          mode="sync", role="data")

    s.phase("② 配置与校验（前端进程内，三端同序，只有 role 不同）")
    s.step("fe", "CLI 解析 → EngineArgs.create_engine_config()",
           "vllm/engine/arg_utils.py:2054-2062",
           "--additional-config '{\"lwd_config\":{path,role,instance_id}}'"
           " ＋ --kv-transfer-config（P=producer / D=consumer）", role="edge")
    s.step("fe", "LwdConfig.from_dict → LwdTopology.from_file",
           "vllm/config/lwd/entry.py:43-55 · vllm/config/lwd/topology.py",
           "拓扑是唯一事实源；sha256 digest 打到日志，三端必须相同", role="edge")
    s.step("fe", "VllmConfig.__post_init__ → lwd_check_parallel_config",
           "vllm/config/vllm.py:1297-1299 → vllm/distributed/lwd/lwd_parallel_groups.py:199-263",
           "拒绝 PP&gt;1 / PCP / DCP / elastic EP；host 模式不校验 rank，只校验 TP==每 DP rank 数", role="edge")
    s.step("fe", "platform.check_and_update_config → 选定 worker_cls",
           "vllm-ascend/vllm_ascend/platform.py:536 → :1363-1384",
           "role=edge → LwdEdgeWorker；role=cloud → LwdCloudWorker（两者都继承 NPUWorker）", role="edge")

    s.phase("③ 起服与进程树")
    s.step("fe", "构建 OpenAI app + AsyncLLM 前端",
           "vllm/entrypoints/openai/api_server.py · vllm/v1/engine/async_llm.py",
           "云侧额外在起 EngineCore 之前先拉起 LWD coord server（P:8100 / D:8101）", role="edge")
    s.step("fe", "start_lwd_cloud_control_server（仅云端）",
           "vllm/v1/engine/utils.py:170-188 → control_cloud_scheduler/lwd_cloud_control.py:662",
           "PD proxy 打的就是这个端口，不是 8600/8601 的 OpenAI 端口", role="hi_p")
    s.msg("fe", "ec", "spawn EngineCore 子进程 + ZMQ 握手", mode="sync", role="ctrl")
    s.step("ec", "run_engine_core()：按角色选引擎类",
           "vllm/v1/engine/core.py:1287 · :1351-1380",
           "edge → LwdEdgeEngineCore；cloud → LwdCloudEngineCore（MoE+DP&gt;1 才是 LwdCloudDPEngineCore）",
           role="ctrl")
    s.step("ec", "起 input/output socket 线程，进入 busy loop",
           "vllm/v1/engine/core.py:1111-1130 · run_busy_loop:1465",
           "EngineCore 主线程只做调度；收包/发包各一条线程", role="ctrl")
    s.msg("ec", "wk", "executor_class(vllm_config) → spawn WorkerProc × TP", mode="sync",
          role="ctrl")
    s.step("wk", "WorkerProc.__init__ → init_device",
           "vllm-ascend/vllm_ascend/worker/lwd_base_model/lwd_worker_base.py:29-34",
           "各角色 NPU 绑定：边 NPU 4（TP=1）· P NPU 0,1（TP=2）· D NPU 2,3（TP=2）", role="cloudp")
    s.step("wk", "init_worker_distributed_environment（host 模式走 native）",
           "vllm/distributed/parallel_state.py:2038-2044 · vllm-ascend/.../parallel_state.py:74-82",
           "只建实例内 TP/MC2 组；不建任何 LWD data-plane 组（device 模式才建）", role="cloudp")
    s.step("wk", "init_model_runner → LwdEdgeModelRunner / LwdCloudModelRunner",
           "lwd_edge_worker.py:39 · lwd_cloud_worker.py:83", role="cloudp")
    s.step("wk", "load_model（→ P3 / P4 展开）",
           "vllm_ascend/worker/model_runner_v1.py:4372 · vllm/model_executor/model_loader/default_loader.py",
           "云侧额外 enable_prompt_embeds=True（必须能吃外部 embedding）", role="cloudp")

    s.phase("④ sidecar 传输 daemon（每机一个，首个引擎进程拉起）")
    s.msg("wk", "sd", "首次 open_transport(mode=\"daemon\")：connect → 失败则 flock 后 spawn",
          mode="sync", role="data")
    s.step("sd", "serve_forever：accept ctl → ATTACH/DETACH/PING",
           "vllm/v1/lwd_comm/data_comm/ucx_daemon.py:371 · :456-491",
           "ATTACH 时用 SCM_RIGHTS 回传 3 个 fd（memfd 段 + 两对 pipe）；空闲 30s 自杀", role="proxy")

    s.phase("⑤ 控制面握手（ZMQ ROUTER/DEALER）")
    s.step("ec", "_lwd_start_control_plane：ROUTER bind ctrl_port",
           "lwd_edge_engine.py:126 · vllm/v1/lwd_control/lwd_comm/lwd_control_plane.py",
           "边端 bind（连 6100/6101 的两条链路），云端按拓扑 bind 自己的 ctrl_port", role="ctrl")
    s.msg("ec", "ot", "HELLO + CloudParams（并发上限 / hash block / stage_role / pd_version）",
          mode="sync", role="ctrl")
    s.msg("ot", "ec", "参数回执 → wait_until_ready(handshake_timeout)", mode="return",
          role="ctrl")
    s.step("ec", "PD 校验：必须同时存在 prefill + decode 且 hash block 一致",
           "vllm/v1/lwd_control/pd/edge_runtime.py:74-88", role="cloudd")

    s.phase("⑥ 数据面建链（只有边端阻塞等）")
    s.step("wk", "lwd_init_host_channels：身份驱动建链 + UP 块定尺 + 内存预算闸",
           "vllm-ascend/.../host/lwd_host_channels.py:219-375",
           "UP 块按 hidden_size / hash 层 / max_num_batched_tokens 推导并 32MiB 对齐；"
           "预算 12GiB/rank", role="data")
    s.msg("wk", "ot", "wait_ready（边端 → 云：单条 605s，最坏 4 条）", mode="sync",
          role="data")
    s.note(("fe", "ec"),
           "边端启动会静默等链路、进展日志是 WARNING 级；云端只 bind/connect 不等。"
           "这也是 README 要求「边端最后启动」的原因。")

    s.phase("⑦ KV cache 分配与 warmup（→ P5 展开）")
    s.step("ec", "_initialize_kv_caches：spec → 显存画像 → 块池 → initialize_from_config",
           "vllm/v1/engine/core.py:254-383",
           "边端 get_kv_cache_spec 返回 {} → 走 else 分支，连 determine_available_memory 都不调用", role="ctrl")
    s.step("ec", "compile_or_warm_up_model（P: eager · D: FULL_DECODE_ONLY）",
           "vllm/v1/engine/core.py:356",
           "边端整段跳过：LwdEdgeWorker.compile_or_warm_up_model 直接返回 0", role="ctrl")
    s.msg("fe", "ot", "uvicorn 起服 · /health 200 · 对端握手完成 → ready", mode="return",
          role="ctrl")
    s.gap(18)
    s.note(("sh", "ot"),
           "读法：本页讲的是<b>进程骨架</b>，与具体角色无关，所以方框一律中立色；"
           "只有橙/绿两块是真正角色专属的步骤（云端的 coord server、PD 校验）。"
           "箭头颜色表示平面：紫=控制面 ZMQ，黄=数据面 UCX，灰=HTTP/进程创建。")

    # ---- 统一图例（全文件同义）----
    legend_bar(p, s._layout_rows(dry=True) + 44,
               extra_fill=("灰阶 = 进程泳道（非角色）", "proc3"))
