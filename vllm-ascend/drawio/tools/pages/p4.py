"""P4 · 模型加载的代码调用链（分层，逐函数）。

配色纪律：填色只表示「角色/归属」——中立=通用 vLLM 代码，蓝=边端专属，
橙=云 P 专属，绿=云 D 专属。阶段分组不另设颜色，用色带 + 编号 + 虚线框表达；
★ 分叉点用虚线边框标出（不是失败，所以不用红色）。
"""
from dg import Doc, box_for, legend_bar, measure_h

X1, W1 = 40, 700          # 调用链列
X2, W2 = 790, 650         # 注释列
GAP = 30                  # 行间距（要够放得下一条可见的连线）
GROUP_H = 32
GROUP_GAP = 26

VERT = "exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
DASHED = "dashed=1;"


def build(doc: Doc) -> None:
    p = doc.page("04 模型加载·调用链", pw=1500, ph=2200)
    p.text(36, 16, 1400, 26,
           "④ · 模型加载的代码调用链（CLI → 配置 → 进程 → worker → loader → 权重 → KV）",
           "title")
    p.text(36, 44, 1420, 20,
           "左列＝真实调用栈（自上而下，按阶段分组）；右列＝这一步做什么 / 为什么 / 谁在这里分叉。"
           "填色＝归属（沙＝通用，蓝＝边端，橙＝云 P，绿＝云 D）；虚线框＝★分叉/易错。"
           "锚点按 feat/pd-sep-lwd-v0.30.0。", "subtitle")
    p.text(X1 + 6, 80, W1, 20, "<b>调用链（自上而下）</b>", "text_b")
    p.text(X2 + 6, 80, W2, 20, "<b>说明 / 分叉点</b>", "text_b")

    y = 110

    def group(label, hint, rows):
        """rows: (title, anchor, side, fill_role, is_branch)"""
        nonlocal y
        p.box(X1, y, X2 + W2 - X1, GROUP_H,
             f'<b>{label}</b>'
             f'<font style="font-size:10px" color="#555555">　{hint}</font>',
             "phase")
        y += GROUP_H + 14
        ids: list[str] = []
        for title, anchor, side, role, branch in rows:
            extra = DASHED if branch else ""
            h = measure_h(title, anchor or "", "", W1)
            cid, hh = box_for(p, X1, y, W1, title, anchor, None,
                              role=role or "neutral", h=h)
            p.cells[-1].style += extra
            ids.append(cid)
            if side:
                sh = max(hh, measure_h(side, "", "", W2, min_h=30))
                p.box(X2, y, W2, sh, side, "note", "code")
            y += hh + GAP
        for a, b in zip(ids, ids[1:]):
            p.edge(a, b, st="arrow", role="http", extra_style=VERT)
        y += GROUP_GAP - GAP

    group("① CLI 入口", "三端完全一致：argparse → dispatch → 起服务", [
        ("`python -m vllm.entrypoints.cli.main serve <model> …`",
         "vllm/entrypoints/cli/main.py:111 → main():39",
         "argv 里唯一决定形态的是 `--additional-config` 的 `lwd_config`", None, False),
        ("`ServeSubcommand.cmd()`",
         "vllm/entrypoints/cli/serve.py:52",
         "LWD 钩子 `serve.py:81 lwd_dispatch_attach_frontend(args)`（本形态 no-op）", None, False),
        ("`uvloop.run(run_server(args))`（单 API server 分支）",
         "serve.py:159-161 → launchers/api_server/entry.py:163",
         "api_server_count=1：不起多前端 / headless / dp_supervisor", None, False),
    ])

    group("② 配置构建", "配置期决定「哪种 yaml 会被拒、谁是哪个角色」", [
        ("`create_engine_config()`",
         "launchers/api_server/entry.py:82 → vllm/engine/arg_utils.py:2044",
         "AsyncEngineArgs.from_cli_args 之后立刻建配置", None, False),
        ("`resolve_lwd_config()` → `LwdConfig.from_dict()`",
         "arg_utils.py:2056-2060 → vllm/config/lwd/__init__.py:25 · entry.py:43",
         "lwd_config 只允许 {path, role, instance_id} 三个键（extra=forbid）", None, False),
        ("`LwdTopology.from_file()` → `validate_layout()`",
         "vllm/config/lwd/topology.py:330-336 → :385-424",
         "实例数自洽 / ctrl_port 只在云 / PD 追加 6 条 / host 布局 / 端口唯一", None, False),
        ("`digest = sha256(canonical json)` 前 16 hex",
         "topology.py:352-367",
         "日志 `[LWD][config] resolved=…` 带它；三端不同＝拓扑没同步", None, False),
        ("`LwdConfig.from_topology()`：取本实例 + coordination",
         "entry.py:57-85 · coordination.py:16-34",
         "identity 决定连哪条控制面通道、绑哪个端口", None, False),
        ("`VllmConfig.__post_init__` → `log_lwd_config` → `lwd_check_parallel_config`",
         "vllm/config/vllm.py:1261-1299",
         "host 模式在 lwd_parallel_groups.py:240-251 早退（不查 rank）", None, False),
        ("role=cloud → 强制 `skip_tokenizer_init=False` / `mm_encoder_remote=True`",
         "vllm/config/vllm.py:1276-1282",
         "云端要能处理视觉占位、吃外部 embedding", "cloudp", False),
        ("host comm 且 world_size==1 → 换 `LwdHostUniProcExecutor`",
         "vllm/config/vllm.py:1306-1316",
         "只对 edge 生效：让阻塞式 host 收发与引擎线程重叠", "edge", False),
        ("`platform.check_and_update_config` → 写 `parallel_config.worker_cls`",
         "vllm-ascend/vllm_ascend/platform.py:536 → :1363-1384",
         "★分叉①：role=edge → LwdEdgeWorker；role=cloud → LwdCloudWorker", None, True),
    ])

    group("③ 前端起服与进程树", "cloud 的 coord server 先于 EngineCore 起来", [
        ("`AsyncLLM.from_vllm_config` → `EngineCoreClient.make_async_mp_client`",
         "vllm/v1/engine/async_llm.py · core_client.py:558-658",
         "前端进程先建 ZMQ 客户端 socket：handshake / input / output 三通道", None, False),
        ("cloud：`start_lwd_cloud_control_server`（coord_port 8100/8101）",
         "vllm/v1/engine/utils.py:167-188 → lwd_cloud_control.py:662-719",
         "先占端口再起 uvicorn；PD proxy 打的是它，不是 8600/8601", "cloudp", False),
        ("`CoreEngineProcManager` → `proc.start()`",
         "vllm/v1/engine/utils.py:238-244 · :283",
         "spawn `EngineCore`；握手回填 parallel_config 的地址与端口", None, False),
        ("`EngineCoreProc.run_engine_core()`：按 role 选引擎类",
         "vllm/v1/engine/core.py:1287 → :1351-1380",
         "★分叉②：edge→LwdEdgeEngineCore；cloud→LwdCloudEngineCore", None, True),
    ])

    group("④ Worker 初始化", "edge 的 worker 与引擎同进程（uni 执行器）", [
        ("`EngineCore.__init__`：executor → KV → scheduler",
         "core.py:108-251（executor :134 · KV :145 · scheduler :162）",
         "顺序即真理：worker 全部 ready 之后才做 KV 画像（→ P5）", None, False),
        ("`executor_class(vllm_config)`：spawn / 内建 worker",
         "multiproc_executor.py:641（cloud）· uniproc_executor.py:52（edge）",
         "同序三步：init_worker → init_device → load_model", None, False),
        ("`init_device` → `_init_device`（NPU 绑定 + 显存快照）",
         "vllm_ascend/worker/worker.py:498 → :391-473",
         "ASCEND_RT_VISIBLE_DEVICES 在此生效；之后禁止换卡", None, False),
        ("`_init_worker_distributed_environment`（hccl world + 并行组）",
         "worker.py:490 → :1283-1304 · vllm/distributed/parallel_state.py:2038-2044",
         "host 模式走 native：只建实例内 TP/MC2 组，不建任何 LWD 组", None, False),
        ("LWD 覆写：换 runner + `_init_lwd_data_plane()` → 建 host 通道",
         "lwd_worker_base.py:29-34 · :52-74 · host/lwd_host_channels.py:219",
         "★分叉③：edge→LwdEdgeModelRunner；cloud→LwdCloudModelRunner。"
         "sidecar daemon 也在这里被拉起（ucx_port.py:173-242）", None, True),
    ])

    group("⑤ 模型装载", "构造期决定显存，装载期决定读盘 IO；两仓在此交接", [
        ("`NPUWorker.load_model()` → `runner.load_model()`",
         "vllm_ascend/worker/worker.py:854-865",
         "edge/cloud 的 runner 覆写在此接上 super()", None, False),
        ("`NPUModelRunner.load_model()`：预热线程 + 显存 profiler",
         "vllm_ascend/worker/model_runner_v1.py:4372-4405",
         "日志 `Starting to load model …` 的落点", None, False),
        ("`get_model()` → `DefaultModelLoader.load_model()`",
         "model_loader/__init__.py:130 · base_loader.py:56-93",
         "set_default_torch_dtype → target_device → create_model → load_weights", None, False),
        ("`create_model()` → `initialize_model()`（LWD 守卫）",
         "base_loader.py:68-72 · model_loader/utils.py:38 · models/utils.py:864-896",
         "★分叉④：按 LWD_MODULE_OWNERSHIP 建/不建，缺失处 PPMissingLayer 占位", None, True),
        ("`get_all_weights()`：role → skip 集合",
         "default_loader.py:341-374 · lwd_weight_filter.py:60",
         "在 safetensors 迭代时丢弃被跳过的张量：省读盘 + 省显存", None, False),
        ("`model.load_weights()` + `track_weights_loading`",
         "default_loader.py:458-476 · :490-513",
         "严格未加载检查：漏覆盖即报错（三层防漏的第二层）", None, False),
        ("`process_weights_after_loading()`",
         "model_loader/loader/utils.py:97",
         "量化 / 重排 / 融合；之后权重定型", None, False),
    ])

    group("⑥ 装载后与 KV 规格", "三端在这里收尾分叉", [
        ("edge：装 `tid2eid` + 固定 UP layout + 跳过 warmup/capture",
         "lwd_edge_model_runner.py:84-105 · lwd_edge_worker.py:62-71",
         "0 层拓扑跑完整前向会在 final norm 解包失败，故整段跳过", "edge", False),
        ("cloud：Engram 校验 + 层数/首末层日志 + 图捕获",
         "lwd_cloud_model_runner.py:190-211 · lwd_cloud_worker.py:161-182",
         "P 用 eager；D 用 FULL_DECODE_ONLY（cloud_d.sh:31）", "cloudp", False),
        ("`_initialize_kv_caches()`：spec → 画像 → 块池 → warmup",
         "vllm/v1/engine/core.py:254-383",
         "edge 因 spec 为空走 else 分支（:310-312）→ 详见 P5", None, False),
    ])

    legend_bar(p, y + 30, extra_line=("虚线框 = ★分叉 / 易错点", "neutral"))
    p.text(X1, y + 106, 1400, 22,
           "<b>读法</b>：同一条竖链在三端走法一致；★ 的 4 处是唯一分叉点。"
           "右列每条注释与左边同高的方框一一对应。", "subtitle")
