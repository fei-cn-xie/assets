from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import load_workbook
from lib.build_lwd_perf_excel import base_sheet, write_table

BOOKS = [Path(p) for p in os.environ["PD_CODE_MAP_BOOKS"].split(os.pathsep) if p]

ROWS = [
    ("first_scheduled", "P/D调度", "Scheduler.schedule", "vllm/v1/core/sched/scheduler.py", "348-1002", "first_scheduled 标记在1000；accepted→首次计算调度", "scheduler.py:1848"),
    ("first_token", "P/D输出", "Scheduler.update_from_output", "vllm/v1/core/sched/scheduler.py", "1363-1611", "first_token 标记在1489；调度→首个 token", "scheduler.py:1848"),
    ("token_output", "P/D输出", "Scheduler.update_from_output", "vllm/v1/core/sched/scheduler.py", "1493-1500", "每次 new_token_ids 输出；含 new_token_count/completion_tokens", "scheduler.py:1848"),
    ("finished", "P/D输出", "Scheduler.update_from_output", "vllm/v1/core/sched/scheduler.py", "1563", "请求完成；含 prompt_tokens/completion_tokens", "scheduler.py:1848"),
    ("frontend_output", "P/D输出", "AsyncLLM.generate", "vllm/v1/engine/async_llm.py", "559-681", "前端实际 RequestOutput 事件在625-636", "async_llm.py:625"),
    ("model-forward.npu_ms", "NPU", "NPUModelRunner.execute_model + _model_forward", "vllm-ascend/vllm_ascend/worker/model_runner_v1.py", "1961-2414；2890-2926", "同步在2333/2338；真正模型调用在2334-2336，_model_forward 在2890", "model_runner_v1.py:2340"),
    ("cloud-step.sample", "D采样/采集", "LwdCloudWorker.sample_tokens", "vllm-ascend/vllm_ascend/worker/lwd_cloud/lwd_cloud_worker.py", "777-780", "_t0=778，model_runner.sample_tokens=779，_t_sample=780", "lwd_cloud_worker.py:844"),
    ("cloud-step.send", "D DOWN提交", "LwdCloudWorker.sample_tokens", "vllm-ascend/vllm_ascend/worker/lwd_cloud/lwd_cloud_worker.py", "780-853；812-822", "_t_sample 后到函数尾；submit_send 在812-822", "lwd_cloud_worker.py:844"),
    ("collect.rank", "D rank收集", "LwdCloudModelRunner._lwd_collect_down_payload", "vllm-ascend/vllm_ascend/worker/lwd_cloud/lwd_cloud_model_runner.py", "1035-1205", "_t0=1058；rank encode 在1107/1188；_t_pack=1205", "lwd_cloud_model_runner.py:1249"),
    ("collect.pinned", "D meta组装", "LwdCloudModelRunner._lwd_collect_down_payload", "vllm-ascend/vllm_ascend/worker/lwd_cloud/lwd_cloud_model_runner.py", "1205-1256", "payload/meta_dev 组装；不是输出线程 pinned 收割本身", "lwd_cloud_model_runner.py:1249"),
    ("meta-harvest", "输出线程收割", "WorkerProc.enqueue_output", "vllm/v1/executor/multiproc_executor.py", "1185-1299", "_t_sync=1232；ready_event wait=1234；copy=1247；done.synchronize=1251", "multiproc_executor.py:1271"),
    ("send.queue", "Sidecar发送", "SidecarLwdChannel.submit + activate + _step", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "492-550；365-368", "submit 排队到 activated；seqno FIFO 激活", "sidecar_channel.py:305"),
    ("send.clone", "Sidecar发送", "SidecarDataRuntime._step + stager.snapshot", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "220-224；373-380", "activated→snapshot_done；等待设备快照完成", "sidecar_channel.py:305"),
    ("send.slot_wait", "Sidecar发送", "SidecarDataRuntime._claim_slot", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "228-233；375-378；398-401", "等待 staging slot；slot_wait_start→slot_acquired", "sidecar_channel.py:305"),
    ("send.d2h", "Sidecar发送", "SidecarDataRuntime._step + stager.stage_send", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "379-383", "stage_send 后等待 D2H Event；snapshot_done→d2h_done", "sidecar_channel.py:305"),
    ("send.serialize", "Sidecar发送", "SidecarDataRuntime._step + stager.send_bytes", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "384-385", "D2H 完成后组装 Host bytes；d2h_done→host_bytes_done", "sidecar_channel.py:305"),
    ("send.network", "Sidecar发送", "SidecarDataRuntime._post + _on_host", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "261-282；245-259", "host_bytes_done→network_done，含 HostDataClient 排队和 ACK", "sidecar_channel.py:305"),
    ("send.total", "Sidecar发送", "SidecarLwdChannel.submit → _step(network done)", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "492-540；350-397", "submit→network_done 的完整发送操作", "sidecar_channel.py:305"),
    ("recv.queue", "Sidecar接收", "SidecarLwdChannel.submit + activate + _step", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "492-550；365-370", "接收操作 submit→activated 并 post recv", "sidecar_channel.py:329"),
    ("recv.rx_wait", "Sidecar接收", "SidecarDataRuntime._post + _on_host", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "370；391-398；245-259", "activated→rx_host_done，等待对端/Host 数据就绪", "sidecar_channel.py:329"),
    ("recv.slot_wait", "Sidecar接收", "SidecarDataRuntime._claim_slot", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "228-233；398-401", "等待接收 staging slot", "sidecar_channel.py:329"),
    ("recv.h2d", "Sidecar接收", "SidecarDataRuntime._step + stager.stage_recv", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "403-410", "stage_recv 后等待 H2D Event；rx_host_done→h2d_done", "sidecar_channel.py:329"),
    ("recv.total", "Sidecar接收", "SidecarLwdChannel.submit → _step(h2d done)", "vllm-ascend/vllm_ascend/distributed/lwd_comm/sidecar_channel.py", "492-540；350-411", "submit→h2d_done 的完整接收操作", "sidecar_channel.py:329"),
    ("unembed.post_recv", "边侧UNEMBED", "LwdEdgeWorker._execute_lwd_unembed", "vllm-ascend/vllm_ascend/worker/lwd_edge_worker.py", "514-525", "_t0=514；submit_recv=515-524；_t_post=525", "lwd_edge_worker.py:617"),
    ("unembed.wait_tensor", "边侧UNEMBED", "LwdEdgeWorker._execute_lwd_unembed", "vllm-ascend/vllm_ascend/worker/lwd_edge_worker.py", "526-533", "wait_for_comm=527；result=528；_t_tensor=533；不是纯网络时间", "lwd_edge_worker.py:617"),
    ("unembed.lm_head", "边侧UNEMBED", "LwdEdgeWorker._execute_lwd_unembed + model.compute_logits", "vllm-ascend/vllm_ascend/worker/lwd_edge_worker.py", "534-575", "compute_logits=574；_t_lm=575", "lwd_edge_worker.py:617"),
    ("unembed.select", "边侧UNEMBED", "select_token_batch + _t_sel", "vllm-ascend/vllm_ascend/worker/lwd_edge_worker.py", "595-605", "rank→token 选择、排序与 D2H；_t_sel=605", "lwd_edge_worker.py:617"),
    ("unembed.total", "边侧UNEMBED", "LwdEdgeWorker._execute_lwd_unembed", "vllm-ascend/vllm_ascend/worker/lwd_edge_worker.py", "514-605", "_t0=514 → _t_sel=605", "lwd_edge_worker.py:617"),
    ("edge_interval", "边侧输出节奏", "相邻两次 _execute_lwd_unembed 的完成 mono 差值", "vllm-ascend/vllm_ascend/worker/lwd_edge_worker.py", "相邻 unembed 日志的 mono", "不是独立 kernel；包含调度、通信、等待和恢复", "lwd_edge_worker.py:617"),
    ("decode_generation", "边云请求", "LwdEdgeScheduler.lwd_pd_deliver_tokens", "vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py", "2476-2575", "first_token_at=2535；d_finished_at=2563", "lwd_edge_scheduler.py:2562"),
    ("D import ready", "D KV导入", "LwdCloudMixedScheduler._lwd_pd_collect_imports", "vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_mixed_scheduler.py", "303-374", "kv_ready_ts=346；admit_to_ready 日志在348-353", "lwd_cloud_mixed_scheduler.py:348"),
    ("wire send/recv", "TCP transport", "TcpHostTransport.send / receive", "vllm/v1/lwd_sidecar/transport.py", "73-94；96-118", "writer.drain/readexactly 的实际 transport 循环", "transport.py:83/106"),
    ("frame_host_ready", "Sidecar frame", "LwdSidecarServer frame 完成处理", "vllm/v1/lwd_sidecar/server.py", "795-835", "receive_done=797；grant/receive/pool_write/total=829-834", "server.py:818"),
    ("HTTP first_content/done", "HTTP输出", "OpenAIServingChat.chat_completion_stream_generator", "vllm/vllm/entrypoints/openai/chat_completion/serving.py", "450-1004；1001", "first_content=925-932；[DONE] 前记录完成；full_response=1438", "serving.py:931/1001/1438"),
]


def main():
    for path in BOOKS:
        wb = load_workbook(path)
        if "Decode代码地图" in wb.sheetnames:
            del wb["Decode代码地图"]
        ws = base_sheet(
            wb,
            "Decode代码地图",
            "行号以当前工作树为准；部署 commit 变化时优先按函数名定位。日志行号与业务执行行号分列。",
        )
        headers = ["序号", "耗时字段/阶段", "架构层", "实际业务函数", "代码文件", "代码行号", "计时起止与含义", "日志打印位置"]
        table_rows = []
        for idx, (field, layer, func, file, lines, timing, log) in enumerate(ROWS, 1):
            table_rows.append([idx, field, layer, func, file, lines, timing, log])
        write_table(ws, 4, headers, table_rows, [8, 26, 18, 48, 64, 26, 66, 30])
        ws.freeze_panes = "A5"
        wb.save(path)
        checked = load_workbook(path, read_only=True)
        assert "Decode代码地图" in checked.sheetnames
        assert checked["Decode代码地图"].max_row == len(ROWS) + 4
        print(f"updated={path.resolve()}; rows={len(ROWS)}")


if __name__ == "__main__":
    main()
