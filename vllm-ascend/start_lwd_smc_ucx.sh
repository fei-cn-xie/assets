#!/usr/bin/env bash
# =============================================================================
# LWD 边云 prefill_only 拉起脚本（一个脚本管两侧）
# 适用分支：smc_prefill_only_chunked_prefill_mm_1017_ucx（vllm / vllm-ascend 两仓同分支）
# 用法：
#     边机（1 卡）:  ./start_lwd_smc_ucx.sh edge
#     云机（8 卡）:  ./start_lwd_smc_ucx.sh cloud
#
# 相对老基线 prefill_only_chunked_prefill_mm 的差异（本脚本已按 smc/ucx 分支适配）：
#   1. 拆世界：world/TP = 本侧卡数、PP 恒 1 —— 因此【删除了 -pp / --tensor-parallel-size】；
#   2. 跨机不再有 HCCL 通信域 —— 因此【不再需要覆盖 9 卡世界的 rank table / 白名单】，
#      HCCL 相关配置只服务本机实例内（云 8 卡 TP 扩散）；
#   3. 数据面（边云传 hidden states）改为自研传输层，新增
#      lwd_config.data_plane_family = smc | tcp | verbs | ucx（对应 env
#      VLLM_ASCEND_LWD_DATA_PLANE_FAMILY 可覆盖），默认 smc；
#   4. 控制面端口保持 VLLM_ASCEND_LWD_PRE_OUT_PORT / POST_OUT_PORT（默认 5558/5559）；
#      HELLO 超时默认已提到 1200s，本脚本显式设 1800s；
#   5. 云端 --master-addr 必须填【边侧 IP】（云端 POST_OUT 连 master_addr:post_out_port
#      打回边侧）；torch rendezvous 会被拆世界逻辑自动改写为本机（日志出现
#      "[lwd] rendezvous rewritten to local" 属正常），逃生开关 VLLM_RDZV_FORCE_LOCAL=0。
#
# 依据：assets/vllm-ascend/lwd_edge_cloud_launch_analysis.md（原脚本逐项审计）
#       + 本分支 vllm/vllm/config/vllm.py、vllm/v1/lwd_control/control_edge_scheduler/
#         lwd_edge_assemble.py、vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py
#       + vllm-ascend/vllm_ascend/distributed/lwd_comm/{smc_wire,ucx_npu_wire}.py
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. 角色参数
# -----------------------------------------------------------------------------
ROLE="${1:-}"
if [[ "$ROLE" != "edge" && "$ROLE" != "cloud" ]]; then
    echo "用法: $0 <edge|cloud>"
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. 部署拓扑（按现场改这 8 个）
# -----------------------------------------------------------------------------
EDGE_IP="76.76.7.134"            # 边机管理/控制面 IP
CLOUD_IP="76.76.7.159"           # 云机控制面 IP（PRE_OUT 通告给边侧的可达 IP）
CLOUD_ROCE_IP="$CLOUD_IP"        # 云机 RoCE 网段 IP（数据面 bind/connect 地址；
                                 #   与控制面不同网段时改成 RoCE 网段 IP）
MODEL_DIR="/home/weight/Qwen3.6-27B"                 # 权重目录，两侧同一份
CHAT_TEMPLATE="/home/weight/Qwen3.6-27B/chat_template.jinja"  # 边侧对外服务用
CODEBASE_DIR="/vllm-workspace"                       # 含 vllm / vllm-ascend 两仓的目录
DATA_PLANE_FAMILY="smc"          # 数据面：smc（默认）| tcp | verbs | ucx
UCX_HOME="/opt/ucx-1.22"         # family=ucx 时：UCX 1.22 安装前缀（含 lib/libucp.so）
HTTP_PORT="8314"                 # 边侧对外 HTTP 端口
LOG_DIR="/tmp/lwd-logs"          # 运行日志目录

# 控制面端口（两侧必须同值；避开现场已占用端口）
LWD_PRE_OUT_PORT="7200"          # 云 bind，边来连（边→云 控制面）
LWD_POST_OUT_PORT="7201"         # 边 bind，云来连（云→边 控制面）
LWD_DATA_PLANE_PORT="7202"       # 数据面端口基址：UP=base、DOWN=base+1（smc/tcp），
                                 # verbs/ucx 握手 TCP 走 base+2/base+3
LWD_HELLO_TIMEOUT_S="1800"       # 边等云首拍 HELLO 的预算（要覆盖云侧全量初始化+捕图）

# 云侧 cudagraph 尺寸上限（4 的倍数 = 1+MTP-3；720 只覆盖 180 条序列，
# 想要 max-num-seqs 196 全走图可改成 784，代价是启动捕图更久）
CUDAGRAPH_MAX="720"

# -----------------------------------------------------------------------------
# 2. 代码准备（首次部署执行一次，之后可注释）
# -----------------------------------------------------------------------------
echo ">>> 代码准备检查：两仓都必须在本分支上"
for repo in vllm vllm-ascend; do
    if [[ -d "$CODEBASE_DIR/$repo/.git" ]]; then
        br=$(git -C "$CODEBASE_DIR/$repo" branch --show-current 2>/dev/null || echo "?")
        echo "    $repo 当前分支: $br"
        [[ "$br" == "smc_prefill_only_chunked_prefill_mm_1017_ucx" ]] \
            || echo "    !! $repo 不在目标分支上，请: git -C $CODEBASE_DIR/$repo checkout smc_prefill_only_chunked_prefill_mm_1017_ucx"
    else
        echo "    !! 未找到 $CODEBASE_DIR/$repo"
    fi
done
# 两仓安装（源码树优先）：
#   cd "$CODEBASE_DIR/vllm"        && pip install -e . --no-build-isolation
#   cd "$CODEBASE_DIR/vllm-ascend" && pip install -e . --no-build-isolation
export PYTHONPATH="$CODEBASE_DIR/vllm-ascend:$CODEBASE_DIR/vllm:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# 3. 网络身份 / 设备可见性（按角色）
# -----------------------------------------------------------------------------
export VLLM_HOST_IP="$CLOUD_IP"        # 云侧通告自己
export GLOO_SOCKET_IFNAME="enp189s0f0" # Gloo 走这块网卡（拆世界后仅实例内使用）
export HCCL_SOCKET_IFNAME="enp189s0f0" # HCCL 走这块网卡（拆世界后仅实例内 TP 扩散）
if [[ "$ROLE" == "edge" ]]; then
    export VLLM_HOST_IP="$EDGE_IP"
    export ASCEND_RT_VISIBLE_DEVICES=0          # 必须与 --edge-npu-count 1 一致
    export GLOO_SOCKET_IFNAME="enp125s0f1"
    export HCCL_SOCKET_IFNAME="enp125s0f1"
    export HCCL_IF_BASE_PORT=16657             # 仅边侧设；同机双实例时两侧都设且错开
else
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 必须与 --cloud-npu-count 8 一致
fi

# -----------------------------------------------------------------------------
# 4. 显存 / 进程 / 通信调优（均为 mm 线有效项，无效装饰项已剔除）
# -----------------------------------------------------------------------------
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"  # 降显存碎片
export HCCL_OP_EXPANSION_MODE="AIV"                       # AIV 直接调度 RoCE
export HCCL_BUFFSIZE=1024                                 # 实例内 HCCL 收发缓冲（MB）
export VLLM_WORKER_MULTIPROC_METHOD=spawn                 # worker 用 spawn，避 fork 脏状态
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1                                # torch_npu 异步下发队列
export VLLM_ASCEND_ENABLE_NZ=1                            # NZ 布局：1=只对量化权重
# export VLLM_ASCEND_ENABLE_FUSED_MC2=1                   # 仅 MoE 层有效；Qwen3.6-27B 是 dense
#                                                         # 且开 MTP 时 1 档会被拒 → 不设
if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
    export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}"
fi
export VLLM_LOGGING_LEVEL=INFO

# -----------------------------------------------------------------------------
# 5. LWD 控制面 + 数据面（smc/ucx 分支核心配置）
# -----------------------------------------------------------------------------
export VLLM_ASCEND_LWD_PRE_OUT_PORT="$LWD_PRE_OUT_PORT"
export VLLM_ASCEND_LWD_POST_OUT_PORT="$LWD_POST_OUT_PORT"
export VLLM_ASCEND_LWD_HELLO_TIMEOUT_S="$LWD_HELLO_TIMEOUT_S"
export VLLM_ASCEND_LWD_DATA_PLANE_HOST="$CLOUD_ROCE_IP"   # 云 bind / 边 connect 的 RoCE IP
export VLLM_ASCEND_LWD_DATA_PLANE_PORT="$LWD_DATA_PLANE_PORT"
export VLLM_ASCEND_LWD_DATA_PLANE_FAMILY="$DATA_PLANE_FAMILY"
# 拆世界：rendezvous 默认改写为本机（master_addr 保留控制面寻址语义，不改指本机）
export VLLM_RDZV_FORCE_LOCAL=1

case "$DATA_PLANE_FAMILY" in
    ucx)
        # 部署前置：两端 LD_LIBRARY_PATH 指向 UCX 1.22；hns/ mlx5 TLS 建议按下列注释
        export LD_LIBRARY_PATH="$UCX_HOME/lib:${LD_LIBRARY_PATH:-}"
        export VLLM_ASCEND_LWD_UCX_IMPL=core               # core=新双层实现；legacy=回滚
        if [[ "$ROLE" == "edge" ]]; then
            export UCX_TLS="rc_verbs,tcp,self"             # 边侧 hns（无 UD）
        else
            export UCX_TLS="rc_mlx5,tcp,self"              # 云侧 mlx5（或 rc_verbs,tcp,self）
        fi
        # 可选：export VLLM_ASCEND_LWD_UCX_BLOCK_MB=128   # 默认 128MB×4 块
        #       export VLLM_ASCEND_LWD_UCX_CRC=1          # 默认开帧级 CRC
        ;;
    verbs)
        # 拉远高 RTT 场景；可选：export VLLM_ASCEND_LWD_VERBS_BLOCK_MB=64
        #                       export VERBS_DEV=<设备名>（多网卡指定）
        ;;
    smc|tcp)
        ;;  # smc=内核 AF_SMC（默认）；tcp=排障对照。无额外要求
esac

# -----------------------------------------------------------------------------
# 6. additional_config：lwd_config 段（角色是唯一来源，与 --node-rank 无关）
# -----------------------------------------------------------------------------
LWD_SECTION="{\"enabled\":true,\"role\":\"$ROLE\",\"mode\":\"prefill_only\",\"edge_head_tail_layers\":[0,0],\"pre_out_host\":\"$CLOUD_IP\",\"hello_timeout_s\":$LWD_HELLO_TIMEOUT_S,\"data_plane_host\":\"$CLOUD_ROCE_IP\",\"data_plane_port_base\":$LWD_DATA_PLANE_PORT,\"data_plane_family\":\"$DATA_PLANE_FAMILY\"}"
ADDITIONAL_CONFIG="{\"enable_cpu_binding\":true,\"lwd_config\":$LWD_SECTION}"

mkdir -p "$LOG_DIR"

# -----------------------------------------------------------------------------
# 7. 启动（前台运行，Ctrl+C 可优雅退出；数据面有 atexit 兜底关停）
# -----------------------------------------------------------------------------
if [[ "$ROLE" == "edge" ]]; then
    echo ">>> 拉起边侧（1 卡，node-rank 0）..."
    exec vllm serve "$MODEL_DIR" \
        --host 0.0.0.0 \
        --port "$HTTP_PORT" \
        --master-addr "$EDGE_IP" \
        --master-port 29600 \
        --max-model-len 262144 \
        --served-model-name "qwen3.6-27B" \
        --trust-remote-code --nnodes 2 --node-rank 0 \
        --edge-npu-count 1 --cloud-npu-count 8 \
        --additional-config "$ADDITIONAL_CONFIG" \
        --compilation-config '{"cudagraph_mode":"NONE"}' \
        --max-num-seqs 196 \
        --max-num-batched-tokens 8192 \
        --async-scheduling \
        --gpu-memory-utilization 0.9 \
        --mm_processor_cache_type shm --mm-processor-cache-gb 0 \
        --chat-template "$CHAT_TEMPLATE" \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        2>&1 | tee "$LOG_DIR/vllm_edge.log"
else
    echo ">>> 拉起云侧（8 卡，node-rank 1）..."
    exec vllm serve "$MODEL_DIR" \
        --master-addr "$EDGE_IP" \
        --master-port 29600 \
        --max-model-len 262144 \
        --served-model-name "qwen3.6-27B" \
        --trust-remote-code --nnodes 2 --node-rank 1 \
        --edge-npu-count 1 --cloud-npu-count 8 \
        --additional-config "$ADDITIONAL_CONFIG" \
        --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$(seq -s, 4 4 "$CUDAGRAPH_MAX")]}" \
        --max-num-seqs 196 \
        --max-num-batched-tokens 8192 \
        --async-scheduling \
        --gpu-memory-utilization 0.9 \
        --speculative-config '{"num_speculative_tokens": 3, "method":"qwen3_5_mtp", "enforce_eager": true}' \
        --chat-template "$CHAT_TEMPLATE" \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        2>&1 | tee "$LOG_DIR/vllm_cloud.log"
fi

# -----------------------------------------------------------------------------
# 8. 启动后自检（两侧日志各查一遍）
#    grep -n "\[lwd" $LOG_DIR/vllm_edge.log   | head -30   # 边：POST_OUT bind + 等 HELLO
#    grep -n "\[lwd" $LOG_DIR/vllm_cloud.log  | head -30   # 云：PRE_OUT bind + 首拍 HELLO
#    grep -n "duplex channels ready" $LOG_DIR/vllm_*.log   # 数据面 UP/DOWN 双通道就绪
#    grep -n "rendezvous rewritten to local" $LOG_DIR/vllm_cloud.log  # 拆世界改写（正常）
#
# 帧预算预检提示：若启动日志报 frame 超出块大小（真机实录：5013-token 帧 72MB > 64MB 块），
#   调大 VLLM_ASCEND_LWD_{UCX,VERBS}_BLOCK_MB，或调小 --max-num-batched-tokens。
# -----------------------------------------------------------------------------
