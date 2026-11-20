# DeepSeek-V4 集中式 PD：启动与请求验证包

日期：2026-11-20

本目录用于验证 `MooncakeConnectorV1` 集中式 PD 的两项修复：

1. **KV cache 注册基址**：hybrid 视图可能位于共享 arena 的非零 offset，注册时必须恢复 2 MiB 对齐的 backing 基址，不能使用视图地址。
2. **按 KV group 传递 block size**：DSV4 的各组 block size 不同，P scheduler 不能把 EngineCore 计算出的全局最小值（本次为 8）作为所有组的远端块大小。

代码仓：`vllm-ascend`，分支 `fix/pd-sep-hybrid-kv-register`。两台 P/D 机器都必须更新到同一修复分支；`vllm` 仓不需要修改。

## 目录

```
fix_1120/
├── prefill.sh            # 188 Prefill，端口 9730
├── decode.sh             # 187 Decode，端口 9740
├── proxy.sh              # 187 Proxy，端口 8108
├── check.sh              # 启动日志与 /health 自检
├── smoke.sh              # 通过 proxy 发真实请求并校验响应
└── selftest_offline.sh   # 脚本逻辑离线自测，不占 NPU
```

启动脚本中的 vLLM 参数与原验证脚本保持一致。新增预检会在代码缺少任一修复时立即退出，并打印当前分支/提交和修复命令，避免浪费模型加载时间。

## 1. 更新代码（188、187 都执行）

代码远端是 **`pds`**（`git@gitcode.com:xiefei_cn/vllm-ascend_pds.git`）：

```bash
cd /vllm-workspace/vllm-ascend
git fetch pds fix/pd-sep-hybrid-kv-register
git checkout fix/pd-sep-hybrid-kv-register
git reset --hard pds/fix/pd-sep-hybrid-kv-register
git log -1 --oneline            # 期望 92127dd08c

grep -n "has_mamba_group or self.use_hybrid" \
  vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
grep -n "remote_block_sizes=\[group.kv_cache_spec.block_size" \
  vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
```

两条 grep 都必须有输出。启动脚本会做同样的检查，缺哪项会直接告诉你，并打印对齐命令。

> 若你的机器上远端不叫 `pds`，启动脚本支持 `CODE_REMOTE=<远端名>` 覆盖。

同时更新 assets 仓，并进入本目录：

```bash
cd <assets仓>
git pull
cd c_home/bash/pds/fix_1120
```

> P 负责发布 `remote_block_sizes`，D 负责消费。只更新一侧仍会失败，必须两侧一起更新。

## 2. 可选：真实 Python 单测（不占 NPU）

在具有项目测试环境的机器上：

```bash
cd /vllm-workspace/vllm-ascend
python -m pytest -sv \
  tests/ut/distributed/kv_transfer/kv_p2p/test_mooncake_connector.py \
  -k "per_group_remote_block_sizes or sliding_uses_missing_window_suffix or request_finished_trims_mtp_blocks_in_params or grouped_metadata_preserves_prefix_and_topology_contract or registered_hybrid_buffer_recovers_arena_base_behind_descriptor_offset"
```

关键用例 `test_get_kernel_block_ids_uses_per_group_remote_block_sizes` 直接使用本次日志中的六组几何 `[512,16384,128,128,8,32]`。旧代码会在第一组触发 `remote_block_size(8) not divisible by kernel_size(128)`，新代码应按组返回一致的 local/remote block IDs。

`test_get_kernel_block_ids_sliding_uses_missing_window_suffix` 锁定非零 prefix 下的窗口尾块选择：P 已把 sliding 组裁成窗口尾部，D 不能再按全 prompt 绝对偏移切一次，否则会把需要的尾块切空、静默不传。

## 3. 可选：脚本离线自测

```bash
bash selftest_offline.sh
```

末行应为 `FAIL=0`。该自测只验证脚本逻辑，不能替代 Python 单测和 NPU 验证。

## 4. 启动 P/D/Proxy

先启动 Prefill（188）：

```bash
bash prefill.sh
# 等待模型加载，通常数分钟
bash check.sh prefill
```

再启动 Decode（187）：

```bash
bash decode.sh
bash check.sh decode
```

两侧 `check.sh` 都应返回 `[PASS]`，并且日志中有：

```text
Mooncake registering 1 KV cache buffer region(s), ...
```

然后启动 Proxy（187）：

```bash
bash proxy.sh
```

## 5. 发真实 PD 请求

```bash
bash smoke.sh
```

通过标准：

- 输出 `[PASS] PD 请求成功`；
- 返回文本非空；
- Decode 不再出现 `remote_block_size(8) not divisible by kernel_size(128)`；
- Decode 能看到 KV transfer 完成日志，Prefill 最终能延迟释放该请求块。

默认请求使用模型名 `dsv4`。此前日志中的 `deepseek-v4-flash does not exist` 是 curl 模型名写错导致的 404，与引擎无关。

也可以手工执行：

```bash
curl -s http://127.0.0.1:8108/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"用一句话说明什么是 KV cache"}],"max_tokens":64,"temperature":0}'
```

## 6. 失败时收集

```bash
bash check.sh prefill > prefill-check.txt 2>&1 || true
bash check.sh decode > decode-check.txt 2>&1 || true
grep -nE "not aligned|not divisible|EngineDeadError|Worker failed with error|Traceback" \
  logs/*/*.log | tail -80
```

将上述输出、最新的 P/D 日志以及 curl 响应一起提供。注意 `check.sh` 的进程和致命错误判定用于启动及请求后检查；仅 `/health=200` 不代表 PD 请求一定成功，因此必须再执行 `smoke.sh`。
