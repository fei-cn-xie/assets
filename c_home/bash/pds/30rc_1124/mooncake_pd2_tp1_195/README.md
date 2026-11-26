# mooncake_pd2_tp1_195：集中式 PD 分离，两侧都是 2 DP × 1 TP

**集中式 = 没有边端、没有 LWD 控制面**：prefill 与 decode 是两个独立的 `vllm serve`，
KV 通过 Mooncake 连接器点对点传输，前面挂上游的负载均衡 proxy。

```
client ──9060──▶ LB proxy (195, CPU)
                    ├── prefill 9160 (195 NPU 0,1)  DP=2 × TP=1  kv_producer base 30000
                    └── decode  9161 (195 NPU 2,3)  DP=2 × TP=1  kv_consumer base 30100
```

目录名 = `{连接器}_pd{每侧DP数}_tp{每副本TP数}_{主机}`。

## 与同目录其它包的关系

| 包 | 形态 | 并行 | 用途 |
| --- | --- | --- | --- |
| `../mooncake_pd_195` | 集中式 | DP=1 × TP=2 | 已确认可用的基线（TP 切分，单副本） |
| `../dp2_tp1_194_195` | 边云 LWD | 边端 194 + 云 DP=2 × TP=1 | 边云场景；云端多副本 |
| **本包** | **集中式** | **DP=2 × TP=1** | **在无 LWD 干扰的环境下验证多 DP** |

本包是验证「dense 模型 + 多 DP」的最小场景：没有 LWD 拓扑、没有边云控制面，
出问题只可能来自 DP 本身，所以拿它做修复前后的对照最干净。

## 卡与端口

| 角色 | 卡 | 并行 | API | KV 基址（实际监听） | DP 协调 rpc |
| --- | --- | --- | --- | --- | --- |
| proxy | — | — | **9060** | — | — |
| prefill | NPU 0,1 | **DP=2 × TP=1** | 9160 | 30000（→ 30000 / 30001） | 29600 |
| decode | NPU 2,3 | **DP=2 × TP=1** | 9161 | 30100（consumer 只连不绑） | 29601 |

`--data-parallel-size 2` 时 vLLM 默认让 `api_server_count = data_parallel_size`，
即每个引擎有两个 API server 进程共享上述端口，内部做 LB；proxy 每侧仍只需一个端点。

## 文件

| 文件 | 作用 |
| --- | --- |
| `env.sh` | 运行时环境（NIC / HCCL_IF_IP / mooncake site-packages / 内存与端口常量），支持 `--dry-run` |
| `prefill.sh` | P 引擎（`kv_producer`，`--enforce-eager`） |
| `decode.sh` | D 引擎（`kv_consumer`，`FULL_DECODE_ONLY`） |
| `start_proxy.sh` | 调用 vllm-ascend 自带 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`，仓内无 Python 胶水 |
| `check_configs.py` | 静态自检（见文末） |

## 为什么是 DP=2 × TP=1，以及内存账

TP=1 意味着**每个副本把完整 64 层模型装进一张卡**（50.2 GB ≈ 46.8 GiB），
所以 64 GB 卡的 KV 预算很紧：

| gpu-memory-utilization | 权重 | 剩给 KV | 影响 |
| --- | --- | --- | --- |
| 0.90 | 46.8 GiB | ≈3.3 GiB | 65536 上下文直接失败 |
| **0.95（本包）** | 46.8 GiB | ≈5~6.7 GiB | 32768 上下文有余量 |

模型含 GDN(Mamba) 层，每个 decode 序列占一个 Mamba block；开 MTP 后 KV 池只够
~32~63 个 block，**实测 64 就起不来**，因此 `PD_MAX_NUM_SEQS=32`、`PD_MAX_MODEL_LEN=32768`。

## 启动

```bash
python3 check_configs.py        # 静态自检

bash prefill.sh                 # 先 P
bash decode.sh                  # 再 D
bash start_proxy.sh             # 最后 proxy
```

等两个引擎都打印完初始化日志（`Application startup complete`）再打请求：

```bash
curl -s http://76.76.26.195:9060/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

可覆盖的调参项（导出后再启动，两个引擎都读同一份默认值）：

```bash
PD_MAX_MODEL_LEN=32768 PD_MAX_NUM_SEQS=32 PD_GPU_MEM_UTIL=0.95 \
PD_ENABLE_MTP=0 bash prefill.sh
```

## 双分支对照测试：应该看到什么

`PD_ENABLE_MTP` 默认 **1**（与已确认可用的参考一致）。两种失败签名对应两处不同的修复：

| 分支 | MTP | 预期首个失败点 | 归属修复 |
| --- | --- | --- | --- |
| `feat/pd-sep-lwd-v0.30.0` | **开** | P/D 在 worker `load_model` → `self.drafter.load_model` → `_create_draft_vllm_config` → `replace()` → `ValueError: KV transfer 'prefill' config has a conflicting data parallel size. Expected 1, but got 2.` | `data_parallel_size_global` + `check_kv_extra_config` |
| `feat/pd-sep-lwd-v0.30.0` | **关**（`PD_ENABLE_MTP=0`） | P 在 `register_kv_caches` → `ZMQError: Address already in use (…:30000)` → `RuntimeError: KV Cache sending/receiving thread failed to start.`；D 在握手后 `Expected 1, but got 2` | `side_channel_port` 用 `data_parallel_index` |
| `fix/lwd-multidp-dense-pd` | 开或关 | 正常拉起，两个引擎各 2 个 EngineCore，KV 端口 30000/30001 都被绑上 | — |

也就是说：**这一个包就能同时验证三处修复**（MTP 开验证前两处，MTP 关验证第三处），
而且完全不依赖 LWD 代码路径。

## 如何确认 DP=2 真的生效

`curl` 能通并不能说明是 DP=2（`DP=1 × TP=2` 也一样能应答），要看这些**两者必然不同**的量：

| 观测量 | DP=2 × TP=1 | DP=1 × TP=2 | 命令 |
| --- | --- | --- | --- |
| EngineCore 进程 | **2 个**（`EngineCore_DP0/DP1`） | 1 个（`EngineCore`） | `ps -ef \| grep '[E]ngineCore'` |
| 每 rank `world_size` | **1** | **2** | `grep -o 'world_size=[0-9]* rank=[0-9]*' <log> \| sort -u` |
| 每卡权重 | ≈46.8 GiB | ≈23.4 GiB | `npu-smi info` |
| KV 监听端口归属 | 30000、30001 分属**两个不同进程** | 同属一个进程 | `ss -ltnp \| grep -E ':3000[01]'` |
| API server 进程 | 2（`api_server_count=DP`） | 1 | `ss -ltnp \| grep :9160` |

⚠️ 不要用引擎配置行判断：`Initializing a V1 LLM engine … data_parallel_size=1`
在真·DP=2 的 dense 部署里**也会打印 1**（DP 在进程内被坍缩成 1），这是本包要验证的 bug 的另一面。

## `check_configs.py` 检查什么

```bash
python3 check_configs.py            # 本目录
python3 check_configs.py --dir ...  # 其它同构包
```

- 两侧都是 DP=2 × TP=1，且参数取自 `env.sh`（不是各写一份字面量）
- 两张卡的集合不重叠、总数 = DP × TP = 4
- 所有监听端口互不冲突，含 **KV 握手端口区间**（`kv_port + dp_rank` 覆盖到的每个端口）
- KV 连接器角色（producer/consumer）、`engine_id` 不同、`extra_config` 的 dp/tp 与并行度一致
- 两侧必须一致的三个 knobs（`max_model_len` / `max_num_seqs` / `max_num_batched_tokens`）都被两个脚本引用
- `max_num_seqs ≤ 32`（TP=1 + MTP 下的 Mamba block 实测上限）

反向验证过：把 decode 的卡改成 0,1、把 KV 基址改成 30001、把 TP 改成 2、把两侧 knob 写成字面量，
四种错误配置都会被拦下并指名文件与值。
