# in_process：1513 的"非 sidecar"变体（同一套参数）

本目录 = [../1513](../1513/README.md) 的**逐字副本**，只改拓扑里的一个字段：

| 位置 | ../1513 | 本目录 |
| --- | --- | --- |
| `deployment.data_plane_host.deployment` | `sidecar` | **`in_process`** |

其余全部相同：4 个入口脚本、`env.sh`、`pd_proxy.json`、`check_configs.py` 与 1513 **逐字节一致**；
拓扑只有这 1 行不同（地址 `76.76.26.194`/`195`、NPU `4` / `0,1` / `2,3`、端口
`8510/8600/8601/8102`、`coord_port 8100/8101`、`ctrl_port 6100/6101`、`data_port 7200/7201`、
`kv_port 14579/14580`、`up/down` 缓冲区、UCX 调优、模型路径全都不变）。

> 两端的 `deployment` 必须一致。**不要**一边用 1513（sidecar）、一边用本目录（in_process）。

## 两者的实际差别

| 维度 | sidecar（1513） | in_process（本目录） |
| --- | --- | --- |
| 传输进程 | 每机一个 `ucx_daemon`，由首个引擎进程 `Popen` 拉起 | 无独立进程，传输跑在引擎进程内 |
| 云端 listener | daemon 持有 | **引擎自己**在 `data_port` 上 bind |
| 引擎↔传输 | ctl unix socket `/tmp/lwd-host-comm-<uid>.sock` + memfd + SPSC ring | 直接函数调用 |
| 容器要求 | 需要 `memfd_create`、可写 `/tmp` | **不需要**这两个 |
| 可观察性 | 多一个 `ps -ef \| grep ucx_daemon` | 只有一个引擎进程，少一处观测点 |
| 生命周期 | 传输与模型进程解耦 | 传输与引擎同生共死 |

`in_process` 是这条分支的**代码默认值**；`sidecar` 是 v30_rc 在 194-195 上验证过的值。
两者走的是同一套 UHC1 通道实现（`ucx_channel.py`/`ucx_link.py`），业务面完全等价。

## 什么时候用哪个

- **想要最省心、已验证**：用 `../1513`（sidecar）。
- **容器禁 `memfd_create`、`/tmp` 只读、或不想多一个守护进程**：用本目录。
- 想压测/对比两种模式的差异：本目录与 1513 仅差一个字段，可直接对照。

## 启动（与 1513 完全相同）

把整个目录拷到两台机器的同一路径，**两端用同一份 `pd_host_topology.yaml`**。

```bash
python3 check_configs.py        # 可选自检

# 云端 195
bash cloud_p.sh
bash cloud_d.sh
bash start_proxy.sh

# 边端 194（最后起；只有边端会阻塞等链路，605s/条 IO）
bash edge.sh
```

curl 自测（打边端 8510）：

```bash
curl -s http://76.76.26.194:8510/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
```

## 运行时的判断依据

- **主机上不应再有 daemon**：`ps -ef | grep ucx_daemon` 应无输出，`ls /tmp/lwd-host-comm-*.sock` 应为空。
- 云端日志应出现（`lwd_host_channels.py:311`、`:371`）：

  ```
  [LWD] host listener bound for cloud=0 dp_idx=0 on 76.76.26.195:7200
  [LWD] host cloud listener ready; waiting for N edge link(s) asynchronously
  ```

  （sidecar 模式下这一行换成 `[LWD] host sidecar deployment: listeners owned by the transport daemon (ports [...])`。）
- 其余排障照旧：三端 `[LWD][config] digest=` 必须一致；缺 `libucp` 会在启动前被预检拦下（`exit 3`）；
  边端静默等待最长 605s/条链路。完整清单见 [../1513/README.md](../1513/README.md) §6。

## 本目录做过的校验

| 检查 | 方法 | 结果 |
| --- | --- | --- |
| 脚本与配置未被改动 | 与 `../1513` 逐文件哈希比对 | 8 个文件全等（只有 `README.md`、`pd_host_topology.yaml` 按需不同） |
| 拓扑只差一个字段 | 结构化对比两目录的 `pd_host_topology.yaml` | 仅 `deployment.data_plane_host.deployment` 由 `sidecar` → `in_process`，其余键值全等 |
| 拓扑合法 | `check_configs.py`（按 feat 的 `topology.py` 规则） | PASS；host 内存估算 2304 MiB/rank（预算 12288） |
| 脚本语法 | `bash -n` × 5 | 全通过 |
| 入口可用 | 四个入口 `--dry-run` | 全通过，命令行与环境变量与 1513 一致 |
