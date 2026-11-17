# PD 性能日志分析脚本

用于分析边云 PD、集中式 PD，以及两种架构的统一阶段对比。只需要 `openpyxl`，不需要修改启动脚本或设置环境变量。

## 1. 最简用法

在 `assets/vllm-ascend/profiling` 目录执行：

```bash
# 边云 PD：生成 edge_pd_analysis.xlsx
python analyze_pd_logs.py edge --log-dir <边云日志目录>

# 集中式 PD：生成 central_pd_analysis.xlsx
python analyze_pd_logs.py central --log-dir <集中式日志目录>

# 两种架构对比：生成边云、集中式和统一对比表
python analyze_pd_logs.py compare \
  --edge-dir <边云日志目录> \
  --central-dir <集中式日志目录>
```

默认输出到**当前目录**。建议每组实验使用单独输出目录：

```bash
python analyze_pd_logs.py compare \
  --edge-dir /tmp/run-001-edge \
  --central-dir /tmp/run-001-central \
  --output-dir /tmp/run-001-report
```

## 2. 使用当前工作区示例

在工作区根目录执行：

```bash
python assets/vllm-ascend/profiling/analyze_pd_logs.py edge \
  --log-dir private_home/lwd_ec_pd_perf_debug/1117-1020 \
  --output-dir /tmp/edge-report

python assets/vllm-ascend/profiling/analyze_pd_logs.py central \
  --log-dir private_home/lwd_ec_pd_perf_debug/1117-1055 \
  --output-dir /tmp/central-report

python assets/vllm-ascend/profiling/analyze_pd_logs.py compare \
  --edge-dir private_home/lwd_ec_pd_perf_debug/1117-1020 \
  --central-dir private_home/lwd_ec_pd_perf_debug/1117-1055 \
  --output-dir /tmp/pd-comparison
```

## 3. 多请求日志怎么指定请求

默认要求日志目录中只有一组目标请求。存在多个请求时，请指定 ID：

```bash
# 边云单个分析
python analyze_pd_logs.py edge \
  --log-dir /tmp/edge-logs \
  --request-id chatcmpl-xxx

# 集中式单个分析：可传完整 P request_id，也可传公共前缀
python analyze_pd_logs.py central \
  --log-dir /tmp/central-logs \
  --request-id chatcmpl-xxx

# 对比时分别指定
python analyze_pd_logs.py compare \
  --edge-dir /tmp/edge-logs \
  --central-dir /tmp/central-logs \
  --edge-request-id chatcmpl-edge-xxx \
  --central-request-id chatcmpl-central-xxx
```

集中式 P/D 的 request_id 可能不同，脚本会按公共前缀及 KV transfer 日志关联。旧版日志中 `model-forward` 没有 `req_ids` 时，建议一次只采集一个请求，避免不同请求的执行批次混在一起。

## 4. 输出文件

| 命令 | 文件 | 内容 |
|---|---|---|
| `edge` | `edge_pd_analysis.xlsx` | 请求总览、边云主阶段、UP、Decode 逐步、统计与代码地图 |
| `central` | `central_pd_analysis.xlsx` | P/D 阶段、D 逐步 NPU、统计与代码地图 |
| `compare` | `edge_pd_analysis.xlsx` | 边云详细分析 |
| `compare` | `central_pd_analysis.xlsx` | 集中式详细分析 |
| `compare` | `pd_comparison.xlsx` | 单 sheet 阶段对比、共同阶段、架构特有阶段、Decode 统计、代码地图 |

统一对比表包含：

```text
序号
阶段
耗时-集中式(ms)
耗时-边云PD(ms)
time.monotonic()-集中式
time.monotonic()-边云PD
代码位置
父阶段
父阶段序号
```

集中式没有对应阶段时填 `"-"`，不填 0；均值或分位数统计不伪造单点 mono。

## 5. 其他选项

```bash
# 指定输出目录
--output-dir /tmp/report

# 打印脚本执行过程
--verbose
```

依赖检查：

```bash
python -c "import openpyxl; print(openpyxl.__version__)"
```

如果未安装：

```bash
pip install openpyxl
```

## 6. 数据口径注意事项

- `mono` 是各实例自己的 `time.monotonic()`，不要直接跨主机相减。
- P/D 跨实例排序与交接耗时使用 `ts_wall`。
- `elapsed_ms/total_ms` 优先使用日志原值；表格派生值只在同一单调钟或明确公式下计算。
- `model-forward.npu_ms` 和 `embedding.npu_ms` 是批级同步墙钟，不是单请求纯 kernel 时间。
- sidecar 的 queue/clone/D2H/serialize/network/rx_wait/H2D 是嵌套子阶段，不能与外层阶段全部相加。
- `torch.npu.synchronize()` 会改变异步流水，做架构性能比较时应同时评估打点开销。
- 新日志中有 `completion_tokens` 时才计算 TPOT；旧日志缺失时保留 `"-"` 或“不可计算”。
