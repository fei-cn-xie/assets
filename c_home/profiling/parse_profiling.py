#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# ms_service_profiler 的 parse 包装器
#
# 为什么需要它
# ------------
# ms_service_profiler 的任务注册表 regist_map 是**全局**的，只在
# `@Task.register(...)` 装饰器被执行时填充，也就是“某个模块被 import 的那一刻”。
# CANN 9.2.1 自带的 parse 入口没有 import ms_service_profiler.data_source /
# .pipeline / .plugins，于是：
#
#   get_register_by_name("data_source:db") -> None
#   get_data_dag()                         -> 空
#   get_dag() 的 head_tasks                -> 空
#   tasks_run() 一次循环都不进              -> 只留下预建的空 profiler.db
#
# 现象就是：`python3 -m ms_service_profiler.parse ...` 秒退、只剩一个
# profiler.db，没有 csv / chrome_tracing.json，也没有任何报错。
#
# 本脚本先 import 这三个包把注册表补满，再走与 CLI 完全相同的解析路径，
# 因此产物与正常版本一致（request.csv / batch.csv / kvcache.csv /
# request_status.csv / chrome_tracing.json / profiler.db）。
#
# 用法
# ----
#   python3 parse_profiling.py --input-path <采集目录> --output-path <输出目录>
#
# 采集目录 = prof_dir 下的时间戳目录（内含 Trace_Service/*.db）
# 输出目录会被创建；同一目录重复解析会覆盖同名产物。
#
# 只检查环境与注册状态：
#   python3 parse_profiling.py --input-path <采集目录> --check-only
# ---------------------------------------------------------------------------
import argparse
import os
import sys

PRELOAD_PACKAGES = (
    # data_source:db / :msprof / :mspti / :torch_profiler
    "ms_service_profiler.data_source",
    # pipeline:* 任务（把 data_source:service 转成 tx_data_df 等）
    "ms_service_profiler.pipeline",
    # plugin:* 任务
    "ms_service_profiler.plugins",
)

OUTPUT_HINTS = {
    "request.csv": "请求级数据（TTFT 在这里）",
    "request_status.csv": "请求状态变化",
    "batch.csv": "组批调度",
    "kvcache.csv": "KV cache",
    "chrome_tracing.json": "可用 MindStudio Insight 打开",
    "profiler.db": "合并后的 sqlite 库",
}


def die(message):
    print(f"\n[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


def preload_packages():
    """Import the packages whose decorators fill the task registry."""
    failed = []
    for name in PRELOAD_PACKAGES:
        try:
            __import__(name)
        except Exception as exc:                                    # noqa: BLE001
            failed.append(f"{name}: {exc!r}")
        else:
            print(f"[preload] ok   {name}")
    for line in failed:
        print(f"[preload] FAIL {line}")
    return failed


def print_registry():
    from ms_service_profiler.task.task_register import (           # noqa: PLC0415
        get_data_dag,
        get_register_by_name,
    )

    registered = get_register_by_name("data_source:db")
    flow, _ = get_data_dag()
    producers = flow.get("data_source:service", {}).get("from_task_name")
    consumers = flow.get("prof_path", {}).get("to_task_name")
    print(f"[registry] data_source:db      -> {registered}")
    print(f"[registry] service producers   -> {producers}")
    print(f"[registry] prof_path consumers -> {consumers}")
    return registered


def call_parse(argv):
    """Invoke the profiler's own parse entry with an already-populated registry."""
    from ms_service_profiler import parse as parse_module      # noqa: PLC0415

    if hasattr(parse_module, "_setup_parser_arguments"):
        parser = argparse.ArgumentParser(prog="ms_service_profiler parse")
        parse_module._setup_parser_arguments(parser)
        return parse_module.main(parser.parse_args(argv))

    # 兜底：某些版本只暴露 subcommand 入口
    from ms_service_profiler.cli import create_subcommand_parser, run_parser  # noqa: PLC0415

    parser, subparsers = create_subcommand_parser()
    parse_module.arg_parse(subparsers)
    saved = sys.argv
    sys.argv = ["msserviceprofiler", "parse", *argv]
    try:
        return run_parser(parser)
    finally:
        sys.argv = saved


def report_outputs(path):
    if not os.path.isdir(path):
        print(f"[output] 目录不存在: {path}")
        return
    print(f"[output] {path}")
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        size = os.path.getsize(full)
        hint = f"  # {OUTPUT_HINTS[name]}" if name in OUTPUT_HINTS else ""
        print(f"         {size:>10,d}  {name}{hint}")
    csvs = [n for n in os.listdir(path) if n.endswith(".csv")]
    if not csvs:
        print("[warn] 没有生成任何 csv：多半是采集期间请求太少或时长太短，"
              "重新采集时至少跑够 2 分钟、几百个请求")


def main():
    parser = argparse.ArgumentParser(
        description="解析 ms_service_profiler 采集数据（自动补齐任务注册表）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python3 parse_profiling.py -i /bash_dir/profiling/prof_dir/1128-0332 "
               "-o /bash_dir/profiling/prof_dir/output/A\n",
    )
    parser.add_argument("-i", "--input-path", required=True,
                        help="采集目录（prof_dir 下的时间戳目录，内含 Trace_Service/*.db）")
    parser.add_argument("-o", "--output-path", default=None,
                        help="解析产物输出目录（--check-only 时可省略）")
    parser.add_argument("-l", "--log-level", default="info",
                        choices=["debug", "info", "warning", "error", "fatal", "critical"],
                        help="透传给 profiler 的日志级别（默认 info）")
    parser.add_argument("-f", "--format", nargs="+", default=None,
                        choices=["db", "csv", "json"],
                        help="透传给 profiler 的输出格式（默认 db csv json 全出）")
    parser.add_argument("--check-only", action="store_true",
                        help="只检查环境与任务注册状态，不做解析")
    args = parser.parse_args()

    try:
        import ms_service_profiler                                  # noqa: F401,PLC0415
    except ImportError as exc:
        die(f"ms_service_profiler 不可导入（{exc}）；确认它在 python3 的 sys.path 里，"
            "或在容器内使用 CANN 自带的 site-packages")
    print(f"[env] ms_service_profiler @ {ms_service_profiler.__file__}")

    if not os.path.isdir(args.input_path):
        die(f"采集目录不存在: {args.input_path}")

    failed = preload_packages()
    registered = print_registry()

    if registered is None:
        die("数据源任务未注册，解析只会产出空的 profiler.db。\n"
            "      若上面三个包都是 ok，说明该版本的 db_data_source.py 没有挂\n"
            "      @Task.register('data_source:db')，改用可用版本：\n"
            "      export PYTHONPATH=<可用的 msserviceprofiler 目录>:$PYTHONPATH")

    if failed:
        print("[warn] 有包导入失败，注册表可能不完整，结果仅供参考", file=sys.stderr)
    if args.check_only:
        print("[check] 注册表正常，可以解析")
        return
    if not args.output_path:
        die("缺少 --output-path")

    argv = [f"--input-path={args.input_path}",
            f"--output-path={args.output_path}",
            f"--log-level={args.log_level}"]
    if args.format:
        argv.append("--format=" + " ".join(args.format))

    print(f"\n[parse] input={args.input_path}")
    print(f"[parse] output={args.output_path}\n")
    call_parse(argv)
    report_outputs(args.output_path)


if __name__ == "__main__":
    main()
