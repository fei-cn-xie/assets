#!/usr/bin/env python3
"""把一个完整的 HF safetensors checkpoint 按 LWD 角色(edge/cloud)切成两个子集。

背景:边云 LWD 的每一侧只运行一部分模块(配套 model_load_update 分支
里的 ``patch_lwd_module_scope`` + ``lwd_weight_filter``),所以每侧的模型
目录只需要放自己会用到的张量。但 HF 的 15 个分片是**按张量交错打包**的,
不能"整片给某一侧",必须按张量名拆分。

用法(先 dry-run 看清单,再真拆):

    # 只看规则会保留什么(不写文件)
    python split_lwd_checkpoint.py --src /home/extra/Qwen3.6-27B \
        --role edge --dry-run

    # 真正拆分到 /home/extra/Qwen3.6-27B-edge
    python split_lwd_checkpoint.py --src /home/extra/Qwen3.6-27B \
        --dst /home/extra/Qwen3.6-27B-edge --role edge

    # 云侧(几乎不省:云侧仍需要 embed/norm/lm_head + 全部 transformer 层)
    python split_lwd_checkpoint.py --src /home/extra/Qwen3.6-27B \
        --dst /home/extra/Qwen3.6-27B-cloud --role cloud

规则(可用 --keep-prefix 覆盖/追加,默认同时兼容 VL 与纯文本两种命名):

  edge  : 顶层 lm_head.* + model.language_model.embed_tokens.* + model.visual.*
          (边侧只做 embed / unembed;transformer 层与 final norm 都不建)
  cloud : 顶层 lm_head.* + model.language_model.*(全部:embed/layers/norm)
          + mtp.* / model.mtp.*(开启投机解码时)
          (云侧不建视觉塔 visual)

非权重文件(所有 *.json / tokenizer / *.jinja / preprocessor 配置)默认整份
拷贝:总量只有几十 MB,而且 VL 模型类在**构造期**就会建 tokenizer
(vllm/model_executor/models/qwen3_vl.py 的 cached_tokenizer_from_config),
两侧都不能省。仅跳过 .git*、README.md、LICENSE。

输出:按 --max-shard-gb 切成若干 model-XXXXX-of-YYYYY.safetensors;多于一个
分片时写新的 model.safetensors.index.json,只有一个分片则不写(避免留下会
过滤掉自建分片的旧索引)。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

INDEX_NAME = "model.safetensors.index.json"

# 每侧必须保留的 checkpoint 名前缀。
ROLE_RULES: dict[str, tuple[str, ...]] = {
    "edge": (
        "lm_head.",  # 边侧 rank-replay 还原 token
        "model.language_model.embed_tokens.",  # VL 布局:文本嵌入
        "model.embed_tokens.",  # 纯文本布局:文本嵌入
        "model.visual.",  # 视觉塔(多模态必须)
    ),
    "cloud": (
        "lm_head.",  # 云侧采样 + 算秩
        "model.language_model.",  # VL 布局:embed/layers/norm
        "model.layers.",  # 纯文本布局:transformer 层
        "model.norm.",  # 纯文本布局:final norm
        "model.embed_tokens.",  # 纯文本布局:文本嵌入(decode 行本地嵌入)
        "mtp.",  # 独立 MTP 权重(若开投机解码)
        "model.mtp.",
    ),
}

SKIP_FILE_EXACT = {"README.md", "LICENSE", ".gitattributes"}
SKIP_FILE_PREFIX = (".git", INDEX_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="完整模型目录(含 *.safetensors)")
    parser.add_argument("--dst", help="输出目录(与 --src 不同;--dry-run 时可省略)")
    parser.add_argument("--role", required=True, choices=sorted(ROLE_RULES))
    parser.add_argument("--max-shard-gb", type=float, default=5.0, help="输出分片上限(默认 5GB;内存紧张就调小)")
    parser.add_argument("--keep-prefix", action="append", default=[], help="追加保留前缀(可重复)")
    parser.add_argument("--dry-run", action="store_true", help="只打印将保留/丢弃的清单,不写文件")
    return parser.parse_args()


def load_weight_map(src: str) -> dict[str, str]:
    index_path = os.path.join(src, INDEX_NAME)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            return json.load(handle)["weight_map"]
    # 无索引:单分片或自建分片场景,退化为只按文件名列表
    weight_map: dict[str, str] = {}
    for name in sorted(os.listdir(src)):
        if name.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(os.path.join(src, name), framework="pt") as handle:
                for key in handle.keys():
                    weight_map[key] = name
    return weight_map


def select_keys(weight_map: dict[str, str], prefixes: tuple[str, ...]) -> tuple[dict[str, list[str]], int]:
    """按前缀归组要保留的张量;返回 {前缀: [张量名]} 与命中总数。"""
    grouped: dict[str, list[str]] = defaultdict(list)
    matched = 0
    for key in weight_map:
        for prefix in prefixes:
            if key.startswith(prefix):
                grouped[prefix].append(key)
                matched += 1
                break
    return grouped, matched


def report(weight_map: dict[str, str], src: str, grouped: dict[str, list[str]], prefixes: tuple[str, ...]) -> None:
    def gib(num_bytes: int) -> float:
        return num_bytes / 2**30

    per_shard_bytes = {name: os.path.getsize(os.path.join(src, name)) for name in set(weight_map.values())}
    total_bytes = sum(per_shard_bytes.values())

    print(f"[in ] {len(weight_map)} tensor(s), {gib(total_bytes):.2f} GiB on disk")
    print(f"[rule] keep prefixes: {list(prefixes)}")
    print("\n  matched tensors per prefix:")
    kept = 0
    for prefix in prefixes:
        keys = grouped.get(prefix, [])
        if not keys:
            print(f"    {prefix:40s} 0 tensor(s)  <-- 没命中,检查命名布局/是否该保留")
            continue
        kept += len(keys)
        print(f"    {prefix:40s} {len(keys)} tensor(s)")
    print(f"\n  => keep {kept} tensor(s) of {len(weight_map)}; drop {len(weight_map) - kept}")

    # 被丢弃的张量按"前 3 段"聚一下,便于确认丢的都是层权重
    dropped_groups: dict[str, int] = defaultdict(int)
    keep = {key for keys in grouped.values() for key in keys}
    for key in weight_map:
        if key in keep:
            continue
        dropped_groups[".".join(key.split(".")[:3])] += 1
    print("\n  dropped tensors grouped by leading components (top 15):")
    for name, count in sorted(dropped_groups.items(), key=lambda item: -item[1])[:15]:
        print(f"    {name:48s} {count}")


def copy_metadata(src: str, dst: str) -> list[str]:
    copied = []
    for name in sorted(os.listdir(src)):
        if name.endswith(".safetensors") or name.startswith(SKIP_FILE_PREFIX):
            continue
        if name in SKIP_FILE_EXACT:
            continue
        source_path = os.path.join(src, name)
        if not os.path.isfile(source_path):
            continue
        shutil.copy2(source_path, os.path.join(dst, name))
        copied.append(name)
    return copied


def write_slices(src: str, dst: str, weight_map: dict[str, str], keep: set[str], max_shard_bytes: int) -> list[str]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    # 按源分片归组,顺序读取以控制内存峰值
    want_per_shard: dict[str, list[str]] = defaultdict(list)
    for key in keep:
        want_per_shard[weight_map[key]].append(key)

    written: list[str] = []
    buffer: dict[str, object] = {}
    buffer_bytes = 0

    def flush() -> None:
        nonlocal buffer, buffer_bytes
        if not buffer:
            return
        name = f"model-{len(written) + 1:05d}-of-XXXXX.safetensors"
        save_file(buffer, os.path.join(dst, name), metadata={"format": "pt"})
        written.append(name)
        buffer = {}
        buffer_bytes = 0

    for shard in sorted(want_per_shard):
        with safe_open(os.path.join(src, shard), framework="pt") as handle:
            for key in want_per_shard[shard]:
                tensor = handle.get_tensor(key)
                buffer[key] = tensor
                buffer_bytes += tensor.numel() * tensor.element_size()
                del tensor
                if buffer_bytes >= max_shard_bytes:
                    flush()
    flush()

    # 分片名要带总数,先按最终数量重命名
    total = len(written)
    final_names = []
    for index, name in enumerate(written, start=1):
        final_name = f"model-{index:05d}-of-{total:05d}.safetensors"
        os.replace(os.path.join(dst, name), os.path.join(dst, final_name))
        final_names.append(final_name)
    return final_names


def write_index(dst: str, weight_map: dict[str, str], keep: set[str], shard_names: list[str]) -> None:
    index_path = os.path.join(dst, INDEX_NAME)
    if len(shard_names) <= 1:
        # 单分片不需要索引;留着旧索引反而会把自建分片过滤掉
        if os.path.isfile(index_path):
            os.remove(index_path)
        return
    # 重新计算每个 key 落在哪个新分片
    new_weight_map: dict[str, str] = {}
    for name in shard_names:
        from safetensors import safe_open

        with safe_open(os.path.join(dst, name), framework="pt") as handle:
            for key in handle.keys():
                new_weight_map[key] = name
    total_size = sum(os.path.getsize(os.path.join(dst, name)) for name in shard_names)
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, handle, indent=2)
    missing = keep - set(new_weight_map)
    if missing:
        raise RuntimeError(f"{len(missing)} 个张量没有写进新分片,例如 {sorted(missing)[:5]}")


def main() -> int:
    args = parse_args()
    prefixes = tuple(args.keep_prefix) + ROLE_RULES[args.role]
    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        print(f"源目录不存在: {src}", file=sys.stderr)
        return 2

    weight_map = load_weight_map(src)
    grouped, matched = select_keys(weight_map, prefixes)
    report(weight_map, src, grouped, prefixes)
    if matched == 0:
        print("\n没有任何张量命中保留规则,请用 --keep-prefix 指定正确前缀。", file=sys.stderr)
        return 3
    if args.dry_run:
        return 0

    if not args.dst:
        print("非 dry-run 必须给 --dst", file=sys.stderr)
        return 2
    dst = os.path.abspath(args.dst)
    if dst == src or src.startswith(dst + os.sep):
        print("为安全起见,--dst 不能等于/包含 --src", file=sys.stderr)
        return 2
    os.makedirs(dst, exist_ok=True)

    copied = copy_metadata(src, dst)
    print(f"\n[copy] {len(copied)} metadata file(s): {copied}")

    keep = {key for keys in grouped.values() for key in keys}
    shard_names = write_slices(src, dst, weight_map, keep, int(args.max_shard_gb * 2**30))
    print(f"[write] {len(shard_names)} shard(s): {shard_names}")
    write_index(dst, weight_map, keep, shard_names)

    # 复核:输出目录的键集合必须与计划一致
    from safetensors import safe_open

    written_keys: set[str] = set()
    for name in shard_names:
        with safe_open(os.path.join(dst, name), framework="pt") as handle:
            written_keys |= set(handle.keys())
    if written_keys != keep:
        print(f"[verify] FAIL: 计划 {len(keep)} / 实际 {len(written_keys)}", file=sys.stderr)
        return 4
    total = sum(os.path.getsize(os.path.join(dst, name)) for name in shard_names)
    print(f"[verify] OK: {len(written_keys)} tensor(s), {total / 2**30:.2f} GiB in {dst}")
    print("[next ] 用 split 出来的目录启动该侧;若仍报 'Following weights were not initialized',按报错名单补张量")
    return 0


if __name__ == "__main__":
    sys.exit(main())
