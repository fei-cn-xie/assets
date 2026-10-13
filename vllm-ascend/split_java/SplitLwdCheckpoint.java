import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.FileStore;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * 把一个完整的 HF safetensors checkpoint 按 LWD 角色(edge/cloud)切成两个子集。
 *
 * <p>与 Python 版 {@code split_lwd_checkpoint.py} 语义一致,单文件、零第三方依赖:
 * safetensors 是"8 字节小端头长度 + JSON 头 + 原始张量字节",所以这里只做
 * <b>按字节范围搬运</b>,不需要任何张量库;写出的分片头部重新生成 data_offsets,
 * 数据段逐块流式拷贝(内存峰值与分片大小无关)。
 *
 * <pre>
 * 用法:
 *   javac SplitLwdCheckpoint.java
 *   java  SplitLwdCheckpoint --src ./Qwen3.6-27B --role edge --dry-run
 *   java  SplitLwdCheckpoint --src ./Qwen3.6-27B --dst ./Qwen3.6-27B-edge  --role edge  --max-shard-gb 8
 *   java  SplitLwdCheckpoint --src ./Qwen3.6-27B --dst ./Qwen3.6-27B-cloud --role cloud --max-shard-gb 8
 *
 * 退出码: 0 成功 / 2 参数或路径非法 / 3 无张量命中 / 4 复核失败 / 5 目标盘空间不足
 * </pre>
 */
public final class SplitLwdCheckpoint {

    private static final String INDEX_NAME = "model.safetensors.index.json";
    private static final long GIB = 1L << 30;

    /** 每侧必须保留的 checkpoint 名前缀。 */
    private static final Map<String, String[]> ROLE_RULES = new LinkedHashMap<String, String[]>();
    /** 每侧需要拷贝的非权重文件("auto" 策略);"*" 表示全拷。 */
    private static final Map<String, String[]> METADATA_RULES = new LinkedHashMap<String, String[]>();

    static {
        ROLE_RULES.put(
                "edge",
                new String[] {
                    "lm_head.", // 边侧 rank-replay 还原 token
                    "model.language_model.embed_tokens.", // VL 布局:文本嵌入
                    "model.embed_tokens.", // 纯文本布局:文本嵌入
                    "model.visual." // 视觉塔(多模态必须)
                });
        ROLE_RULES.put(
                "cloud",
                new String[] {
                    "lm_head.", // 云侧采样 + 算秩
                    "model.language_model.", // VL 布局:embed/layers/norm
                    "model.layers.", // 纯文本布局:transformer 层
                    "model.norm.", // 纯文本布局:final norm
                    "model.embed_tokens.", // 纯文本布局:文本嵌入(decode 行本地嵌入)
                    "mtp.", // 独立 MTP 权重(若开投机解码)
                    "model.mtp."
                });
        // 边侧对外服务、并且自己跑视觉塔编码 -> 全部配置/分词文件都要。
        // 云侧配套 vllm 的 LWD 改动(skip_tokenizer_init + mm_encoder_remote):
        // 不建 tokenizer、不建多模态 processor、不对外服务 -> 只需要 config.json。
        METADATA_RULES.put("edge", new String[] {"*"});
        METADATA_RULES.put("cloud", new String[] {"config.json"});
    }

    private static final Set<String> SKIP_FILE_EXACT =
            new LinkedHashSet<String>(
                    java.util.Arrays.asList("README.md", "LICENSE", ".gitattributes"));

    private SplitLwdCheckpoint() {}

    // ------------------------------------------------------------------
    // 参数
    // ------------------------------------------------------------------

    private static final class Args {
        String src;
        String dst;
        String role;
        String metadata = "auto";
        double maxShardGb = 5.0;
        boolean dryRun;
        boolean help;
        final List<String> keepPrefix = new ArrayList<String>();
    }

    private static Args parseArgs(String[] argv) {
        Args a = new Args();
        for (int i = 0; i < argv.length; i++) {
            String arg = argv[i];
            if ("--src".equals(arg)) {
                a.src = next(argv, ++i, arg);
            } else if ("--dst".equals(arg)) {
                a.dst = next(argv, ++i, arg);
            } else if ("--role".equals(arg)) {
                a.role = next(argv, ++i, arg);
            } else if ("--metadata".equals(arg)) {
                a.metadata = next(argv, ++i, arg);
            } else if ("--max-shard-gb".equals(arg)) {
                a.maxShardGb = Double.parseDouble(next(argv, ++i, arg));
            } else if ("--keep-prefix".equals(arg)) {
                a.keepPrefix.add(next(argv, ++i, arg));
            } else if ("--dry-run".equals(arg)) {
                a.dryRun = true;
            } else if ("--help".equals(arg) || "-h".equals(arg)) {
                a.help = true;
            } else {
                throw new IllegalArgumentException("未知参数: " + arg);
            }
        }
        return a;
    }

    private static String next(String[] argv, int i, String flag) {
        if (i >= argv.length) {
            throw new IllegalArgumentException(flag + " 缺少取值");
        }
        return argv[i];
    }

    private static void usage() {
        System.out.println("用法: java SplitLwdCheckpoint --src <完整模型目录> --role {edge|cloud} [选项]");
        System.out.println("选项:");
        System.out.println("  --dst <目录>            输出目录(非 dry-run 必填,不能等于/包含 --src)");
        System.out.println("  --max-shard-gb <GB>     输出分片上限,默认 5(内存紧张/想单文件就调大)");
        System.out.println("  --keep-prefix <前缀>    追加保留前缀(可重复)");
        System.out.println("  --metadata {auto|all}   非权重文件策略:auto=按角色(云侧只拷 config.json),all=全拷");
        System.out.println("  --dry-run               只打印将保留/丢弃的清单,不写文件");
    }

    // ------------------------------------------------------------------
    // safetensors 头
    // ------------------------------------------------------------------

    private static final class TensorInfo {
        final String name;
        final String dtype;
        final List<Object> shape;
        final long offset; // data_offsets[0]
        final long size; // data_offsets[1] - data_offsets[0]
        final String sourceShard;
        final long sourceDataStart; // 该分片数据段起点 = 8 + 头长度

        TensorInfo(
                String name,
                String dtype,
                List<Object> shape,
                long offset,
                long size,
                String sourceShard,
                long sourceDataStart) {
            this.name = name;
            this.dtype = dtype;
            this.shape = shape;
            this.offset = offset;
            this.size = size;
            this.sourceShard = sourceShard;
            this.sourceDataStart = sourceDataStart;
        }
    }

    private static long readHeaderLength(FileChannel ch) throws IOException {
        ByteBuffer head = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        int got = 0;
        while (got < 8) {
            int r = ch.read(head, got);
            if (r < 0) {
                break;
            }
            got += r;
        }
        if (got != 8) {
            throw new IOException("不是合法的 safetensors(文件太短)");
        }
        head.flip();
        return head.getLong();
    }

    /** 读取一个 safetensors 文件的头部,得到每个张量的 dtype/shape/data_offsets。 */
    private static List<TensorInfo> readHeader(Path path, String shardName) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = ch.size();
            long headerLen = readHeaderLength(ch);
            if (headerLen <= 0 || headerLen > fileSize - 8) {
                throw new IOException(path + ": 头部长度非法: " + headerLen);
            }
            byte[] raw = new byte[(int) headerLen];
            ByteBuffer buf = ByteBuffer.wrap(raw);
            long pos = 8;
            while (buf.hasRemaining()) {
                int r = ch.read(buf, pos);
                if (r < 0) {
                    throw new EOFException(path + ": 头部被截断");
                }
                pos += r;
            }
            Object parsed = Json.parse(new String(raw, StandardCharsets.UTF_8));
            if (!(parsed instanceof Map)) {
                throw new IOException(path + ": 头部不是 JSON 对象");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> header = (Map<String, Object>) parsed;
            long dataStart = 8 + headerLen;
            List<TensorInfo> out = new ArrayList<TensorInfo>();
            for (Map.Entry<String, Object> e : header.entrySet()) {
                if ("__metadata__".equals(e.getKey())) {
                    continue;
                }
                if (!(e.getValue() instanceof Map)) {
                    continue;
                }
                @SuppressWarnings("unchecked")
                Map<String, Object> info = (Map<String, Object>) e.getValue();
                Object dtype = info.get("dtype");
                Object shape = info.get("shape");
                Object offsets = info.get("data_offsets");
                if (!(shape instanceof List) || !(offsets instanceof List)) {
                    throw new IOException(path + ": 张量 " + e.getKey() + " 头部缺 shape/data_offsets");
                }
                List<?> offList = (List<?>) offsets;
                if (offList.size() != 2) {
                    throw new IOException(path + ": 张量 " + e.getKey() + " data_offsets 非法");
                }
                long start = ((Number) offList.get(0)).longValue();
                long end = ((Number) offList.get(1)).longValue();
                if (end < start) {
                    throw new IOException(path + ": 张量 " + e.getKey() + " data_offsets 反序");
                }
                @SuppressWarnings("unchecked")
                List<Object> shapeList = (List<Object>) shape;
                out.add(
                        new TensorInfo(
                                e.getKey(),
                                String.valueOf(dtype),
                                shapeList,
                                start,
                                end - start,
                                shardName,
                                dataStart));
            }
            return out;
        }
    }

    /** 无 index 时的退化路径:按文件名扫描所有 *.safetensors 的头部取张量名。 */
    private static Map<String, String> loadWeightMap(Path src) throws IOException {
        Path index = src.resolve(INDEX_NAME);
        Map<String, String> weightMap = new LinkedHashMap<String, String>();
        if (Files.isRegularFile(index)) {
            Object parsed = Json.parse(new String(Files.readAllBytes(index), StandardCharsets.UTF_8));
            if (!(parsed instanceof Map)) {
                throw new IOException(index + ": 不是 JSON 对象");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> root = (Map<String, Object>) parsed;
            Object wm = root.get("weight_map");
            if (!(wm instanceof Map)) {
                throw new IOException(index + ": 缺 weight_map");
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> map = (Map<String, Object>) wm;
            for (Map.Entry<String, Object> e : map.entrySet()) {
                weightMap.put(e.getKey(), String.valueOf(e.getValue()));
            }
            return weightMap;
        }
        for (String shard : listSafetensors(src)) {
            for (TensorInfo t : readHeader(src.resolve(shard), shard)) {
                weightMap.put(t.name, shard);
            }
        }
        return weightMap;
    }

    private static List<String> listSafetensors(Path dir) throws IOException {
        List<String> names = new ArrayList<String>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
            for (Path p : ds) {
                String n = p.getFileName().toString();
                if (n.endsWith(".safetensors") && Files.isRegularFile(p)) {
                    names.add(n);
                }
            }
        }
        Collections.sort(names);
        return names;
    }

    // ------------------------------------------------------------------
    // 主流程
    // ------------------------------------------------------------------

    private static int run(String[] argv) {
        Args args;
        try {
            args = parseArgs(argv);
        } catch (RuntimeException e) {
            System.err.println("参数错误: " + e.getMessage());
            usage();
            return 2;
        }
        if (args.help) {
            usage();
            return 0;
        }
        if (args.src == null || args.role == null) {
            usage();
            return 2;
        }
        if (!ROLE_RULES.containsKey(args.role)) {
            System.err.println("--role 必须是 edge 或 cloud");
            return 2;
        }
        if (!"auto".equals(args.metadata) && !"all".equals(args.metadata)) {
            System.err.println("--metadata 必须是 auto 或 all");
            return 2;
        }

        try {
            Path srcRaw = expand(args.src);
            if (!Files.isDirectory(srcRaw)) {
                System.err.println("源目录不存在: " + srcRaw.toAbsolutePath());
                return 2;
            }
            Path src = srcRaw.toRealPath();

            Map<String, String> weightMap = loadWeightMap(src);

            List<String> prefixes = new ArrayList<String>(args.keepPrefix);
            Collections.addAll(prefixes, ROLE_RULES.get(args.role));

            // 前缀归组
            Map<String, List<String>> grouped = new LinkedHashMap<String, List<String>>();
            for (String p : prefixes) {
                grouped.put(p, new ArrayList<String>());
            }
            int matched = 0;
            for (String key : weightMap.keySet()) {
                for (String p : prefixes) {
                    if (key.startsWith(p)) {
                        grouped.get(p).add(key);
                        matched++;
                        break;
                    }
                }
            }

            Set<String> keep = new LinkedHashSet<String>();
            for (List<String> keys : grouped.values()) {
                keep.addAll(keys);
            }

            long keepBytes = report(weightMap, src, grouped, prefixes, keep);
            if (matched == 0) {
                System.err.println("\n没有任何张量命中保留规则,请用 --keep-prefix 指定正确前缀。");
                return 3;
            }

            // 非权重文件计划
            List<String> candidates = metadataCandidates(src);
            String[] allowed =
                    "all".equals(args.metadata)
                            ? new String[] {"*"}
                            : METADATA_RULES.get(args.role);
            List<String> metaCopy = new ArrayList<String>();
            List<String> metaSkip = new ArrayList<String>();
            for (String name : candidates) {
                boolean want = false;
                for (String rule : allowed) {
                    if ("*".equals(rule) || rule.equals(name)) {
                        want = true;
                        break;
                    }
                }
                if (want) {
                    metaCopy.add(name);
                } else {
                    metaSkip.add(name);
                }
            }
            System.out.println("\n[meta] 将拷贝 " + metaCopy.size() + " 个非权重文件: " + metaCopy);
            if (!metaSkip.isEmpty()) {
                System.out.println("[meta] 主动跳过 " + metaSkip.size() + " 个: " + metaSkip);
            }
            if ("auto".equals(args.metadata) && "cloud".equals(args.role)) {
                System.out.println(
                        "[meta] 云侧最小化依赖 vllm 的 LWD 改动(skip_tokenizer_init + mm_encoder_remote);"
                                + "若云侧仍跑未打补丁的 vllm,请加 --metadata all");
            }

            if (args.dryRun) {
                return 0;
            }
            if (args.dst == null) {
                System.err.println("非 dry-run 必须给 --dst");
                return 2;
            }

            Path dst = expand(args.dst).toAbsolutePath().normalize();
            if (dst.equals(src) || src.startsWith(dst)) {
                System.err.println("为安全起见,--dst 不能等于/包含 --src");
                return 2;
            }

            // 空间预检
            Path probe = dst;
            while (probe != null && !Files.exists(probe)) {
                probe = probe.getParent();
            }
            if (probe != null && keepBytes > 0) {
                FileStore fs = Files.getFileStore(probe);
                long free = fs.getUsableSpace();
                long need = keepBytes + 256 * (1L << 20);
                System.out.printf(
                        Locale.ROOT,
                        "%n[space] 目标 %s 可用 %.2f GiB,需要 %.2f GiB%n",
                        probe,
                        free / (double) GIB,
                        need / (double) GIB);
                if (free < need) {
                    System.err.println("空间不足:换一个有空间的 --dst(另一块盘/挂载点),或先清理目标盘。");
                    return 5;
                }
            }

            Files.createDirectories(dst);
            List<String> copied = new ArrayList<String>();
            for (String name : metaCopy) {
                Path from = src.resolve(name);
                if (!Files.isRegularFile(from)) {
                    continue;
                }
                Files.copy(from, dst.resolve(name), StandardCopyOption.REPLACE_EXISTING);
                copied.add(name);
            }
            System.out.println("\n[copy] " + copied.size() + " metadata file(s): " + copied);

            // 源分片顺序:按分片名排序,片内按头部顺序 -> 读盘近似顺序,并保证确定性
            Map<String, Set<String>> wantedPerShard = new TreeMap<String, Set<String>>();
            for (String key : keep) {
                String shard = weightMap.get(key);
                Set<String> wanted = wantedPerShard.get(shard);
                if (wanted == null) {
                    wanted = new LinkedHashSet<String>();
                    wantedPerShard.put(shard, wanted);
                }
                wanted.add(key);
            }
            List<TensorInfo> ordered = new ArrayList<TensorInfo>();
            for (Map.Entry<String, Set<String>> e : wantedPerShard.entrySet()) {
                Path shardPath = src.resolve(e.getKey());
                if (!Files.isRegularFile(shardPath)) {
                    continue;
                }
                for (TensorInfo t : readHeader(shardPath, e.getKey())) {
                    if (e.getValue().contains(t.name)) {
                        ordered.add(t);
                    }
                }
            }

            long maxShardBytes = (long) (args.maxShardGb * GIB);
            List<List<TensorInfo>> shards = planShards(ordered, maxShardBytes);
            int totalShards = shards.size();
            List<String> shardNames = new ArrayList<String>();
            for (int i = 0; i < totalShards; i++) {
                shardNames.add(String.format(Locale.ROOT, "model-%05d-of-%05d.safetensors", i + 1, totalShards));
            }

            try {
                writeShards(src, dst, shards, shardNames);
            } catch (Throwable t) {
                for (String name : shardNames) {
                    Path p = dst.resolve(name);
                    try {
                        Files.deleteIfExists(p);
                    } catch (IOException ignored) {
                        // best effort
                    }
                }
                System.err.println(
                        "[clean] 写出失败,已删除本次生成的分片(目标目录只留 metadata 文件,可直接重跑)");
                throw t;
            }
            System.out.println("[write] " + shardNames.size() + " shard(s): " + shardNames);

            writeIndex(dst, keep, shardNames);

            // 复核:输出目录的键集合必须与计划一致
            Set<String> written = new LinkedHashSet<String>();
            for (String name : shardNames) {
                for (TensorInfo t : readHeader(dst.resolve(name), name)) {
                    written.add(t.name);
                }
            }
            if (!written.equals(keep)) {
                System.err.printf(
                        Locale.ROOT, "[verify] FAIL: 计划 %d / 实际 %d%n", keep.size(), written.size());
                return 4;
            }
            long total = 0;
            for (String name : shardNames) {
                total += Files.size(dst.resolve(name));
            }
            System.out.printf(
                    Locale.ROOT,
                    "[verify] OK: %d tensor(s), %.2f GiB in %s%n",
                    written.size(),
                    total / (double) GIB,
                    dst);
            System.out.println(
                    "[next ] 用 split 出来的目录启动该侧;若仍报 'Following weights were not initialized',按报错名单补张量");
            return 0;
        } catch (IOException e) {
            System.err.println("失败: " + e);
            return 1;
        } catch (RuntimeException e) {
            System.err.println("失败: " + e);
            return 1;
        }
    }

    private static List<List<TensorInfo>> planShards(List<TensorInfo> ordered, long maxShardBytes) {
        List<List<TensorInfo>> shards = new ArrayList<List<TensorInfo>>();
        List<TensorInfo> cur = new ArrayList<TensorInfo>();
        long curBytes = 0;
        for (TensorInfo t : ordered) {
            if (!cur.isEmpty() && curBytes + t.size > maxShardBytes) {
                shards.add(cur);
                cur = new ArrayList<TensorInfo>();
                curBytes = 0;
            }
            cur.add(t);
            curBytes += t.size;
        }
        if (!cur.isEmpty()) {
            shards.add(cur);
        }
        return shards;
    }

    private static void writeShards(
            Path src, Path dst, List<List<TensorInfo>> shards, List<String> shardNames) throws IOException {
        byte[] buf = new byte[8 << 20];
        for (int i = 0; i < shards.size(); i++) {
            List<TensorInfo> tensors = shards.get(i);
            String shardName = shardNames.get(i);

            // 目标头部:与源 dtype/shape 一致,data_offsets 按打包顺序重算
            StringBuilder header = new StringBuilder();
            header.append('{');
            long offset = 0;
            for (int k = 0; k < tensors.size(); k++) {
                TensorInfo t = tensors.get(k);
                if (k > 0) {
                    header.append(',');
                }
                header.append(Json.quote(t.name)).append(":{");
                header.append("\"dtype\":").append(Json.quote(t.dtype)).append(',');
                header.append("\"shape\":").append(shapeJson(t.shape)).append(',');
                header.append("\"data_offsets\":[").append(offset).append(',').append(offset + t.size).append("]}");
                offset += t.size;
            }
            if (tensors.size() > 0) {
                header.append(',');
            }
            header.append("\"__metadata__\":{\"format\":\"pt\"}}");

            byte[] json = header.toString().getBytes(StandardCharsets.UTF_8);
            int pad = (8 - ((json.length + 8) % 8)) % 8;
            long headerLen = json.length + pad;
            ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            lenBuf.putLong(headerLen);
            lenBuf.flip();

            Path outPath = dst.resolve(shardName);
            try (FileChannel out =
                    FileChannel.open(
                            outPath,
                            StandardOpenOption.CREATE,
                            StandardOpenOption.TRUNCATE_EXISTING,
                            StandardOpenOption.WRITE)) {
                writeFully(out, 0, lenBuf);
                ByteBuffer jsonBuf = ByteBuffer.wrap(json);
                writeFully(out, 8, jsonBuf);
                if (pad > 0) {
                    byte[] padBytes = new byte[pad];
                    java.util.Arrays.fill(padBytes, (byte) ' ');
                    writeFully(out, 8 + json.length, ByteBuffer.wrap(padBytes));
                }
                long outPos = 8 + headerLen;
                for (TensorInfo t : tensors) {
                    Path inPath = src.resolve(t.sourceShard);
                    try (FileChannel in = FileChannel.open(inPath, StandardOpenOption.READ)) {
                        outPos = copyRange(in, t.sourceDataStart + t.offset, out, outPos, t.size, buf);
                    }
                }
            }
        }
    }

    private static String shapeJson(List<Object> shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            Object v = shape.get(i);
            sb.append(v instanceof Number ? String.valueOf(((Number) v).longValue()) : Json.quote(String.valueOf(v)));
        }
        return sb.append(']').toString();
    }

    private static void writeFully(FileChannel out, long pos, ByteBuffer buf) throws IOException {
        long p = pos;
        while (buf.hasRemaining()) {
            int w = out.write(buf, p);
            if (w <= 0) {
                throw new IOException("写文件失败");
            }
            p += w;
        }
    }

    private static long copyRange(FileChannel in, long inPos, FileChannel out, long outPos, long bytes, byte[] buf)
            throws IOException {
        long remaining = bytes;
        long ip = inPos;
        long op = outPos;
        while (remaining > 0) {
            int want = (int) Math.min(buf.length, remaining);
            ByteBuffer bb = ByteBuffer.wrap(buf, 0, want);
            int got = 0;
            while (got < want) {
                int r = in.read(bb, ip + got);
                if (r < 0) {
                    throw new EOFException("源分片被截断,缺少 " + (want - got) + " 字节");
                }
                got += r;
            }
            bb.flip();
            int written = 0;
            while (bb.hasRemaining()) {
                int w = out.write(bb, op + written);
                if (w <= 0) {
                    throw new IOException("写文件失败");
                }
                written += w;
            }
            ip += want;
            op += want;
            remaining -= want;
        }
        return op;
    }

    private static void writeIndex(Path dst, Set<String> keep, List<String> shardNames) throws IOException {
        Path index = dst.resolve(INDEX_NAME);
        if (shardNames.size() <= 1) {
            // 单分片不需要索引;留着旧索引反而会把自建分片过滤掉
            Files.deleteIfExists(index);
            return;
        }
        Map<String, String> newMap = new LinkedHashMap<String, String>();
        long totalSize = 0;
        for (String name : shardNames) {
            totalSize += Files.size(dst.resolve(name));
            for (TensorInfo t : readHeader(dst.resolve(name), name)) {
                newMap.put(t.name, name);
            }
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\"metadata\":{\"total_size\":").append(totalSize).append("},\"weight_map\":{");
        boolean first = true;
        for (Map.Entry<String, String> e : newMap.entrySet()) {
            if (!first) {
                sb.append(',');
            }
            first = false;
            sb.append(Json.quote(e.getKey())).append(':').append(Json.quote(e.getValue()));
        }
        sb.append("}}");
        Files.write(
                index,
                sb.toString().getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE);
        List<String> missing = new ArrayList<String>();
        for (String key : keep) {
            if (!newMap.containsKey(key)) {
                missing.add(key);
            }
        }
        if (!missing.isEmpty()) {
            throw new IOException(
                    missing.size() + " 个张量没有写进新分片,例如 " + missing.subList(0, Math.min(5, missing.size())));
        }
    }

    // ------------------------------------------------------------------
    // 报告
    // ------------------------------------------------------------------

    private static List<String> metadataCandidates(Path src) throws IOException {
        List<String> names = new ArrayList<String>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(src)) {
            for (Path p : ds) {
                String name = p.getFileName().toString();
                if (name.endsWith(".safetensors")) {
                    continue;
                }
                if (name.startsWith(".git") || name.equals(INDEX_NAME)) {
                    continue;
                }
                if (SKIP_FILE_EXACT.contains(name)) {
                    continue;
                }
                if (!Files.isRegularFile(p)) {
                    continue;
                }
                names.add(name);
            }
        }
        Collections.sort(names);
        return names;
    }

    private static long report(
            Map<String, String> weightMap,
            Path src,
            Map<String, List<String>> grouped,
            List<String> prefixes,
            Set<String> keep) {
        long totalBytes = 0;
        Set<String> shards = new LinkedHashSet<String>(weightMap.values());
        for (String shard : shards) {
            Path p = src.resolve(shard);
            if (Files.isRegularFile(p)) {
                try {
                    totalBytes += Files.size(p);
                } catch (IOException ignored) {
                    // ignore
                }
            }
        }

        System.out.printf(
                Locale.ROOT,
                "[in ] %d tensor(s), %.2f GiB on disk%n",
                weightMap.size(),
                totalBytes / (double) GIB);
        System.out.println("[rule] keep prefixes: " + prefixes);

        System.out.println("\n  matched tensors per prefix:");
        long kept = 0;
        for (String prefix : prefixes) {
            List<String> keys = grouped.get(prefix);
            if (keys == null || keys.isEmpty()) {
                System.out.printf("    %-40s 0 tensor(s)  <-- 没命中,检查命名布局/是否该保留%n", prefix);
                continue;
            }
            kept += keys.size();
            System.out.printf("    %-40s %d tensor(s)%n", prefix, keys.size());
        }
        System.out.printf(
                "%n  => keep %d tensor(s) of %d; drop %d%n",
                kept, weightMap.size(), weightMap.size() - kept);

        Map<String, Integer> dropped = new TreeMap<String, Integer>();
        for (String key : weightMap.keySet()) {
            if (keep.contains(key)) {
                continue;
            }
            String[] parts = key.split("\\.");
            StringBuilder coarse = new StringBuilder();
            for (int i = 0; i < Math.min(3, parts.length); i++) {
                if (i > 0) {
                    coarse.append('.');
                }
                coarse.append(parts[i]);
            }
            String k = coarse.toString();
            Integer c = dropped.get(k);
            dropped.put(k, c == null ? 1 : c + 1);
        }
        List<Map.Entry<String, Integer>> entries = new ArrayList<Map.Entry<String, Integer>>(dropped.entrySet());
        Collections.sort(
                entries,
                new Comparator<Map.Entry<String, Integer>>() {
                    @Override
                    public int compare(Map.Entry<String, Integer> a, Map.Entry<String, Integer> b) {
                        return b.getValue() - a.getValue();
                    }
                });
        System.out.println("\n  dropped tensors grouped by leading components (top 15):");
        for (int i = 0; i < Math.min(15, entries.size()); i++) {
            System.out.printf("    %-48s %d%n", entries.get(i).getKey(), entries.get(i).getValue());
        }

        // 精确的保留体积(只读头部,不读权重数据)
        long keepSize = 0;
        int missingShards = 0;
        Map<String, List<String>> wantPerShard = new TreeMap<String, List<String>>();
        for (String key : keep) {
            String shard = weightMap.get(key);
            List<String> list = wantPerShard.get(shard);
            if (list == null) {
                list = new ArrayList<String>();
                wantPerShard.put(shard, list);
            }
            list.add(key);
        }
        try {
            for (Map.Entry<String, List<String>> e : wantPerShard.entrySet()) {
                Path p = src.resolve(e.getKey());
                if (!Files.isRegularFile(p)) {
                    missingShards++;
                    continue;
                }
                Set<String> wanted = new LinkedHashSet<String>(e.getValue());
                for (TensorInfo t : readHeader(p, e.getKey())) {
                    if (wanted.contains(t.name)) {
                        keepSize += t.size;
                    }
                }
            }
        } catch (IOException ex) {
            System.out.println("\n  [size] 无法统计保留体积: " + ex);
            return keepSize;
        }
        String note = missingShards > 0 ? "(有 " + missingShards + " 个分片文件缺失,统计不完整)" : "";
        System.out.printf(
                Locale.ROOT,
                "%n  [size] 将写出 %.2f GiB / 源 %.2f GiB (%.1f%%) %s%n",
                keepSize / (double) GIB,
                totalBytes / (double) GIB,
                totalBytes > 0 ? keepSize * 100.0 / totalBytes : 0.0,
                note);
        return keepSize;
    }

    private static Path expand(String raw) {
        String home = System.getProperty("user.home");
        if ("~".equals(raw)) {
            return Paths.get(home);
        }
        if (raw.startsWith("~/") || raw.startsWith("~\\")) {
            return Paths.get(home, raw.substring(2));
        }
        return Paths.get(raw);
    }

    // ------------------------------------------------------------------
    // 极简 JSON 解析(只覆盖 index.json 与 safetensors 头部用到的子集)
    // ------------------------------------------------------------------

    static final class Json {
        private final String s;
        private int i;

        private Json(String s) {
            this.s = s;
        }

        static Object parse(String text) {
            Json p = new Json(text);
            p.skipWs();
            Object v = p.value();
            return v;
        }

        private void skipWs() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
                i++;
            }
        }

        private Object value() {
            skipWs();
            if (i >= s.length()) {
                throw new IllegalArgumentException("JSON 提前结束");
            }
            char c = s.charAt(i);
            switch (c) {
                case '{':
                    return object();
                case '[':
                    return array();
                case '"':
                    return string();
                case 't':
                    expect("true");
                    return Boolean.TRUE;
                case 'f':
                    expect("false");
                    return Boolean.FALSE;
                case 'n':
                    expect("null");
                    return null;
                default:
                    return number();
            }
        }

        private void expect(String word) {
            if (!s.startsWith(word, i)) {
                throw new IllegalArgumentException("JSON 期望 " + word + " @" + i);
            }
            i += word.length();
        }

        private Map<String, Object> object() {
            Map<String, Object> map = new LinkedHashMap<String, Object>();
            i++; // '{'
            skipWs();
            if (i < s.length() && s.charAt(i) == '}') {
                i++;
                return map;
            }
            while (true) {
                skipWs();
                String key = string();
                skipWs();
                if (i >= s.length() || s.charAt(i) != ':') {
                    throw new IllegalArgumentException("JSON 期望 ':' @" + i);
                }
                i++;
                Object v = value();
                map.put(key, v);
                skipWs();
                if (i >= s.length()) {
                    throw new IllegalArgumentException("JSON 对象未闭合");
                }
                char c = s.charAt(i++);
                if (c == '}') {
                    return map;
                }
                if (c != ',') {
                    throw new IllegalArgumentException("JSON 期望 ',' 或 '}' @" + (i - 1));
                }
            }
        }

        private List<Object> array() {
            List<Object> list = new ArrayList<Object>();
            i++; // '['
            skipWs();
            if (i < s.length() && s.charAt(i) == ']') {
                i++;
                return list;
            }
            while (true) {
                list.add(value());
                skipWs();
                if (i >= s.length()) {
                    throw new IllegalArgumentException("JSON 数组未闭合");
                }
                char c = s.charAt(i++);
                if (c == ']') {
                    return list;
                }
                if (c != ',') {
                    throw new IllegalArgumentException("JSON 期望 ',' 或 ']' @" + (i - 1));
                }
            }
        }

        private String string() {
            if (s.charAt(i) != '"') {
                throw new IllegalArgumentException("JSON 期望字符串 @" + i);
            }
            i++;
            StringBuilder sb = new StringBuilder();
            while (true) {
                if (i >= s.length()) {
                    throw new IllegalArgumentException("JSON 字符串未闭合");
                }
                char c = s.charAt(i++);
                if (c == '"') {
                    return sb.toString();
                }
                if (c != '\\') {
                    sb.append(c);
                    continue;
                }
                char e = s.charAt(i++);
                switch (e) {
                    case '"':
                        sb.append('"');
                        break;
                    case '\\':
                        sb.append('\\');
                        break;
                    case '/':
                        sb.append('/');
                        break;
                    case 'b':
                        sb.append('\b');
                        break;
                    case 'f':
                        sb.append('\f');
                        break;
                    case 'n':
                        sb.append('\n');
                        break;
                    case 'r':
                        sb.append('\r');
                        break;
                    case 't':
                        sb.append('\t');
                        break;
                    case 'u':
                        sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                        break;
                    default:
                        throw new IllegalArgumentException("JSON 非法转义 \\" + e);
                }
            }
        }

        private Object number() {
            int start = i;
            if (i < s.length() && (s.charAt(i) == '-' || s.charAt(i) == '+')) {
                i++;
            }
            boolean floating = false;
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c >= '0' && c <= '9') {
                    i++;
                } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                    floating = true;
                    i++;
                } else {
                    break;
                }
            }
            String text = s.substring(start, i);
            if (floating) {
                return Double.valueOf(text);
            }
            return Long.valueOf(text);
        }

        static String quote(String raw) {
            StringBuilder sb = new StringBuilder("\"");
            for (int k = 0; k < raw.length(); k++) {
                char c = raw.charAt(k);
                switch (c) {
                    case '"':
                        sb.append("\\\"");
                        break;
                    case '\\':
                        sb.append("\\\\");
                        break;
                    case '\n':
                        sb.append("\\n");
                        break;
                    case '\r':
                        sb.append("\\r");
                        break;
                    case '\t':
                        sb.append("\\t");
                        break;
                    default:
                        if (c < 0x20) {
                            sb.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                }
            }
            return sb.append('"').toString();
        }
    }

    public static void main(String[] argv) {
        System.exit(run(argv));
    }
}
