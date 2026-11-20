# AI Infra / 大模型推理框架工程师：半年学习路线

> **学习目标：** 从当前的 vLLM 二次开发工作出发，逐步建立模型、推理框架、NPU 硬件、算子和性能优化能力，为转向 AI Infra 性能优化与推理框架架构方向做准备。
>
> **建议周期：** 约 26 周（半年）  
> **建议投入：** 每天 1～2 小时，每周 8～10 小时  
> **当前背景：** 2022 年本科毕业；有技术咨询、少量 Java 开发经历；目前从事 Python 开发及基于 vLLM 的 VLM 推理框架二次开发；目标硬件为深腾 A2、A3 NPU。  
> **职业目标：** 未来考虑成都或深圳，目标月薪 25K～30K 及以上。

---

## 一、整体学习路线

| 阶段 | 学习方向 | 预计时间 | 核心目标 |
|---|---|---:|---|
| 1 | 计算机基础 | 3 周 | 补齐底层基础 |
| 2 | Python、C++、PyTorch | 3 周 | 理解框架开发基础 |
| 3 | Transformer、LLM、VLM | 4 周 | 理解模型运行原理 |
| 4 | vLLM 源码架构 | 4 周 | 掌握推理框架 |
| 5 | NPU、硬件架构、算子 | 4 周 | 理解底层执行机制 |
| 6 | 性能优化与 Profiling | 4 周 | 独立定位性能问题 |
| 7 | 算子开发与综合实践 | 4 周 | 建立底层开发实践能力 |

总计约 26 周。各阶段可以根据工作任务适当交叉，不必机械地等一个阶段全部结束才开始下一个阶段。

---

## 二、第一阶段：计算机基础（第 1～3 周）

目标：补齐与 AI Infra 工作直接相关的计算机基础，不必重新完整学习所有计算机专业课程。

### 1. 操作系统

- [ ] 进程和线程
- [ ] 线程调度
- [ ] 并发与并行
- [ ] 锁、互斥锁、条件变量
- [ ] 死锁
- [ ] 内存管理
- [ ] 虚拟内存
- [ ] 内存分页
- [ ] CPU Cache
- [ ] 上下文切换
- [ ] Linux 进程管理

**重点理解：**

- 为什么多线程任务可能出现性能下降？
- 为什么 CPU 利用率不高，但程序执行时间很长？
- 为什么某些任务会出现锁竞争？

### 2. Linux 基础

- [ ] Linux 文件系统
- [ ] Linux 进程管理
- [ ] Shell 脚本
- [ ] 环境变量
- [ ] 动态链接库与静态链接库
- [ ] ELF 文件
- [ ] 进程间通信

**重点工具：**

| 工具 | 用途 |
|---|---|
| `top` / `htop` | 查看 CPU、内存及进程资源 |
| `ps` | 查看进程 |
| `strace` | 跟踪系统调用 |
| `lsof` | 查看文件句柄 |
| `perf` | CPU 性能分析 |
| `gdb` | 程序调试 |
| `pmap` | 进程内存分析 |

### 3. 计算机网络

- [ ] TCP/IP
- [ ] HTTP、HTTP/2
- [ ] RPC
- [ ] Socket
- [ ] TCP 三次握手与四次挥手
- [ ] 网络延迟与吞吐量
- [ ] 序列化和反序列化

**为什么要学：** vLLM 不只是执行模型计算，还涉及 API Server、Engine、Worker 等模块之间的通信。

**阶段交付物：**

- [ ] 能使用 Linux 工具排查常见 CPU、内存、网络和进程问题。
- [ ] 能解释一次请求在网络和进程层面可能经历的主要环节。

---

## 三、第二阶段：Python、C++、PyTorch（第 4～6 周）

目标：巩固 Python 开发能力，并补齐面向底层 AI Infra 研发所需的 C++ 和 PyTorch 基础。

### 1. Python 进阶

- [ ] 装饰器
- [ ] 生成器与迭代器
- [ ] 多线程
- [ ] 多进程
- [ ] `asyncio`
- [ ] GIL
- [ ] Python 内存管理与 GC
- [ ] Cython
- [ ] Python C API

**重点理解：**

- Python 如何调用 C++ 扩展？
- Python 如何调用底层 NPU 算子？
- Python 与底层执行之间有哪些边界和开销？

### 2. C++ 基础

建议重点学习 C++11、C++14、C++17。

- [ ] 指针和引用
- [ ] 内存管理
- [ ] 堆和栈
- [ ] 类和对象
- [ ] 继承和多态
- [ ] 虚函数
- [ ] STL
- [ ] 智能指针
- [ ] RAII
- [ ] 模板
- [ ] Lambda
- [ ] 多线程
- [ ] 原子操作
- [ ] 内存模型

重点掌握：

- `std::shared_ptr`
- `std::unique_ptr`
- `std::thread`
- `std::mutex`
- `std::condition_variable`
- `std::future`

### 3. PyTorch

- [ ] Tensor
- [ ] Tensor Shape
- [ ] Tensor Stride
- [ ] Tensor Memory Layout
- [ ] Autograd
- [ ] Computational Graph
- [ ] Module
- [ ] Forward
- [ ] Hook
- [ ] `torch.compile`
- [ ] Dispatcher
- [ ] 自定义算子

**重点问题：**

以 `output = model(input)` 为例，理解这行代码背后发生了什么：

1. PyTorch 如何调用底层算子？
2. 算子如何分发到 CPU 或 NPU？
3. Python、C++、Dispatcher 和设备 Kernel 之间如何衔接？

**阶段交付物：**

- [ ] 能看懂常见 PyTorch 模型代码。
- [ ] 能解释 Python 到 C++ 再到底层算子的基本调用过程。

---

## 四、第三阶段：Transformer、LLM、VLM（第 7～10 周）

目标：理解模型内部计算，而不只是会调用模型。

### 1. Transformer

建议按以下顺序学习：

- [ ] Tokenizer
- [ ] Embedding
- [ ] Positional Encoding
- [ ] Self-Attention
- [ ] Multi-Head Attention
- [ ] Q、K、V
- [ ] RoPE
- [ ] RMSNorm
- [ ] LayerNorm
- [ ] FFN
- [ ] SwiGLU
- [ ] Residual Connection
- [ ] Transformer Block

重点理解 Attention：

`Attention(Q, K, V) = Softmax(QKᵀ / √dₖ) V`

需要能够推导每一步的 Tensor Shape，并理解矩阵运算的计算量和访存特征。

### 2. LLM 推理原理

- [ ] Prefill
- [ ] Decode
- [ ] Autoregressive Generation
- [ ] KV Cache
- [ ] Batch
- [ ] Continuous Batching
- [ ] PagedAttention
- [ ] Sampling
- [ ] Temperature
- [ ] Top-K
- [ ] Top-P

重点理解：

- 为什么 Prefill 通常更偏计算密集？
- 为什么 Decode 容易受到显存带宽限制？
- 为什么 KV Cache 会影响并发能力？
- Batch Size、序列长度和 KV Cache 占用如何影响性能？

### 3. VLM 多模态模型

- [ ] Vision Encoder
- [ ] ViT
- [ ] Patch Embedding
- [ ] Image Token
- [ ] Vision-Language Alignment
- [ ] Cross Attention
- [ ] 多模态 Embedding
- [ ] 图像分辨率与 Token 数量
- [ ] 多模态 Prefill
- [ ] 图像特征缓存

**阶段交付物：**

- [ ] 能从模型结构角度解释一次完整的 LLM/VLM 推理过程。
- [ ] 能画出 Transformer Block 的结构图，并标注主要 Tensor Shape。
- [ ] 能解释 Prefill、Decode 和 KV Cache 的关系。

---

## 五、第四阶段：vLLM 源码架构（第 11～14 周）

这是当前工作最直接相关的阶段。以实际使用的 vLLM 版本和二次开发代码为准，不必一开始就试图看懂所有源码。

### 1. vLLM 整体架构

重点模块：

- [ ] API Server
- [ ] AsyncLLM
- [ ] LLMEngine
- [ ] EngineCore
- [ ] Scheduler
- [ ] Executor
- [ ] Worker
- [ ] Model Runner
- [ ] KV Cache Manager

### 2. 请求调用链路

先建立逻辑认识，再对照实际代码补充真实类名、函数名和调用关系：

```text
Client
  |
  v
API Server
  |
  v
Request Handler
  |
  v
Engine
  |
  v
Scheduler
  |
  v
Executor
  |
  v
Worker
  |
  v
Model Runner
  |
  v
Model Forward
  |
  v
NPU Execution
  |
  v
Sampling
  |
  v
Output
```

需要追踪：

- [ ] 请求从哪个接口进入？
- [ ] 经过哪些关键方法？
- [ ] Scheduler 如何调度？
- [ ] 请求被分配到哪个 Worker？
- [ ] Model Runner 如何执行？
- [ ] NPU 算子在哪里被调用？
- [ ] 结果经过哪些模块返回？

### 3. Scheduler

- [ ] Request Scheduling
- [ ] FCFS
- [ ] Continuous Batching
- [ ] Waiting Queue
- [ ] Running Queue
- [ ] Preemption
- [ ] Token Budget
- [ ] Scheduling Policy

### 4. KV Cache

- [ ] KV Cache 分配
- [ ] Block Manager
- [ ] Block Table
- [ ] Block Reuse
- [ ] Prefix Caching
- [ ] Cache Eviction
- [ ] PagedAttention

### 5. Worker 和 Model Runner

- [ ] Worker 初始化
- [ ] Model Loading
- [ ] Input Preparation
- [ ] Forward
- [ ] Sampling
- [ ] Output Processing

**阶段交付物：**

- [ ] 绘制当前二次开发版本的真实调用链路图。
- [ ] 整理关键类、关键函数及其调用关系。
- [ ] 标记编测、运测以及 Embedding、Transformer Forward、Unembedding 在实际代码中的位置。
- [ ] 标记关键阶段的耗时采集点。

---

## 六、第五阶段：NPU、硬件架构与算子（第 15～18 周）

目标：理解模型计算如何映射到加速器，以及算子从接口到设备执行的过程。

> **注意：** 当前目标硬件为深腾 A2、A3 NPU。应先确认硬件对应的软件栈、编译器、Runtime、算子库和开发工具，再选择具体 API 学习。不要未经确认就把其他厂商的开发接口直接套用到目标平台。

### 1. 硬件架构

- [ ] CPU Architecture
- [ ] GPU Architecture
- [ ] NPU Architecture
- [ ] SIMD
- [ ] SIMT
- [ ] Tensor Core / 矩阵计算单元
- [ ] Memory Hierarchy
- [ ] Cache
- [ ] HBM
- [ ] Memory Bandwidth
- [ ] Compute Throughput
- [ ] PCIe
- [ ] RDMA

重点理解：

- 计算密集型与访存密集型任务的区别。
- 为什么加速器理论算力很高，实际推理性能却可能不理想？
- 算力、带宽、访存和通信如何共同影响性能？

### 2. 常见算子

- [ ] Matrix Multiplication
- [ ] GEMM
- [ ] Batch GEMM
- [ ] Softmax
- [ ] LayerNorm
- [ ] RMSNorm
- [ ] RoPE
- [ ] Attention
- [ ] FlashAttention
- [ ] AllReduce
- [ ] AllGather

### 3. 算子开发流程

先理解通用流程，再对应目标 NPU 的实际实现：

```text
PyTorch Model
      |
      v
Operator API
      |
      v
Operator Dispatch
      |
      v
Operator Implementation
      |
      v
Compiler
      |
      v
Kernel
      |
      v
NPU Execution
      |
      v
Output
```

学习内容：

- [ ] 算子接口设计
- [ ] Tensor 描述
- [ ] Shape 和 Dtype
- [ ] Kernel 开发
- [ ] 算子编译
- [ ] 算子注册
- [ ] 算子调度
- [ ] 算子性能测试
- [ ] 算子精度验证

### 4. 硬件厂商软件栈

根据 A2、A3 的实际平台确认并学习：

- [ ] 编译器
- [ ] Runtime
- [ ] 算子库
- [ ] 设备管理
- [ ] 内存管理
- [ ] 通信库
- [ ] Profiling 工具

如果确认平台使用昇腾/CANN 兼容体系，再针对性学习 ACL、ACLNN、CANN 等组件；如果是其他软件栈，则以实际官方文档和内部代码为准。

**阶段交付物：**

- [ ] 能解释一个 Transformer 算子从 PyTorch 调用到 NPU 执行的过程。
- [ ] 能识别算子接口、编译、Runtime、Kernel 和设备执行之间的关系。

---

## 七、第六阶段：性能优化与 Profiling（第 19～22 周）

这一阶段直接结合日常工作，是形成可展示技术成果的关键阶段。

### 1. 推理性能指标

| 指标 | 含义 |
|---|---|
| TTFT | 首 Token 延迟 |
| ITL | Token 间延迟 |
| TPOT | 每个输出 Token 的平均耗时 |
| E2E Latency | 请求端到端延迟 |
| QPS | 每秒请求数 |
| Throughput | 吞吐量 |
| P50 | 50 分位延迟 |
| P90 | 90 分位延迟 |
| P99 | 99 分位延迟 |

### 2. ITL 问题排查

#### 第一层：请求层

- [ ] 请求是否排队？
- [ ] 是否存在请求拥塞？
- [ ] Batch Size 是否波动？
- [ ] 请求长度是否变化？

#### 第二层：调度层

- [ ] Scheduler 是否存在等待？
- [ ] 是否出现 Preemption？
- [ ] 是否存在调度不均衡？
- [ ] 是否发生 KV Cache 分配等待？

#### 第三层：框架层

- [ ] Model Runner 执行是否异常？
- [ ] 是否存在 CPU/NPU 同步？
- [ ] 是否存在数据拷贝？
- [ ] 是否存在 Python 调度开销？

#### 第四层：算子层

- [ ] Attention 耗时是否异常？
- [ ] GEMM 耗时是否异常？
- [ ] 是否存在算子性能退化？
- [ ] Kernel Launch 是否过多？

#### 第五层：硬件层

- [ ] NPU 利用率如何？
- [ ] HBM 带宽是否达到预期？
- [ ] 多卡通信是否存在瓶颈？
- [ ] 是否存在资源争抢？

### 3. Profiling 工具

- [ ] Python Profiler
- [ ] PyTorch Profiler
- [ ] Nsight Systems（用于理解 NVIDIA GPU 系统级 Profiling 方法）
- [ ] Nsight Compute（用于理解 NVIDIA Kernel 分析方法）
- [ ] 目标 NPU 厂商提供的 Profiling 工具

**注意：** Nsight 是 NVIDIA 工具，不能直接用于分析其他 NPU。实际分析应以目标硬件支持的工具为准。

**阶段交付物：**

完成一次真实 ITL 波动问题分析，形成报告：

```text
问题现象
   |
   v
数据采集
   |
   v
调用链路分析
   |
   v
耗时拆解
   |
   v
瓶颈定位
   |
   v
优化方案
   |
   v
实验验证
   |
   v
性能对比
```

报告至少包含：

- [ ] 问题背景和复现条件
- [ ] 测试环境与负载配置
- [ ] 原始性能数据
- [ ] 调用链路与耗时拆解
- [ ] 瓶颈判断及证据
- [ ] 优化方案
- [ ] 优化前后对比
- [ ] 结论及遗留问题

---

## 八、第七阶段：算子开发与综合实践（第 23～26 周）

最后一个阶段以动手实践为主，建议完成三个项目。

### 项目一：vLLM 调用链路分析

**目标：** 完整梳理一次 VLM 请求的执行流程。

交付物：

- [ ] 架构图
- [ ] 函数调用关系
- [ ] 关键模块说明
- [ ] 关键阶段耗时

### 项目二：ITL 性能优化

**目标：** 针对真实业务中的 ITL 波动进行分析。

交付物：

- [ ] Profiling 数据
- [ ] 瓶颈分析
- [ ] 优化方案
- [ ] 优化前后性能对比

### 项目三：实现一个简单算子

**目标：** 选择 RMSNorm 等相对简单的算子，完成从原理到目标 NPU 实践的学习。

步骤：

1. [ ] 理解数学原理。
2. [ ] 使用 PyTorch 实现 Reference。
3. [ ] 理解底层算子实现。
4. [ ] 在目标 NPU 环境开发或适配算子。
5. [ ] 进行精度验证。
6. [ ] 进行性能 Benchmark。

**最终交付物：** 一份完整的算子开发实践记录，包含数学定义、接口、实现方式、精度测试和性能数据。

---

## 九、半年学习进度总表

可以把下面的清单作为长期 Checklist，完成一项就勾选一项。

| 学习模块 | 具体内容 | 优先级 | 状态 |
|---|---|---|---|
| 操作系统 | 进程、线程、内存、锁 | 高 | [ ] |
| Linux | perf、gdb、strace | 高 | [ ] |
| 网络 | TCP、HTTP、RPC | 中 | [ ] |
| Python | 多线程、多进程、异步 | 高 | [ ] |
| C++ | 指针、内存、STL、多线程 | 高 | [ ] |
| PyTorch | Tensor、Dispatcher、自定义算子 | 高 | [ ] |
| Transformer | Attention、FFN、RoPE | 高 | [ ] |
| LLM | Prefill、Decode、KV Cache | 高 | [ ] |
| VLM | ViT、Vision Encoder、多模态 | 高 | [ ] |
| vLLM | Engine、Scheduler、Worker | 最高 | [ ] |
| vLLM | Model Runner、KV Cache | 最高 | [ ] |
| NPU | 硬件架构、内存体系 | 高 | [ ] |
| 算子 | GEMM、RMSNorm、Attention | 高 | [ ] |
| 编译器 | 编译、Kernel、Runtime | 高 | [ ] |
| 性能优化 | ITL、TTFT、Throughput | 最高 | [ ] |
| Profiling | 性能数据采集与分析 | 最高 | [ ] |
| 实践 | vLLM 调用链路分析 | 最高 | [ ] |
| 实践 | ITL 性能优化 | 最高 | [ ] |
| 实践 | NPU 算子开发 | 高 | [ ] |

---

## 十、学习方式与时间分配

### 1. 不要脱离工作学习

把工作中的问题当作学习入口：

- 遇到 ITL 波动，就研究 ITL 背后的原理。
- 遇到 KV Cache 问题，就研究 KV Cache 的源码实现。
- 遇到 NPU 算子问题，就研究算子调用链路。

让工作成为学习和验证的实验场。

### 2. 不要只看，要输出

每学一个模块，至少留下一个成果：

- 一张架构图
- 一份源码分析
- 一篇技术笔记
- 一个实验
- 一个性能优化案例

半年后，这些内容会成为自己的技术资产，也能帮助准备面试。

### 3. 学习时间分配建议

| 方向 | 时间占比 |
|---|---:|
| vLLM 源码及框架 | 30% |
| Transformer、LLM、VLM | 15% |
| C++、Python、PyTorch | 15% |
| NPU、算子、编译器 | 20% |
| 性能分析与优化实践 | 20% |

CUDA 建议先学习硬件架构和编程模型的通用知识；在当前 NPU 工作背景下，暂时不必把 CUDA Kernel 开发作为主要投入方向。

---

## 十一、最终职业能力目标

这条路线最终要建立的是一条完整的技术能力链：

**看懂模型 → 看懂框架 → 看懂硬件 → 理解算子 → 分析性能 → 独立优化。**

希望半年后，能够基于实际项目和数据，独立回答：

- 为什么 ITL 会突然变长？
- 为什么 NPU 利用率不高？
- 为什么 Attention 算子性能不理想？
- 为什么多卡推理存在通信瓶颈？
- 如何从框架层和算子层解决这些问题？

对于目标薪资 25K～30K 及以上，关键不在于把所有知识点都背下来，而在于能否证明自己具备独立分析问题、定位瓶颈、设计优化方案并验证效果的能力。

**最有价值的求职材料**是 1～2 个能够深入讲解的真实项目：明确问题、个人贡献、分析过程、技术决策，以及优化前后的量化结果。

---

## 附：每周学习记录模板

每周复制一份，持续记录学习和工作中的问题。

### 第 __ 周（日期：________）

**本周学习主题：**

- 

**本周完成内容：**

- [ ] 
- [ ] 
- [ ] 

**本周遇到的技术问题：**

- 

**源码 / 文档阅读记录：**

- 

**实践或实验结果：**

- 

**本周产出：**

- [ ] 架构图
- [ ] 技术笔记
- [ ] 代码或实验
- [ ] 性能分析报告

**尚未解决的问题：**

- 

**下周计划：**

- [ ] 
- [ ] 
- [ ] 
