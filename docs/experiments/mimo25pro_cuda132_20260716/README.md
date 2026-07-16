# MiMo 最终版 NVFP4 MegaMoE Kernel：CUDA 13.2 性能与上线边界报告

- 日期：2026-07-16
- MiMo 正式性能证据：[evidence/perf](evidence/perf/)
- DSR1 初步精度证据：[evidence/dsr1_preliminary](evidence/dsr1_preliminary/)
- ImagePerf 参考图：[image_perf.png](image_perf.png)

## 一、先说结论

最终版在 MiMo-V2.5-Pro 的 11 个 M 点上完成了同节点、同 CUDA 13.2 容器、每点 30 个独立进程的正式测试。以 ImagePerf 同口径的 `rank0` 延时为主指标，11 点等权几何平均相比 Aichen baseline **降低 2.779%**；按时间倒数换算是 `1.02858×`，即约 2.858% speedup。相比 ImagePerf 的 11 点等权几何平均是 **-0.093%**，整体几乎重合。

这两个汇总数不能掩盖逐点差异：

- M=8、16、32 分别比 baseline 快 10.42%、11.14%、9.57%，小 M 改进很明显。
- M=64、128、512 也更快，但幅度较小。
- M=256、1024、4096 基本持平，差异只有 +0.30%、+0.03%、+0.05%。
- M=2048 慢 1.15%。
- **M=8192 明确回退 3.56%**。这一点不能用整体平均数消掉；如果生产流量可能到 M=8192，必须单独处理或明确接受该回退。
- 最终版虽然与 ImagePerf 的几何平均只差 -0.093%，逐点按 ±2% 判断仍只有 2/11 点对齐。这是快慢点相互抵消后的整体结果，不等于整条曲线逐点复现。

本轮性能实验的结构、原始日志和复算均通过：30 个正式进程完整保留，没有失败重试、删慢样本或 best-of-N；独立复算与正式 CSV 逐值零差异；129 个归档文件 hash 通过；测试前后抽取的 432 个 NVLink 错误计数和 80 个 ECC/恢复相关字段均为零或没有增长。

但“最终版”必须准确理解为下面这个实际执行件：

```text
DeepGEMM 75186dde9dac140c053c9007ace0ce7cce41150c
+ cuda132-dependent-template-compat-3files-16sites
```

**原始、未打补丁的 `75186dd` 不是 CUDA 13.2 上线件。** 它在 3 个文件的 16 个模板调用处缺少 CUDA 13.2 所需的 `.template` 语法。正式测试执行的是已经封存、可复核 hash 的 16 处兼容补丁版本，必须把这个补丁烘进部署制品，不能只写“部署 75186dd”。

当前可以给出的上线判断是：

这里的 RC（Release Candidate）指“可进入部署验收的发布候选”，不等于整套服务已经获准全量上线。

| 范围 | 当前判断 |
|---|---|
| CUDA 13.2、8×H200、MiMo 固定 shape 的核心 kernel 性能 | **RC 通过**，但 M=8192 回退需有明确处置 |
| `75186dd` 原始源码直接部署到 CUDA 13.2 | **禁止**；必须使用 16-edit patched artifact |
| DeepGEMM 与 SGLang 集成 | 必须成对固定 `75186dd+补丁` 与 `b6a68c9` |
| DSR1 初步整模精度回归 | **通过**：完整 1316 题 warmup 独立复算为 1266/1316（96.2006%），invalid=0、输出损坏特征=0；正式 timed 轮未启动，因此不属于完整 E2E 验收 |
| DSR1 开启 EPLB（动态专家负载均衡） | **禁止**；DSR1 当前 mover 返回 source 权重，而 kernel 读取 packed 权重，在线移动后可能不一致 |
| MiMo 开启 EPLB | 当前发布范围也保持关闭；MiMo 已有 packed-aware mover，但在线迁移整模 E2E 尚未完成，不能外推为已验证 |
| MiMo-V2.5-Pro 整模上线 | **仍是条件性 NO-GO**；真实 MiMo checkpoint 的 69 层加载、HBM 显存、文本精度、CUDA Graph、长稳（soak）和小流量灰度（canary）尚未完成 |

因此，这不是“所有上线门禁都已关闭”的报告。它证明了核心 kernel 在当前窄范围内的性能和可复现性，也给出了 DSR1 初步整模精度回归没有异常的证据；但 DSR1 正式 timed 轮未完成，而且 DSR1 代理通过不能替代真实 MiMo 整模验收。

## 二、版本到底是什么

### 2.1 Baseline

Baseline 是 Aichen 原始 NVFP4 MegaMoE 分支提交：

```text
ba7ee0944c1fe31874b049ae354657ff62dae20b
+ cuda132-dependent-template-compat（2 个文件、10 处纯语法修改）
```

它使用仓库原本的自动策略：M≤1024 走 `BN256-standard`，M≥2048 走 `BN128-split`。`BN` 可以简单理解为一次 kernel 在输出维度上处理的块宽；不同 BN 和执行路径会影响不同 M 下的效率。

### 2.2 最终版

最终版源码基线为：

```text
commit: 75186dde9dac140c053c9007ace0ce7cce41150c
tree:   7d362b6f6069164fc326faddb2a99e8b158bb5cf
branch: codex/mimo25pro-release-20260715
```

向 Aichen 仓库交付时不修改原 `megamoe_nvfp4` 分支，而是建立同级独立分支：

```text
branch:             megamoe_nvfp4_mimo25pro_cuda132_final
CUDA 13.2 code commit: a8f17ad58f97b49b33ed63a4ed7a7304c897d4af
CUDA 13.2 code tree:   80d517297b932911f785b6ffaf1b6adf608f73c8
```

其中 `a8f17ad` 只包含下面 3 个文件、16 处 `.template` 兼容修改；后续报告提交不会改变这个已测试 kernel code tree。原 `megamoe_nvfp4` 继续保持在 `ba7ee094`，没有覆盖或强推。

CUDA 13.2 实际执行件还包含封存的纯语法兼容补丁：

| 项目 | 固定值 |
|---|---|
| 补丁范围 | 3 个 kernel 文件、16 处替换 |
| 修改内容 | `.get_base_ptr<float>()` → `.template get_base_ptr<float>()` |
| 补丁 SHA-256 | `333154d5254467c5e4a399d5286b414c405dca7566f927f93cca76ea30fcb07d` |
| 补丁前封存源码 tar SHA-256 | `401f50a608becd12b486478f65e6d2ae2a427323e83661c1bd92baf50e70c904` |
| 算法、数学、内存布局和 block-N 是否由补丁改变 | 否 |
| 部署身份标记 | `stock_source=false` |

补丁本身不做性能优化，只让 CUDA 13.2 按正确的 C++ 模板语法解析同一调用。封存前后各有 6980 个文件 hash，只有补丁声明的 3 个文件发生变化。便携身份说明见 [SOURCE_IDENTITY.md](SOURCE_IDENTITY.md)，测试时使用的补丁见 [兼容补丁](evidence/perf/final_deepgemm-75186dd-cuda132-dependent-template-compat.patch)。

最终版对 MiMo 固定使用 `BN256-grouped`。`grouped` 指权重在计时前经过一次无损的 nibble 顺序重排，并由对应的 grouped decoder 读取；nibble 是 4 bit 数据单元。Baseline 同样会在计时前预打包，因此本轮不能把收益解释成“最终版省掉了 baseline 每次调用都做的预打包”；测到的是两种稳态数据布局和 active kernel 路径的差异。

这次比较的是“baseline 的实际默认策略”对“最终发布策略”，不是只替换一行代码的消融实验。因此计时结果包含 grouped 布局、active kernel 的流水线/同步变化和 BN 策略变化，不能把全部 2.779% 都归因于某一项修改。JIT 编译、权重预打包、source 释放、生命周期检查和 SGLang 服务均在计时范围外；它们属于发布加固，不能用这张性能表证明其速度收益。

### 2.3 与 SGLang 的固定配对

部署时 SGLang 也不能随便换版本。当前审查过的配对版本是：

```text
DeepGEMM: 75186dd + CUDA 13.2 16-edit compatibility patch
SGLang:   b6a68c9acb6590b2849febe2b66807553923fc71
```

两边要一起固定绑定（pin）。单独升级 DeepGEMM 或 SGLang，都需要重新跑集成、精度和生命周期门禁。

## 三、最终版主要做了什么

下面用尽量直白的方式概括从 baseline 到最终版的关键变化。

1. **修复 GPU 异步流水线中的潜在竞态。** 权重搬运、解量化和矩阵计算由不同硬件阶段协同执行。旧代码中有些“前一阶段已经可见”的保证不完整，极端时序下可能读到尚未完全发布的数据。最终版补齐了 proxy fence 和阶段同步。
2. **为 MiMo 的 I=2048 路径加入 grouped-nibble 权重布局。** 权重在加载时做一次无损 nibble 重排，kernel 通过 grouped decoder 直接消费已打包的 `mega_l1/mega_l2`。本轮没有单独做“只改布局、其它代码完全不变”的消融实验，所以不对这一项单独归因。
3. **固定 MiMo 使用 BN256 grouped 策略。** 这使小 M 明显变快，但也解释了为什么大 M 必须单独观察：baseline 在 M≥2048 会切到 BN128，而最终版不切换。
4. **完善预打包权重的生命周期。** 增加 source 释放、版本检查、非法 scale 拒绝和失败时关闭路径，避免权重状态半新半旧。
5. **降低冷启动重复 JIT 编译。** JIT 是第一次遇到某个 shape 时临时编译 GPU kernel。当前实现对已覆盖的多进程启动和 Python GIL 场景做了并发保护；不过这不等于任意跨容器、跨版本共享 cache 都安全，部署仍必须使用按版本隔离的干净 JIT cache。
6. **明确 expert mover 的复制契约。** MiMo 的 mover 必须复制完整 80B packed expert row（包括布局 marker），并使用会增加 PyTorch `_version` 的 `copy_`，这样下一次使用前会重新校验布局状态；`_version` 变化本身不会自动重建 packed 权重。DSR1 当前 mover 仍返回 source 参数，而 runtime kernel 读取 packed 权重，因此 DSR1 必须关闭在线 expert move/EPLB。

还有一个容易写错的性能解释：代码审查确认，BN256 当前实际执行的是 **math-side dequant**，即在数学计算一侧应用反量化比例；所谓 loader-dequant 分支在这个配置下是死分支。报告不能把本轮提升归因于一个没有实际运行的 loader-dequant 分支。

## 四、测试环境和工作量

| 项目 | 正式配置 |
|---|---|
| 调度范围 | 单一内部 Slurm allocation、同一节点完成两版测试；主报告不记录内部作业号和主机名 |
| GPU | 8 × NVIDIA H200，固定同一 GPU UUID 顺序 |
| GPU 互联 | 全 NVSwitch；56/56 个有向 mapped P2P 对通过 |
| 驱动 | `595.58.03` |
| 容器 | `nvcr.io/nvidia/pytorch@sha256:f572dd504a3fef02277c21f228977f100f7831576ac73140a250c473f74d3ad3` |
| 容器发行版 | NVIDIA PyTorch 26.03 |
| PyTorch / CUDA | PyTorch `2.11.0a0+a6c236b`；CUDA 13.2 |
| 实际 JIT 编译器 | `/usr/local/cuda/bin/nvcc`，CUDA 13.2 V13.2.51 |
| NCCL | 2.29.7 |
| MiMo shape | hidden=6144，intermediate=2048，experts=384，top-k=8，ranks=8 |
| M 点 | 8、16、32、64、128、256、512、1024、2048、4096、8192 |

`M` 是每个 rank/GPU 在路由前的 token 工作量，不是 8 卡总和。`recv` 是固定随机路由后 rank0 真正收到的 token 数，用来确认两版跑的是同一份工作负载。

这是一组 standalone DeepGEMM microbenchmark：输入是固定随机种子的合成权重和固定随机路由，不是真实 MiMo checkpoint，也不包含 tokenizer、SGLang 调度、网络请求或整模推理。它直接回答 kernel 稳态延时问题，不能单独回答服务端到端延时和文本精度问题。

这里的 `rank0` 也不是“只测一张卡”。每个进程始终由 8 张 H200 共同执行；`rank0` 只是取第 0 个分布式进程对应 GPU 的 kernel 时间，与 ImagePerf 图的主数值口径一致。报告也保留了 8 卡平均和最慢卡结果，最慢卡 11 点几何平均改善 2.749%，说明 rank0 的整体改善没有被明显的跨卡拖尾反转。

代码审查发现 DeepGEMM 的 CUDA 搜索逻辑在某些环境中可能先找到 `/usr/local/cuda-13.0`。因此只写“容器是 CUDA 13.2”不够。本次 runner 另外核对了 DeepGEMM JIT 真正调用的 nvcc，正式记录为 CUDA 13.2 V13.2.51，见 [deep_gemm_cuda_resolution.log](evidence/perf/final_deep_gemm_cuda_resolution.log)。生产镜像也必须保留这一显式检查。

## 五、测试方法

### 5.1 最终版

1. 先启动 1 个完整进程，把 11 个 M 全部跑一遍作为预热；这份数据排除，不进入统计。
2. 再启动 30 个全新的完整 Python 进程。每个进程都按相同顺序跑完 11 个 M。
3. 每个 M 在 benchmark 内执行 `--num-tests 20`，日志打印 20 次完整 MegaMoE 调用的平均值。BN256 每次调用是 1 个 kernel；baseline 的 BN128-split 每次调用包含 L1、L2 两个 kernel，harness 把两段时间合成为一次调用时间。
4. 对同一个 M 收集 30 个独立进程结果，排序后取中位数。
5. 30 轮全部原样保留，没有失败重试、删除、替换或选最好值。
6. 正式测量前完成一次 JIT，预热后到 30 轮结束的 JIT cache hash 完全一致，避免把编译时间混进 kernel 时间。
7. benchmark 没有强制 `--nvfp4-block-n`，实际走最终版 release-auto 策略；所有额外 `DG_*` 调优变量都被拒绝，只允许单独指定隔离的 `DG_JIT_CACHE_DIR`。

完整命令见 [benchmark_command_mimo.txt](evidence/perf/final_benchmark_command_mimo.txt)，31 个进程的时间、退出码和日志 hash 见 [process_ledger.tsv](evidence/perf/final_process_ledger.tsv)。

### 5.2 Baseline 与最终版如何比较

Baseline 和最终版都来自 30 个完整进程、相同 11 个 M、相同 `--num-tests 20`、相同 benchmark harness，并在同一个 Slurm 作业、同一节点、同一组 GPU 和同一个 CUDA 13.2 容器中完成。统计器不是直接复用旧汇总表，而是重新读取两边的原始日志后统一计算。

但实验顺序并非严格交错 A/B：

- baseline 的 30 个 MiMo 进程夹在 Flash、Pro、MiMo 三模型轮转测试中；
- 最终版的 30 个 MiMo 进程是之后连续执行的；
- 因此这是“同节点、同配置的两组 30 轮中位数比较”，不是 A/B/B/A 或逐轮配对实验。

另外，runner 没有锁定 GPU 时钟，A/B 又没有交错执行。因此置信区间只能描述各组内部的随机散布，不能排除测试时序、温度或动态频率带来的组间系统差异。这不会让原始数字失效，但会影响差异的解释。像 M=256 的 +0.30%、M=1024 的 +0.03% 和 M=4096 的 +0.05%，应视为基本持平，不能写成确定回退；同理，小幅提升也不应夸大。M=8/16/32 的约 10% 改进和 M=8192 的 3.56% 回退远大于这些微小漂移，更值得作为信号处理，但若要把其幅度作为严格发布门槛，仍应补一次锁频、交错 A/B 复验。

还有一个真实服务差异：microbenchmark 会针对每个 M 新建并预打包一份合成权重，所以 baseline 能在不同 M 分别测试 BN256 和 BN128。整模服务通常在加载时只预打包一种静态布局，不能随着每个请求的 M 零成本切换；若同时保留两套布局，还会增加 HBM（显存）和加载开销。本报告比较的是两版各自的 benchmark 默认策略，不代表生产服务可以直接做免费的逐请求策略切换。

## 六、MiMo 11 点完整性能结果

以下时间单位都是微秒（µs）。“最终 vs baseline”为负表示最终版更快；正值表示最终版更慢。`最终 max-rank 中位数` 直接来自实验时保存的同一批 30 轮日志：每个正式进程先取 8 个 rank 中最慢 rank 的 kernel 时间，再对 30 个进程的该值取中位数；它不是 8 张卡耗时之和。两个百分比列仍按 rank0 计算，不能拿 max-rank 与 ImagePerf rank0 混算。

| M | recv | baseline 布局 → 最终布局 | ImagePerf rank0 | baseline rank0 中位数 | 最终 rank0 中位数 | 最终 max-rank 中位数 | 最终 vs baseline | 最终 vs ImagePerf |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 8 | 64 | BN256-standard → BN256-grouped | 491.4 | 503.55 | 451.10 | 460.60 | **-10.42%** | -8.20% |
| 16 | 118 | BN256-standard → BN256-grouped | 536.4 | 576.05 | 511.85 | 519.05 | **-11.14%** | -4.58% |
| 32 | 241 | BN256-standard → BN256-grouped | 554.7 | 578.55 | 523.20 | 531.85 | **-9.57%** | -5.68% |
| 64 | 546 | BN256-standard → BN256-grouped | 568.8 | 550.75 | 541.95 | 557.60 | -1.60% | -4.72% |
| 128 | 984 | BN256-standard → BN256-grouped | 524.4 | 570.50 | 566.15 | 570.60 | -0.76% | +7.96% |
| 256 | 2037 | BN256-standard → BN256-grouped | 540.2 | 563.55 | 565.25 | 573.30 | +0.30% | +4.64% |
| 512 | 4097 | BN256-standard → BN256-grouped | 1002.2 | 1035.00 | 1027.00 | 1032.50 | -0.77% | +2.47% |
| 1024 | 8140 | BN256-standard → BN256-grouped | 1500.1 | 1547.00 | 1547.50 | 1553.00 | +0.03% | +3.16% |
| 2048 | 16409 | BN128-split → BN256-grouped | 2852.3 | 2862.00 | 2895.00 | 2899.00 | +1.15% | +1.50% |
| 4096 | 32686 | BN128-split → BN256-grouped | 5337.9 | 5450.50 | 5453.00 | 5458.00 | +0.05% | +2.16% |
| 8192 | 65538 | BN128-split → BN256-grouped | 10398.5 | 10195.50 | 10558.50 | 10571.50 | **+3.56%** | +1.54% |

机器可读的完整结果见 [comparison_baseline_vs_final_30.csv](evidence/perf/final_comparison_baseline_vs_final_30.csv)，其中还包含每点 P10–P90（第 10 到第 90 百分位范围）、30 轮中位数置信区间、8 卡平均和最慢卡中位数；660 个 baseline+最终版逐进程数值见 [individual_baseline_vs_final_30.csv](evidence/perf/final_individual_baseline_vs_final_30.csv)。

### 6.1 汇总指标

| 指标 | 最终版相对 baseline | 最终版相对 ImagePerf |
|---|---:|---:|
| rank0，11 点等权几何平均延时 | **-2.779%** | **-0.093%** |
| 8 rank 平均时间，11 点等权几何平均 | -2.824% | — |
| 8 rank 最慢时间，11 点等权几何平均 | -2.749% | — |

“等权几何平均”是先算每个 M 的相对比值，再让 11 个 M 各占相同权重。它适合概括整条曲线，但不代表真实线上请求分布。如果线上绝大多数请求集中在某几个 M，最终还需要按真实流量加权；本报告没有用未知的线上流量比例替代实验事实。

### 6.2 怎么理解这条曲线

小 M 是最终版最明确的收益区间。M=8、16、32 的 rank0、8 卡平均和最慢卡三个口径都接近 10% 改进，说明收益不是 rank0 偶然快了一张卡。

M=64 到 4096 大部分点是小幅改进或基本持平。M=2048 的 +1.15% 需要保留，但还没有达到 M=8192 那样明确的回退程度。

M=8192 的 baseline 中位数区间为 10183–10202 µs，最终版为 10548–10575 µs，两段不重叠，说明它远比千分之几的点更值得关注；但因为没有锁频和交错 A/B，它仍不能完全排除跨时段系统漂移。一个合理线索是 baseline 在大 M 切换到 BN128-split，而最终版始终保持 BN256-grouped；本实验没有单独拆分所有变量，因此只能说这个策略差异与回退同时出现，不能直接宣布唯一根因。

若生产允许 M=8192，有三种合规处理方式：补充大 M 策略并重新跑完整回归（若保留双布局，还要核算额外 HBM 和加载成本）；以真实业务加权 E2E 证明整体收益且明确接受该点；或者在服务配置中硬性限制 M 上限并监控。不能一边允许 M=8192，一边只引用 -2.779% 的平均值忽略这个点。

## 七、结果为什么可信

本次“可信”指实验执行和统计可复核，不等于所有上线门禁已通过。

- 1 个排除预热和 30 个正式进程全部退出 0；30×11=330 个最终版正式数据点齐全。
- baseline 的 30 个原始 MiMo 日志也重新校验并参与统一复算，两边合计 660 个逐进程数值。
- 每份日志都有固定 shape、M 顺序、`recv` 和触达专家数检查，防止跑成别的工作负载。
- 最终版 runner 的 66 次、baseline runner 的 190 次 GPU 空闲检查全部通过，没有其它计算进程插入。
- 受保护 benchmark harness、运行源码和 JIT cache 在正式测量前后 hash 一致。
- 固定 GPU UUID 顺序，56/56 mapped P2P 通过。
- 独立完整性审计重新校验 129 个归档 hash；测试前后 432 个 NVLink 错误计数与 80 个 ECC/恢复相关字段为零或没有增长。
- 独立统计没有调用正式汇总器，而是重新解析 raw log；逐进程 CSV、中位数、区间和差异字段与正式结果零差异。
- 性能数值没有被当作脚本退出门槛。正式状态是 `protocol=COMPLETED`、`structural=PASS`、`performance=REPORTED_NOT_GATED`，所以不会因为某点不够快就被脚本丢弃或重跑。

状态文件分别见 [protocol_status.txt](evidence/perf/final_protocol_status.txt)、[structural_status.txt](evidence/perf/final_structural_status.txt) 和 [performance_status.txt](evidence/perf/final_performance_status.txt)。本次实际发布文件的 hash 见 [SHA256SUMS](SHA256SUMS)。

## 八、失败预检和未采用数据的透明披露

### 8.1 原始 `75186dd` 不是 CUDA 13.2 可直接部署版本

CUDA 13.2 兼容审查发现 3 个 kernel 文件共 16 处 dependent-template 调用需要显式 `.template`。其中 standard fused 6 处、split-L1 4 处、最终版新增 grouped-nibble fused 6 处。只复用 baseline 的 10 处兼容补丁会漏掉 grouped 路径，因此也不是正确的最终制品。

处理方式不是掩盖这个差异，而是先封存 byte-exact `75186dd`，再应用一份单独封存的 16-edit 纯语法补丁，分别核对补丁前后全部文件 hash。性能表来自 patched artifact。任何只拿 raw commit、漏掉补丁或只打 10 处补丁的 CUDA 13.2 结果，都不能与本报告混用。

### 8.2 第一次正式 runner 尝试在采样前失败

正式运行前有一次未完成的 orchestration 预检。脚本因 shell 变量 `phase` 未绑定，在源码解包、编译、warmup 和正式采样之前就退出，状态为 `NOT_COMPLETED / EXECUTION_FAIL`；ledger 为 0 个 sample，没有任何数据被并入 30 轮结果。

修复 runner 后，使用全新归档从源码恢复、补丁核对、构建、预热到 30 轮全部重新执行。这里披露前一次失败，是为了区分“测试编排错误”和“kernel 性能/正确性失败”，也防止未来误把两个目录的数据拼在一起。

## 九、代码审查发现的部署约束

### 9.1 必须执行的硬约束

1. **CUDA 13.2 只能部署 patched artifact。** raw `75186dd` 在本环境不是上线件；制品清单必须同时记录 commit、补丁 SHA、源码 tar SHA 和构建镜像 digest。
2. **DeepGEMM 与 SGLang 成对固定。** 当前认可组合是 `75186dd+16-edit patch` 与 `b6a68c9`，不能只固定（pin）一边。DeepGEMM 的特殊布局选择主要按几何 shape 判断，可能命中同几何的其它模型；SGLang `b6a68c9` 的 owner-aware 检查才把 release 布局约束到 MiMo owner，二者缺一不可。
3. **DSR1 禁止开启 EPLB。** EPLB 是运行中根据负载移动专家的功能。DSR1 当前 `get_moe_weights()` 返回 source 参数，但 kernel 实际读取预打包的 `mega_l1/mega_l2`；两者在移动后可能不一致并产生静默错误。本轮 DSR1 E2E 和 DSR1 生产必须保持关闭，直到 packed 权重跟随移动并完成专项正确性验证。
4. **MiMo 本发布范围也暂不启用 EPLB。** MiMo 已有返回 packed runtime tensor 的专用 mover，不是上一条相同的 source/packed bug；但在线迁移整模 E2E 尚未完成，所以当前仍按未验证功能关闭。
5. **自定义 expert mover 必须满足 packed-row 合同。** 它必须复制完整 80B packed expert row（包括布局 marker），并使用会增加 `_version` 的 `copy_`，让下一次调用重新校验布局。`_version` 变化不会替用户自动从 source 重建 packed 权重；任何只复制局部 tensor 或只改 source tensor 的 mover 都不在已验证范围内。
6. **显式验证 JIT 使用 CUDA 13.2。** 不能只依赖 `PATH` 或镜像标签；启动时要核对 DeepGEMM 最终解析到的 nvcc。
7. **JIT cache 按版本隔离并从干净状态建立。** 当前并发安全证据覆盖已测的多进程/Python GIL 场景，不应外推到跨版本共享同一 cache。cache key 至少要区分 DeepGEMM commit+patch、CUDA、GPU 架构和关键 shape/策略。
8. **加载失败必须重启进程。** 当前整模加载和预打包不是事务操作；中途失败后，内存中可能留下部分已转换、部分未转换的状态。不能在同一进程里直接重试并继续接流量。
9. **清理 benchmark/debug/tuning 环境变量。** 生产使用明确 allowlist，不继承调试时的 `DG_*`、强制 block-N、输出复用或 profiler 变量。

### 9.2 尚未覆盖的运行方式

- in-process hot restart 没有验证；当前安全做法是进程级冷重启。
- 开启 EPLB 的 expert move + packed 权重一致性没有验证。
- 任意第三方 custom mover 没有验证，除非满足 full-row 和 `_version` 合同并通过专项测试。
- 跨版本、跨容器复用 JIT cache 没有验证。
- M=8192 的生产策略尚未定案。

### 9.3 当前代码层面已经有的证据

发布候选此前完成了 30 种路由负载的数值一致性、8 个受保护数值用例、10 个权重状态/生命周期用例、8 个布局策略用例、25 个 SGLang 接入用例，以及单层 48 个本地专家的实卡构建验证。这些门禁都在固定 SGLang 镜像的 CUDA 13.0.1 环境完成，不是 CUDA 13.2 patched artifact 的直接数值 reference check。16-edit 补丁只改 C++ 模板解析语法，因而旧证据对算法有参考价值，但不能包装成已经在 CUDA 13.2 上逐值重跑。它们支持“核心 kernel + 固定 SGLang + 静态专家”的 RC 判断，仍不能代替真实 69 层模型服务。

还有一个精度边界：DSR1 的 scale-refold 路径没有 MiMo 路径使用的“最大相对误差灾难阈值”保护。下面的完整 warmup 精度回归通过，也只是全局文本回归证据，不是每个 block 最坏误差都已被证明安全。

## 十、DSR1 初步整模精度复测

> **状态：初步精度回归通过；完整正式 E2E 验收未完成。** 一轮完整 1316 题 warmup 已跑完并通过逐题独立复算。随后 runner 因 JIT cache 证据目录为空而退出，正式 timed 轮没有启动。根据本次“初步精度无异常即可停止”的验收范围，不再为取得 timed 数值重跑。

### 10.1 实际运行身份

| 项目 | 实际值 |
|---|---|
| 归档 | [DSR1 初步精度证据包](evidence/dsr1_preliminary/)；公开材料不记录内部作业号 |
| GPU | 8×H200；8 个预检 GPU UUID 全部出现服务进程，公开材料不记录具体 UUID |
| DeepGEMM | `75186dde9dac140c053c9007ace0ce7cce41150c`，运行时扩展 SHA-256 `6a41e4aac5a7c4f98725d71928064bad5b30a3798807d2b6a185440c224c8e2d` |
| SGLang | `b6a68c9acb6590b2849febe2b66807553923fc71` |
| 镜像 | `lmsysorg/sglang@sha256:5027e95bf6ec536856b1b52a91d1f35ff5c564ab83e8a94758a169ff09bb8df3` |
| E2E CUDA 边界 | CUDA 13.0.1 / torch cu130，**不是 CUDA 13.2** |
| DSR1 checkpoint | `DeepSeek-R1-0528-FP4-v2`；163 shards，413,328,348,544 bytes |
| Kernel 路径 | 8 ranks × 58 个 MoE 层，共 464 条构建记录；layer 3..60 全覆盖，全部 `block_n=256, grouped_nibbles=True` |
| EPLB | `enable_eplb=False` |

这里必须区分两个制品：MiMo 性能数据来自 CUDA 13.2 的 `75186dd + 16-edit patch`；DSR1 初步 E2E 使用固定的 CUDA 13.0.1 SGLang 镜像和 raw `75186dd`。16 处补丁只改变 CUDA 13.2 的 dependent-template 解析语法，不改变数学路径，但本轮 DSR1 仍不能冒充“CUDA 13.2 patched binary 的整模 E2E”。

### 10.2 初步精度结果

固定协议为 GSM8K 8-shot、1316 题、并发 1316、temperature 0、top-p 1、max-new-tokens 512。冻结数据集 SHA-256 为 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。

| 项目 | 结果 |
|---|---|
| 完成题数 | 1316/1316 |
| 独立复算正确数 | 1266/1316 |
| 精确 accuracy | 0.962006079（96.2006%） |
| 预注册门槛 | 1260/1316；实际高 6 题 |
| 错误题数 | 50 |
| invalid output | 0 |
| 已知重复、乱码或非法控制字符特征 | 0 |
| prompt ID / prompt 文本 / 存档 correct 标志不一致 | 0 / 0 / 0 |
| warmup 耗时 / 输出吞吐 | 363.423 s / 390.171 token/s；**排除轮，不能作为正式性能** |
| raw / result / log SHA-256 | `225b08acc512250767e9ca4243565932b282eda2a46c85d9c4976bd8982274f0` / `0b4d67d3b532a884156b4bd3b62ac778d02c86093d7e3ab7fe40d3e2a0b8c44b` / `ed8552f6a575e78e54f90d0d578cb74cb9e35d04f2798e30977af06fae036083` |
| 初步精度结论 | **PASS** |

独立复算从冻结 dataset 重新生成 1316 个 prompt 和答案，再从 raw output 提取最后一个数字；不依赖 result JSON 的 `accuracy`，也不只信 raw 中的 `correct` 字段。归档中 81 条文件 hash 重新执行 `sha256sum -c` 全部通过。

### 10.3 为什么整个 run 仍显示 EXECUTION_FAIL

这次 run 的状态是 `EXECUTION_FAIL / NOT_COMPLETED`，最终 phase 为 `warmup_round`。原因不是模型推理失败或低分：完整 warmup 和逐题 validator 都已成功。失败发生在 warmup 之后的证据门，runner 观察的 host `dg_jit_cache` 目录为空，无法封存 grouped-nibble JIT `.cubin`，因此在启动正式 timed 轮前主动退出。

所以本节只能下“初步全模型精度回归通过”的结论，不能下列结论：

- 不能称完整正式 E2E 验收 PASS；
- 不能把 363.423 s 当成正式延时；
- 不能声称 CUDA 13.2 patched binary 已通过 DSR1 整模测试；
- 不能用 DSR1 代理替代真实 MiMo-V2.5-Pro 的 69 层 checkpoint、路由和文本精度验收。

### 10.4 数据集口径说明

历史兼容脚本把 test split 前 8 题同时当作 8-shot 示例，并仍把它们计入 1316 题。这 8 题本轮为 8/8；事后剔除后是 1258/1308，即 96.1774%。因此本测试定位为固定版本回归代理，不是无泄漏的官方 GSM8K 成绩。

## 十一、距离 MiMo-V2.5-Pro 真正上线还缺什么

下面这些门禁仍未被本轮 kernel 性能或 DSR1 代理测试覆盖：

- 真实 ModelOpt NVFP4 MiMo checkpoint 与所有 shard hash；
- 69 层完整加载，以及加载期间峰值和稳定态 HBM；
- 真实 MiMo 文本 E2E 精度与 invalid-output 检查；
- 生产 CUDA Graph capture/replay；
- 冷启动、暖启动、进程重启、加载失败后的恢复和回滚演练；
- 整模长时间 soak；
- M=8192 的策略决定和相应回归；
- 全部门禁通过后的真实 MiMo service canary。

建议把发布状态分成两层，避免一句“可上线”混淆范围：

1. **Kernel RC：** `75186dd + CUDA 13.2 16-edit patch`，固定 `b6a68c9`、静态专家、EPLB 关闭；MiMo 11 点几何平均延时降低 2.779%（`1.02858×`），M=8192 回退 3.56% 已知。
2. **MiMo service release：** 只有上面的真实整模门禁全部关闭后，才能从条件性 NO-GO 改为 canary GO；canary 通过后才能讨论正式上线。

## 十二、可复核材料索引

- [源码与执行身份](SOURCE_IDENTITY.md)
- [正式性能汇总 Markdown](evidence/perf/final_comparison_baseline_vs_final_30.md)
- [正式性能汇总 CSV](evidence/perf/final_comparison_baseline_vs_final_30.csv)
- [正式性能汇总 JSON](evidence/perf/final_comparison_baseline_vs_final_30.json)
- [两版 660 个逐进程数据](evidence/perf/final_individual_baseline_vs_final_30.csv)
- [最终版 30 轮原始日志](evidence/perf/final_raw/)
- [baseline 30 轮原始日志](evidence/perf/baseline_raw/)
- [最终版进程账本](evidence/perf/final_process_ledger.tsv)
- [baseline 三模型进程账本](evidence/perf/baseline_process_ledger_all_three_models.tsv)
- [实际 CUDA/JIT nvcc 解析](evidence/perf/final_deep_gemm_cuda_resolution.log)
- [DSR1 初步精度结果与复算说明](evidence/dsr1_preliminary/README.md)
- [DSR1 1316 条压缩原始输出](evidence/dsr1_preliminary/gsm8k_raw_warmup.jsonl.gz)
- [本发布包 SHA-256 清单](SHA256SUMS)
