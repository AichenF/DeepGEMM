# MiMo2.5-Pro NVFP4 八卡正确性与稳定性 gate

这个 gate 回答一个很具体的问题：在 **MiMo2.5-Pro 的真实 MoE 形状**下，最新版 grouped-nibble kernel 是否与标准 decoder 给出相同结果，并且在真实八卡跨 rank 路由时保持稳定。

它不是性能 benchmark，也不是整模型 E2E 精度测试。性能数字仍由受保护的 benchmark harness 产生；模型精度必须用真实 MiMo2.5-Pro 权重和数据集验证。

## 固定模型形状

- hidden size：6144
- intermediate size：2048
- 总专家数：384
- 每个 token 选择专家数：8
- GPU/rank 数：8
- 每个 rank 的本地专家数：48
- NVFP4 block N：256（标准 decoder 和 grouped candidate 都使用相同的权重分块）

这些值写成脚本常量，命令行不能悄悄换成更小、更容易通过的形状。

## 实际检查内容

每个 case 使用同一份随机 BF16 权重量化得到的 NVFP4 数据，构造两个独立存储副本：

1. **standard**：保留标准 80-byte row 布局，强制走标准 decoder。
2. **grouped**：只对 64-byte FP4 payload 做无损 nibble regroup，走最新版 grouped decoder；scale 和 padding 不变。

脚本检查：

- grouped 与 standard 默认要求逐值完全一致（`candidate_atol=0`）；
- AB/BA 交替重复运行，默认要求同一路径的重复结果逐值一致；
- kernel 输出全部为有限值，不允许 NaN/Inf 或未写出的 NaN；
- kernel 回报的每个本地专家收包数与全局路由表逐项一致；
- standard 输出与独立 PyTorch 参考计算比较。参考计算使用实际 FP8 输入值、实际 NVFP4 反量化权重、SwiGLU、route weight 和八 rank 汇总，不拿另一个 kernel 当参考答案；
- `uniform`：每个 token 向八个 rank 各发一个 route；
- `hot`：每个 rank 固定一个热门专家，制造严重热点，同时每个 token 仍跨八个 rank；
- `zipf`：按 Zipf 长尾分布采样，并保证每个源 rank 至少有一条远端 route；
- 默认覆盖 5 个小 M、3 种路由和 2 个 seed，共 30 个 case。

## 运行前提

正式证据必须来自一台具备以下条件的节点：

- 8 张 H200，单节点；
- SM90 软件栈可用；
- 八卡之间的双向 mapped P2P 已单独验证；
- 推荐 H200 SXM/NVSwitch。只有两个四卡 P2P 小组、跨组走 SYS 的 H200 NVL 节点不应作为上线 gate 机器；
- DeepGEMM 已按待发布 commit 构建，八个进程看到同一套构建产物。
- `DG_SM90_MOE_PHASE_PROFILE` 未开启；这个 gate 会把 48 项专家收包统计当作正确性结果，而不是 profiler 缓冲区。

脚本会检查八个 rank、SM90 和 H200 型号，但无法仅靠 PyTorch 名称证明 NVSwitch 拓扑，因此 mapped P2P preflight 仍是硬前置条件。

## 正式 gate

在仓库根目录运行：

```bash
python tests/test_nvfp4_mega_moe_mimo25pro_8rank_gate.py \
  --num-processes 8 \
  --batches 1 2 4 8 24 \
  --seeds 20260715 20260716 \
  --route-patterns uniform hot zipf \
  --repeats 2 \
  --reference-mode touched
```

只有进程退出码为 0、每个 case 都打印 `PASS`、最后打印 `30/30 cases`，才能记为通过。任何进程异常、hang、NCCL 错误、结果偏差或专家计数偏差都算失败。

## 开发期短检查

下面命令只适合快速发现明显错误，不能替代正式 gate：

```bash
python tests/test_nvfp4_mega_moe_mimo25pro_8rank_gate.py \
  --num-processes 8 \
  --batches 1 4 \
  --seeds 20260715 \
  --route-patterns uniform hot zipf \
  --repeats 1 \
  --reference-mode touched
```

`--allow-non-h200` 也只供开发环境使用；带这个参数得到的日志不能写成 H200 上线证据。

## 参考计算的内存选择

`--reference-mode touched` 是默认值。它每次只反量化当前 rank 实际收到 route 的专家，额外常驻内存较小，更适合可靠跑完整 gate；代价是不同 case 可能重复反量化同一专家。

`--reference-mode full` 会在每个 rank 常驻全部 48 个本地专家的 FP32 参考权重，单 rank 仅参考权重约 6.75 GiB，速度可能更快，但反量化过程还有明显的临时峰值。应先确认显存余量再使用。

两种模式计算的是同一个参考公式，不改变通过阈值。

## 日志中应保留的信息

每个 `PASS` 行会记录：

- seed、路由类型和 M；
- 实际远端 route 数、触达专家数、最热专家负载；
- grouped 与 standard 的最大/平均绝对差；
- standard 与独立参考的最小/平均 cosine、整体 norm ratio、最大绝对差；
- 权重准备时所有 rank 中最大的显存峰值。

这些字段用于区分“真实跨卡且结果正确”与“只在本卡、只跑一个 seed 或只看是否崩溃”的弱测试。
