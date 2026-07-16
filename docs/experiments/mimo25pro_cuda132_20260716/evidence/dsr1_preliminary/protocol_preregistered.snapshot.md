# DeepSeek-R1 最终版整模 E2E 验证协议（预注册）

日期：2026-07-16  
目标：只验证最终 kernel 在真实 DeepSeek-R1 服务链路中的任务精度与异常输出，不把 kernel 单测冒充整模 E2E，也不把本轮 E2E 延时当成 MiMo kernel 微基准。

## 0. 协议修订记录

- `v1` 原始协议 SHA256：`5c137ccd6f5277ba6f7e368b87502650237adc717a7b98b23cc577d784900048`。
- `v1` 首次执行归档：`job3119858_20260716_061351`。该次执行已完成源码恢复、编译和 413 GB 模型加载，服务也已就绪，但在任何 GSM8K 请求发出前，GPU 进程取证使用的辅助容器因 PID namespace 隔离得到空结果；随后 shell 的 `pipefail` 令 runner 直接退出。清理阶段还尝试覆盖只读数据集副本并产生了无关的 `Permission denied`。该归档没有 warmup、timed、raw 或 score 文件，状态永久保留为“计分前基础设施失败”，不能视为一次低分后的重跑。
- `v2` 协议 SHA256：`bc14dc495545696cf1dfe78cbccc2799a55e270e9bb5f0762caafd20a001e925`；runner SHA256：`978b0504e1a88c3d8ac5d2b54fec7ffc4ca78b72fa85db5c910bef619b7554ec`。`v2` 执行归档 `job3119858_20260716_072546` 再次从头完成了硬件门禁、源码恢复、编译和完整模型加载，服务于 `07:30:03` 就绪，但仍在发出任何 GSM8K 请求前被运行时身份门禁拦截。原因是容器内以 root 读取宿主用户拥有的只读 git 仓库时触发 Git `dubious ownership`；旧命令又把失败的 command substitution 放在成功返回的 `echo` 中，最终记录为空的 `DEEPGEMM_COMMIT=`。该 run 同样没有 warmup、timed、raw 或 score 文件，永久保留为计分前基础设施失败。
- 本文是 `v3`。它继承 `v2` 的 host PID namespace、PID 正向控制、容器 PID ownership、只读首次归档、主/清理错误码分离、artifact seal 和 `nvidia-smi nvlink -e`。唯一新增修复是：运行时 commit 查询显式使用一次性的 `git -c safe.directory=/workspace/DeepGEMM`，先把命令结果赋值并检查成功，再输出身份；不修改容器全局 git 配置，也不放宽 commit 精确匹配。
- 从 `v1` 到 `v3`，题集、题序、提示词、采样参数、两轮顺序、精度门槛和“不得按分数重跑”规则始终未改变。新的正式执行必须使用全新 run ID，从硬件/P2P、源码恢复、编译、完整模型加载、完整 warmup 到完整 timed 全流程重跑；不得复用前两个失败 run 的服务、JIT 文件或评分产物。节点操作系统页缓存可能自然缩短后续模型读取时间，但不属于服务或评分产物复用。

## 1. 冻结身份

- DeepGEMM：`75186dde9dac140c053c9007ace0ce7cce41150c`。
- SGLang：`b6a68c9acb6590b2849febe2b66807553923fc71`；固定源码 tar SHA256：`258ed9c1538dd534f6cbd0781b8c559ff99ea934b49b24940c96a279c8af1984`。
- 镜像：`lmsysorg/sglang@sha256:5027e95bf6ec536856b1b52a91d1f35ff5c564ab83e8a94758a169ff09bb8df3`。
- 环境边界：该镜像是 CUDA 13.0.1、PyTorch cu130，**不是 CUDA 13.2**。CUDA 13.2 用于同轮 MiMo kernel 性能实验；整模 E2E 使用历史验证过的 SGLang 固定镜像，报告不得混写。
- 模型：`DeepSeek-R1-0528-FP4-v2`，ModelOpt NVFP4/group-size 16、FP8 KV，H=7168、I=2048、256 experts、top-k=8、61 层，其中 58 个 MoE 层。必须存在连续 163 个 shard，总大小 `413328348544` bytes，metadata hash 必须匹配 runner 中的固定值。
- 数据：OpenAI `grade-school-math` commit `3101c7d5072418e28b9008a6636bde82a006892c` 的 `grade_school_math/data/test.jsonl`；SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。正式运行读取已封存的本地副本 `gsm8k_test_3101c7d.jsonl`，禁止下载浮动 `master` 后直接开测。

## 2. 服务参数

使用一台完整 NVSwitch 的 8×H200 节点。启动前要求 56/56 mapped P2P 通过、GPU 无其他计算进程。固定参数：

- `DG_SM90_NVFP4_BLOCK_N=256`
- `DG_SM90_NVFP4_NIBBLE_GROUP=1`
- `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096`
- 不设置 `SGLANG_MEGAMOE_NVFP4_REQUANTIZE` 和 `DG_SM90_NVFP4_FUSED_B_SCALE`
- `--quantization modelopt_fp4 --tp 8 --dp 8 --enable-dp-attention`
- `--moe-a2a-backend megamoe`
- `--disable-cuda-graph --disable-radix-cache --skip-server-warmup`
- `--max-running-requests 1316 --watchdog-timeout 900 --mem-fraction-static 0.9`
- 不手工传 `--chunked-prefill-size`

源码先在固定镜像中重建 `_C`。服务实际 import 必须指向 `/workspace/DeepGEMM` 和 `/workspace/sglang`。计分前必须从实际 `ServerArgs` 确认 `enable_eplb=False` 及上述核心参数；服务日志必须精确出现 `8 × 58 = 464` 条 `sm90_nvfp4` 权重构建记录，每个 `DP0..7 / TP0..7 / EP0..7` 对应 rank 都必须完整覆盖 layer 3..60，且全部为 `block_n=256, grouped_nibbles=True`。

计分前还必须通过以下证据门：

- 用 host PID namespace 的 `nvidia-smi --query-compute-apps` 得到非空记录，8 个预检 GPU UUID 全部出现；
- 至少 8 个有效 GPU PID，且每个 PID 都存在于本次 runner 创建并校验过 container ID 的服务容器 `docker top` 中；
- runner 在启动正式服务前创建一个短暂的单卡 GPU 正向测试进程，确认同一种查询确实能看见其 host PID，随后清理并等待 GPU 回到空闲；
- 实际 Docker inspect 中的固定镜像、命令和环境变量与本协议一致，两个未启用的实验变量仍不存在。

## 3. GSM8K 方法

同一个服务只跑两轮，次序固定且不允许挑选：

1. `warmup`：完整 1316 题，排除出延时结论，但仍接受精度和异常输出门禁；
2. `timed`：完整 1316 题，正式结果。

两轮都固定：8-shot、1316 题、并发 1316、temperature 0、top-p 1、max-new-tokens 512。不得因分数不理想重跑、替换或删除任一轮。若发生明确基础设施故障，本次 run 标记失败；后续如另开全新 run，必须保留失败产物并同时披露。

## 4. 通过标准

两轮均须满足：

- raw 恰好 1316 条，`prompt_id=0..1315`，每个 prompt 与冻结 dataset 按未修改 harness 生成的文本逐字一致；
- parser 从冻结 dataset 的答案和 raw output 独立重新抽取最后一个数字，不能只信 result JSON 的 accuracy 或 raw 中的 `correct` 字段；
- 至少 `1260/1316` 正确（精确比例 `0.957447`），即独立重算的真实 `accuracy >= 0.957`。`1259/1316=0.956687` 即使被受保护脚本显示为三位小数 `0.957` 也不算通过；
- `invalid=0`；已知竞态特征（替换字符、非法控制字符、长重复数字/词段/标点）为 0；
- 无 CUDA OOM、NCCL error、watchdog timeout、RuntimeError 或 traceback；
- 服务就绪后，8 个预检 H200 UUID 都必须有服务 compute process。result JSON 中 `num_gpus=1` 是客户端脚本硬编码，不能拿它判断服务只有一张卡。
- warmup 前的服务证据验证器必须通过；验证器自身的合成自测必须在昂贵的源码编译和模型加载前通过。

只要任一项失败，整轮状态为 FAIL，不允许用“平均正常”覆盖失败。

## 5. 解释边界和已知风险

- 这是完整 SGLang 服务、真实 413GB DeepSeek-R1 权重、真实文本请求和最终答案的 E2E；`tests/test_nvfp4_mega_moe_sm90_correctness.py` 的合成张量 cosine/gate 是另一类 kernel correctness 证据。
- 最终 `751+b6a` 组合过去没有做过这项整模 E2E。历史 `0.957 / invalid 0 / 117.242s` 来自祖先 `eb8bb43`，只能作锚点，不能预填成本轮结果。
- 固定 SGLang benchmark 会把 test split 的前 8 题同时作为 8-shot 示例，并仍在 1316 道计分题中包含它们。严格说有 8/1316 的内生泄漏。为了与历史团队协议逐点可比，本轮不修改 harness，但最终结论只能称“固定 E2E 回归代理”，不能称“无泄漏官方 GSM8K 分数”。
- 模型验证会固定 163 个 shard 的文件名、大小总和与 index/配置/tokenizer hash；不会额外顺序读取并 SHA256 413GB 权重，因此不构成逐 shard 内容校验。

## 6. 预计时长

源码恢复/编译约 10–20 分钟。首次执行实测模型从容器启动到 ready 约 40 分钟，因此新的保守预估为模型加载 30–45 分钟；两轮 GSM8K 约 5–15 分钟，收尾审计约 5 分钟。总计通常 50–75 分钟，NFS 或首次 JIT 较慢时可能更久。
