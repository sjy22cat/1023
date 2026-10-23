请为当前 SIM-1023 补齐推理评估需要的缺失硬件数据。请执行环境检查、缺口审计、必要综合/仿真/功耗分析和结果校验。已有数据保持不动，只测缺少的项目。

## 入口和已有数据

实际 Python 根目录为 simulator/SIM-1023/simulator/simulator/SIM-1017；与之随包交付的 RTL 在 simulator/SIM-1023/RTL/RTL/RTL_1017SIM。不要因为目录名仍是 SIM-1017 而误用仓库外层旧模拟器。

先阅读 simulator/SIM-1023/README_attention_16x4_20261023.md，以及 Attention_confusion/parallel_row/synopsys/ 下的 existing_data_inventory.json、measurement_requests.json、PROMPT_1_attention.md。

注意随包系统 RTL 尚有接口差异：nonmac_sources.f 引用不存在的 llm_nonmac_fp32.sv，实际文件名是 llm_nonmac_fp32_1017修改版2.sv；MC 文件是 mx_mc_1017修改版.v；yrr_1017_top 默认只有32个MC，新非MAC的cmd_vector/向量端口在旧wrapper中未连接。先做源码/端口审计，单独建立明确文件列表，不得误用旧RTL或仅改文件名就认为已实现新版系统。缺少逐PE MC/向量集成时标记NOT_IMPLEMENTED；不要用原32-MC顶层表征32768-MC新系统。本任务允许新增测量脚本/wrapper，但改变设计逻辑另列明确建议，不擅自修改原RTL。

所有输入 calibration JSON、components_20261019*.json、nonmac_parallel_1024.json 都视为只读。先核对 SHA256；复用其中已有非空值、原单位、来源与限定条件。原报告中数值审核未通过的项目保留原始值和状态，不能静默提升为通过。遇到冲突列出冲突而不覆盖；仅测仍为空、没有对应测量或缺少本工作负载 profile 的项目。不要把 metadata 的 null 当作需要跑综合的硬件项目。

采用 SMIC 28nm SCC28NHKCP_HSC35P140_RVT_V0p2、RVT/SS/0.9V/125°C，先核对单位。约束和功能/映射/黑盒/活动质量要求沿用 PROMPT_1_attention.md。不得采用测试 DW 替代模型。功耗应来自正确非零功能活动和完整映射网表。不要用“面积×经验系数”填能量，也不要把软件执行时间当硬件延迟。

## A. 已新增、必须单独表征的融合 Attention

执行 PROMPT_1_attention.md 的单 head 默认 16×4 综合，得到独立 fused_attention_16x4.measured.json。每个 query/head 内含 QK→online softmax→PV，不再另外加旧 QK/PV block 或 standalone Softmax 能量。

当前模拟器支持 --attention-impl fused_16x4 和 --attention-calibration PATH：64 逻辑 tile/head，32×32 阵列最多 16 head；计算延迟按 ceil(Q_heads/16) 波累计，动态能量按实际 head 数累计。prefill 使用因果上下文，query 串行。每个 query/head 读 Q 一次、读 N 行 K/V 各一次；不假设 GQA multicast、跨 query KV 复用或计算/搬运重叠。N=1..1151 覆盖 1024→128 与 128→1024；1152 另作基线。不要测错成 128 次 decode 都包含新生成的最后一个 token，实际 decode forward 是 output_tokens-1。

profile 周期包括 ceil(N/32) 个理想 KV 装载握手。模拟器扣除这些握手，再以 max(理想握手时间,现有外存传输时间) 计加载，避免重复累计；局部寄存器写入动态能量仍在 profile 中。外部 RRAM/FeRAM/PHY/GSRAM 保持既有计费，不能包含在 single-head dynamic_pj 中再次累加。

## B. 原近存计算和非 MAC：仅补缺项

1. 根据 yrr_v3/rtl_dataset.py 及现有 JSON 契约，核对 warm 1024×1024 GEMV 的 block_cycles/block_pj 是否已有可用结果。若没有才测权重已装载时的首结果延迟、连续命令 II、活动动态能量、完成/可接收下一命令的协议；与外部权重装载、Host/DMA、Softmax 分开。已有 PE_ALL、bf16_align_add32、mx_scale32、MC、PHY、tile_nonpe 等单元数据按原 scope 复用，不能未经定义就把每 PE/32路能量当作每 MAC。
2. 增量测跨 K 块部分和累加、全阵列归约和格式转换缺失项。若 warm block 已包含某一级归约/转换，须建立 scope 表排除重复计费。1024² GEMV 的 block 不是 batch 矩阵 GEMM，不能偷换能量分母。
3. 当前 Qwen3-0.6B 融合模式仍需独立的 RMSNorm(length128、1024)、RoPE_vector(128)、residual(1024)、SwiGLU_vector(1024) profile。MLP 3072 的 pointwise 运算按 3×1024 命令；归约类不能独立切块改变数学结果。仅测所用 op/length 中缺失的组合，已有完整 profile 不重测。每条记录 cycles、dynamic_pj、read_count、write_count、report，并记录 engine+scratch、排除 staging/leakage 的 scope。
   当前精确命令契约校验scratch访问次数；新空间并行RTL的cmd_vector路径可能采用不同流量语义，旧wrapper又未连接该接口。必须确认采用的实际命令入口和scratch适配是否与契约一致；不一致时输出真实新结果并列出需要的适配工作，不要篡改read/write计数以通过旧加载器。
4. 若评估固定格式归约→FP32 scratch 的 staging 路径，则独立表征尚缺的 reduce_staging length；主推理当前没有消费的 staging 项标注“其他入口需要”，不要重复加入 Attention 的局部缓冲开销。
5. nonmac_parallel_1024.json 是 1024 复制单元的 USER_SUPPLIED 估算接口，不是已集成非 MAC 引擎实测。不能拿标量 exp/mul 等的延迟当完整 RMSNorm/Softmax latency；共享 max/sum 树和局部存储要计入。主 run_llm 的 measured 入口使用 nonmac 精确命令表，避免把两种 schema 混写。

## C. 存储与互连：只补缺少的接口/宏/IP数据

保留已有 SRAM/MC/PHY/PE/格式单元的记录，缺失时才补以下 scope：

- RRAM、FeRAM、GSRAM 分别的有效容量、访问粒度、read/write 动态能量、request latency、连续访问 II、可用带宽、bank/channel 数、并发冲突、固定开销与传输时间的拆分；区分 payload 与 E8M0 scale。使用真实 macro/IP .lib/.db、编译器或器件/IP 表征，标准单元综合的寄存器堆、SRAM 行为模型、mx_memdie_channel_model 不能冒充真实 RRAM/FeRAM。
- MC=32768（一 PE 一个），mc_id=tile_id*pe_per_tile+pe_id；外部 channel=32，二者不可混同。read/skip/metadata read 的能量与控制器接受间隔、响应延迟分开。已有 read/skip 数据复用。
- 输入广播和跨 tile 归约要区分首次数据 latency 和 II。现有 0.4mm tile pitch/12.8mm 阵列以及 BookSim/线模型只作为原估算保留；没有物理布局就不能升级成实际互连面积/能耗。不要重测已有原语；如需要整个拓扑的新指标，列为“组合系统新增 scope”。
- PHY 现有能量若按 128bit 给定，保持这一分母，映射成 pJ/bit 时只做明确单位换算。不要再除以/乘以 32 channel。Host/DMA 不在本次 accelerator inference 范围。

## D. 16×4/head 与 MXFP4 硬件之间的新缺口

当前 parallel_row 是 FP32 评估器，并未实例化物理 mx_tile_1017，因此仅综合它不能得到真实多列 MXFP4 阵列融合后的全部指标。

请先检查上传 RTL 中是否已有以下实现；没有就标记 NOT_IMPLEMENTED，列出需要实现的接口，不允许用 behavioral model/经验倍数代替实现，亦不要擅自改动原算术 RTL：

1. MXFP4 Q/K/V 解码为 FP32、FP32 O 输出 MXFP4 编码，lane 数/吞吐/缓冲/面积/动态能量/latency/II；已有的 MXFP4→tagFP4 转换不是 MXFP4→FP32。
2. 16 个 head 单元的数据分发与共享端口仲裁、context tile 装载安排，实际可同时供给多少 head。单 head top 只有每拍一个 32×128 K/V tile 的入口；不能假定整个 core 同时供给 16 head 而没有带宽/调度依据。
3. 如果支持 GQA 两个 Q head 共享一个 KV head 的读取，应有 multicast/缓冲实现和新增开销的独立测量；没有则每 Q head 独立读 KV，不能直接把读量减半。
4. 集成后的面积和漏电差额，明确原 MAC 阵列是复用、替换还是新增。FP32 evaluator 的 MAC/寄存器已在其面积/能量里，禁止在原 MXFP4 MAC 面积上直接再加 16 倍 evaluator 后又声明“无额外 MAC”。

这些是当前 simulator 的显式 pending 项。即使 A 的单 head 数据齐全，也只能给“测量原语驱动的架构预测”，不能发布已验证的全系统 tok/s。SIM-1023 的矩阵/PPL 入口有逐 PE 读 mask 模型，但 run_llm 仍使用串行聚合存储账本；单靠补全价格不会自动启用逐 MC/32-channel 或 batch 调度。需要这种调度时另列实现工作，不能靠除以32来补齐。

## 交付与缺口闭环

在新 results/system_missing_<timestamp>/ 中写：

- missing_data_plan.json：字段/算子长度、原来源、为何缺失、采用复用或新测、单位、scope、目标报告。
- new_measurements.json：只含新增实测结果、报告路径、SHA256、工具/DW/库/PVT/时钟/活动窗口/质量标记。无法测量项保持 null 和原因，区分缺库、缺 RTL、时序未过、未完成运行。
- field_source_mapping.json：旧数据与新数据怎样映射到模拟器契约，每个字段注明能量分母与排重规则。
- rtl_1017sim_completed.json：仅当可以从已有数据与新增数据合法补齐时，另建完整契约副本，原输入 JSON 不动。已有非空值严格保留，冲突/不等价 scope 不得用转换凑齐。非硬件元数据如 status 只在满足所有验证后可在新副本中改为 MEASURED。保留全部已有原始来源。
- 独立 fused_attention_16x4.measured.json、Tcl/SDC、日志、网表、面积/STA/功耗原报告和未完成项表。

从上传包根目录验证新单 head 契约、原全量契约，再在实际 Python 根目录运行：

```
python tests/run_current.py
python run_llm.py --forward shape --kv-shadow none --prompt-tokens 1024 --max-new-tokens 128 --parameters <新完整契约副本> --attention-impl fused_16x4 --attention-calibration <独立Attention实测JSON> --json results_1024_128.json
python run_llm.py --forward shape --kv-shadow none --prompt-tokens 128 --max-new-tokens 1024 --parameters <新完整契约副本> --attention-impl fused_16x4 --attention-calibration <独立Attention实测JSON> --json results_128_1024.json
```

硬件共用时钟必须一致，不能把未闭合 1ns 的 evaluator 的 183 拍硬算成 183ns。单独闭合的慢时钟若与已有计算阵列不同，应报告域转换和调度缺口，不能修改旧价格掩盖。

若严格加载因缺项拒绝运行，列清缺项并保留新增实测，不能把 status 提前改成 MEASURED、注入 synthetic fixture、置零填缺口或删掉验证。始终区分：已有数据复用、单模块新增实测、系统架构预测、仍未实现/未测量的部分。最后重新校验原始输入文件 SHA256 均未改变。
