请在当前上传目录中，用 Synopsys 对 16×4/head 融合 Attention 做完整综合、功能、时序和活动功耗评估。请执行任务，保存可复现脚本和原始报告。

先阅读 Attention_confusion/parallel_row/README.md、该目录下 3 个 RTL 和本 synopsys 目录中的 measurement_requests.json。scripts/sim_operation1.txt 只用于参考既有约束，其旧顶层和旧参数不适用于本任务。

已有模拟器/器件数据一律保留，不重新测量覆盖。existing_data_inventory.json 是原始数据与 SHA256 的清单。本任务仅新增融合 Attention 测量结果，写到新的 run 目录，不覆盖任何输入文件。

## 1. 对象与环境

顶层 attention_head_16ctx_4dim_eval，默认参数 HEAD_DIM=128、TILE_SIZE=32、CONTEXT_PARALLEL=16、MAX_CONTEXT=1152，其余参数保持 RTL 默认值。以 Attention_confusion/parallel_row/sources.f 为 RTL 文件列表，从上传包根目录执行。

一次命令 = 一个 Q[1×128] 对 N 个 K/V token 得到 O[1×128]。16 个 context lane 同时计算、每个 lane 四个维度 block 并行，最大每波 512 token。这是单 head FP32 架构评估器，不是完整 16-head 阵列，也没有实例化 MXFP4 mx_tile_1017。禁止把其面积直接当作原近存 PE 阵列中 64 个 tile 的面积。

检查 dc_shell、VCS、PrimeTime/PrimePower（或等价 PT PX）、SMIC 库及 DesignWare 的版本、许可和实际路径。优先使用 SCC28NHKCP_HSC35P140_RVT_V0p2，RVT/SS/0.9V/125°C。使用安装环境官方 DW_fp_mult、DW_fp_addsub、DW_fp_exp、DW_fp_div 等综合及仿真实现。禁止使用 NONMAC_TEST_DW 或 nonmac_test_dw.sv。所有浮点实例必须核对 FP32 参数，exp 必须为 sig_width=23、exp_width=8、ieee_compliance=0、arch=2；不要依赖默认参数。

## 2. 官方模型先验功能检查

测试台：Attention_confusion/parallel_row/tb_attention_head_16ctx_4dim.sv。
先编译数学参考包 RTL/RTL_1017SIM/tb/nonmac_tb_math.sv，再加入官方 DW 模型、3 个 RTL、测试台；math 包不是 DW 替代模型。给出实际 VCS 命令。

保持默认硬件参数编译一次，以运行时参数选择输入长度：

```
./simv +CONTEXT_N=1152
./simv +CONTEXT_N=1152 +POWER_ACTIVITY +SEED=20261023 +CLOCK_NS=1.0 +DUMP_VCD=attention_1152.vcd
./simv +CONTEXT_FIRST=1 +CONTEXT_LAST=1152 +POWER_ACTIVITY +SEED=20261023
```

第三条可以分段运行，减少内存和活动文件规模。CLOCK_NS 应使用本次检查点的真实时钟。不要为每个 N 重新 elaboration，也不要缩小 HEAD_DIM/TILE_SIZE/CONTEXT_PARALLEL 代替默认硬件。小参数仿真仅可作为额外诊断。

检查非零输入数值与独立实数参考，至少覆盖 1、31、32、33、511、512、513、1023、1024、1025、1151、1152，包括尾部屏蔽、后波最大值、状态重用、复位、输入停顿、结果背压及非法命令。默认数值容差按测试台；报告最大绝对/相对误差、NaN/Inf 和所有位宽警告，不能只测周期。

无停顿时结构周期预期 C(N)=48×ceil(N/512)+ceil(N/32)+3，N=1152 应为 183。实测 o_cycle_count、握手到首次 o_valid 的周期必须一致。若不一致先诊断，不能把公式强行写成实测值。183ns 在 1ns 下只是名义延迟，功能仿真不能证明 STA 通过。

## 3. 综合与 STA

以 scripts/sim_operation1.txt 的原始接口条件为基线：1.000ns 理想时钟；setup/hold uncertainty=0.050/0.020ns；clock transition=0.050ns；数据输入 max/min delay=0.100/0ns，排除 clk/rst_n；输入 transition=0.050ns；输出 max/min delay=0.100/0ns；每输出负载 5fF；max transition=0.200ns；仅 rst_n false path，假设外部同步解除复位。先核对工具库单位，不能把 5fF 错写成 5pF。无 P&R/CTS/SPEF 时明确标为布局前结果。

保存 elaboration/link/check_design/check_timing、未约束路径、latch、黑盒和 unresolved reference 检查。检查整个默认主设计是否实际映射完成，不能用几个算子的预检查代替。报告总面积、组合/时序面积、总单元/时序单元/寄存器位数、层次面积和各类 FP 算子映射统计，区分标准单元面积与宏面积。

报告 setup/hold WNS、TNS、违例端点数、前 20 条路径，保留逐级单元和完整起终点。分析当前网表真实长路径，不复用旧 RTL 的最差路径作为结论。不得用 FSM 状态假路径、multicycle、接口延迟调整来隐藏违例。

若 1ns 不通过，继续保存该点结果，再单独扫描同一网表的 setup 可行周期，并在每个检查点独立检查 hold。原始 input min delay=0ps 的结果必须保留；10ps 只可另作敏感性检查，不得替代基线。不能把换频率/放宽接口后的结果写成 1GHz 达标。若共同系统时钟不是 1ns，也要明确跨域/重新综合问题。

默认配置规模很大。可使用分层综合和层次保留降低峰值内存，但需完成全顶层链接、真实映射和跨层 STA；分模块面积相加只能标为估算，不能伪装完整主设计结果。OOM/许可/超时均保存失败原因，不得改小参数、裁剪 lane 或留下 DW 黑盒后宣布完成。

## 4. 活动功耗与每命令动态能量

先完成默认配置的官方模型数值通过，再做该主设计映射网表上的活动功耗。+POWER_ACTIVITY 使用非零 Q/K≈[-0.5,0.5]、V≈[-1,1]，关闭人为抬高最后 token 的压力输入；无效总线回到确定值。使用 SEED=20261023，另用 20261024/20261025 做代表点的活动敏感性检查，记录活动分布和种子。

测试台打印 MEASURE_BEGIN/MEASURE_END，窗口为命令握手到首次 o_valid；排除复位、被中止命令、输出背压和空闲等待。每命令单独记录 begin_ns/end_ns/cycles/kv_load_cycles。使用 VCD/VPD/SAIF 时明确 tb…dut 到映射顶层的作用域，报告主输入、时序单元和组合内部活动覆盖率，以及传播活动、默认 toggle、X 比例和毛刺处理。测量窗口须与能量窗口完全一致。新综合生成寄存器名称映射时保留 name mapping，不能仅凭 SAIF 文件存在宣布功耗完成。

需求是 N=1..1151 的精确 profile（两种用户工作负载都是这些上下文），外加 N=1152 基线。可以分段批处理。每个 N 必须使用同一硬件网表，保存直接活动与报告；不要用边界 N 的能耗线性插值填中间值。如果无法完成全部 sweep，先输出实测子集并列出缺失 N，状态保留 PENDING。固定 seed 下可分段重跑；测试台每个 N 用 SEED+N 初始化。

分别输出 internal/switching/dynamic/leakage/total 功耗和功耗统计窗口。写入 profile 的 dynamic_pj 仅为 (P_internal+P_switching)×命令窗口；包含本 evaluator 内部 MAC/Softmax/局部 KV 寄存器，排除外部存储和漏电。单位：1mW×1ns=1pJ。总能量和漏电能量另存报告，不能把总能量交给另计漏电的模拟器导致重复计费。时序未闭合点只输出标明条件的诊断功耗，不能当作该频率下有效工作能量。

## 5. 交付

在新目录 results/attention_16x4_<timestamp>/ 保存 Tcl/SDC/文件列表/命令/日志/网表/DDC/SDF（如生成）/时序面积报告/活动及功耗报告/机器可读 summary.json。记录三个 RTL 的 SHA256、工具和 DW 版本、实际 PVT、库路径和约束。

以 Attention_confusion/parallel_row/synopsys/fused_attention_16x4.template.json 为契约，另写 fused_attention_16x4.measured.json；不要修改模板或已有 calibration。各 N 写 cycles、kv_load_cycles、dynamic_pj、report，填写 area_um2、sequential_area_um2、register_bits、leakage_mw、provenance 和 validation。只保留已测 N，不用 null profile 冒充测量。只有所需范围已完整、官方数值通过、全顶层映射无黑盒且该 clock 的 setup/hold 均通过时才置 MEASURED；其余保留 PENDING 并报告缺口。

最后给出面积、时序、周期、实际有效延迟、动态/漏电功耗与能量、测量范围和未完成项，明确“单 head 实测”与“16 head 并行的架构外推”的边界。不自行优化/修改待测 RTL；若发现功能错误，保存诊断并提出具体修正。
