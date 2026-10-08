# BatchMatmulMaxSum — CANN 上合赛区初赛算子赛题

Ascend C 实现的 ColBERT Late Interaction 融合算子：

```
y[b] = Σ_m  max_n ( Σ_k  X1[b, m, k] · X2[b, k, n] )
```

- 输入：`x1`（逻辑 `(B, M, K)`）、`x2`（逻辑 `(B, K, N)`），FP16/BF16（二者同类型）
- 输出：`y`（`(B,)`，FP32）
- 属性：`transposeX1` / `transposeX2` 仅声明存储布局（`transposeX1=1` 时 x1 存储 `(B,K,M)`；`transposeX2=1` 时 x2 存储 `(B,N,K)`），不参与数学计算
- 规模约束：`B∈[1,64]`，`M,N∈[1,8192]`，`K∈[32,8192]` 且 `K%8==0`，`B·M·K ≤ 2^26`，`B·N·K ≤ 2^26`
- 精度：K 维点积、Max、Sum 全程 FP32；相对/绝对误差 < 1e-4；同一测试点多次执行结果必须一致

## 文件结构

```
BatchMatmulMaxSum/
├── op_host/
│   ├── batchmatmulmaxsum.h               # 算子原型注册（REG_OP / 属性 / InferShape 由框架处理）
│   ├── batch_matmul_max_sum_tiling.h     # TilingData POD 结构（host/kernel 双端共享）
│   └── batch_matmul_max_sum_tiling.cpp   # TilingFunc：切分策略 + TCubeTiling 生成
└── op_kernel/
    └── batch_matmul_max_sum.cpp          # Ascend C 核函数（Matmul 高阶 API + Vector 归约融合）
```

## 算法架构

单核函数两阶段设计（MIX 形态：AIV 发起 Matmul，AIC 协同计算，C 经 VECIN 回 UB 归约）：

**Phase 1 — Cube 计算 + 行内 max**
- 子单元 = `(b, mt, s)`（batch × M 块 × N 段），round-robin 分发到各核
- 每个子单元一次 `[mTile × baseN × K]` 小 GEMM：`SetDim(1) + SetFixSplit`，单次 `Iterate<true>` 产出完整 C 块，块位置完全可预测
- M/N 尾块用 `SetTail` 处理；N 对齐段走向量树归约（每行 baseN→8 候选，纯向量 Max），N 尾段走标量归约（`-inf` 初始化，保证全负相似度正确）
- 每行 max 结果写入 workspace 中转区 `ws[u·mTile + m]`

**SyncAll 核间硬同步**（`SyncAll<false>`，blockDim ≤ 物理核数）

**Phase 2 — 跨段合并 + 确定性求和**
- batch `b` 由核 `b % blockNum` 负责：读回 `[mtpb][segS][mTile]`，跨 N 段向量 Max 合并 → 每行 max → 固定顺序（mt 升序、行升序）标量累加 → `y[b]`
- 无任何原子操作 / 乱序累加，多次执行位级一致

## 关键设计点

| 难点 | 方案 |
|---|---|
| 四种存储布局组合 | `SetTensorA/B` 子块指针偏移 + `SetAType/BType(..., isTrans)` 声明转置，步长由 `SetOrgShape(M,N,K)` 统一决定 |
| N 维任意值（含 N=1）尾块 | 主体段（baseN 对齐）向量树归约；尾段 `SetTail(m,n)` + `GetTensorC(enSeq=true)` 紧凑布局标量归约 |
| Cube 结果向 Vector 传递 | Matmul C 类型设为 VECIN，`TQue<VECIN,2>` 双缓冲 + EnQue/DeQue 建立依赖 |
| 小 batch 多核不均衡 | `(b,mt,s)` 子单元 round-robin；组数不足时自动启用 N 维分段并行（segS>1） |
| 全负相似度 | 行 max 初值 `-inf`（严禁 0 初始化） |
| 数值一致性 | Cube FP32 累加 + 全程定序归约，无原子操作 |
| TCubeTiling 跨端传递 | 纯 POD `int32` 块序列化（`cubeRaw[32]`），host `memcpy` 进 / kernel 循环还原，不依赖 tiling 宏 |

## 性能思路

相对基线（bmm + amax + sum 三算子拆分执行）：
- 消除中间 `C (B,M,N)` 的 GM 写读往返（基线需 ≥2×B·M·N·4B 额外流量），融合后 C 全程驻留片上
- 单次 kernel launch，消除算子间同步开销
- 子单元粒度 (128×64) 保证 Cube 利用率；K 由 Matmul API 内部分块流水

## 判题适配说明

- 判题环境：CANN 9.0.0，算子核函数工程(beta)，Atlas A2 (910B)
- 若判题模板对文件名有要求：`op_host` 三个文件与 `op_kernel` 一个文件按模板目录对应放置即可；kernel 通过 `#include "batch_matmul_max_sum_tiling.h"` 引用 host 侧结构（标准工程默认包含 op_host 头文件路径；若无，可将该头文件复制到 op_kernel 目录）
- 属性名按题目接口使用 `transposeX1` / `transposeX2`（驼峰）
- 若 CANN 版本 `SyncAll<false>()` 不可用，可改为 `AscendC::SyncAll()`
- `TCubeTiling` 大小变化超过 128B 时，增大 `batch_matmul_max_sum_tiling.h` 中的 `BMMS_CUBE_TILING_RAW_LEN`（有 static_assert 保护）
