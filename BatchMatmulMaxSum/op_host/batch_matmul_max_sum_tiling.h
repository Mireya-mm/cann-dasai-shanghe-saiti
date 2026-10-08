/**
 * Copyright 2026 The CANN Contest Authors. All Rights Reserved.
 *
 * BatchMatmulMaxSum 赛题（上合赛区初赛）Tiling 数据结构
 *
 * y[b] = sum_m max_n( sum_k X1[b,m,k] * X2[b,k,n] )   (ColBERT Late Interaction 融合算子)
 *
 * 说明：采用"标准 C++ 语法定义 TilingData 结构"（CANN 官方支持的方式），
 *       host 侧与 kernel 侧共享本头文件，不依赖 BEGIN_TILING_DATA_DEF 宏，
 *       对各类工程模板（含 cannjudge 算子核函数工程）兼容性最好。
 *
 * TCubeTiling 通过 raw 字节块传递（host 用 MultiCoreMatmulTiling 生成后
 * memcpy 进来，kernel 侧再还原），避免双端直接依赖 CANN tiling 宏。
 */
#ifndef BATCH_MATMUL_MAX_SUM_TILING_H
#define BATCH_MATMUL_MAX_SUM_TILING_H

#include <cstdint>

/* TCubeTiling 原始字节块上限（CANN 8.x/9.x 的 TCubeTiling 均小于 128B，
 * host 侧有 static_assert 保护，若 CANN 版本变化超限请增大该数组） */
constexpr int32_t BMMS_CUBE_TILING_RAW_LEN = 32;  // 32 * int32 = 128B

struct BmmsTilingData {
    /* ---------- Matmul TCubeTiling 原始字节 ---------- */
    int32_t cubeRaw[BMMS_CUBE_TILING_RAW_LEN];
    /* ---------- 问题规模 ---------- */
    int32_t B;        // batch 数
    int32_t M;        // X1 的 M 维
    int32_t N;        // X2 的 N 维
    int32_t K;        // 规约维（K % 8 == 0）
    int32_t transX1;  // x1 存储布局：1 -> (B,K,M)（转置），0 -> (B,M,K)
    int32_t transX2;  // x2 存储布局：1 -> (B,N,K)（转置），0 -> (B,K,N)
    int32_t isHalf;   // 1 -> fp16, 0 -> bf16
    /* ---------- 切分策略 ---------- */
    int32_t mTile;          // M 维基本块（16 对齐，<=128）
    int32_t mTilesPerBatch; // ceil(M / mTile)
    int32_t baseN;          // N 维基本块（16/32/64）
    int32_t nSegs;          // N 方向总段数 ceil(N / baseN)
    int32_t segS;           // N 方向实际并行的段数（1 ~ nSegs，仅小规模时 >1）
    int32_t totalUnits;     // 总子单元数 = B * mTilesPerBatch * segS
    int32_t blockDim;       // kernel 启动核数
    /* ---------- workspace 用户区布局（元素个数，fp32） ----------
     * wsRowMax[((b * mTilesPerBatch + mt) * segS + s) * mTile + m]
     * 总大小 = totalUnits * mTile 个 fp32
     */
    int32_t wsElemCnt;  // totalUnits * mTile
};

#endif  // BATCH_MATMUL_MAX_SUM_TILING_H
