/**
 * Copyright 2026 The CANN Contest Authors. All Rights Reserved.
 *
 * BatchMatmulMaxSum TilingFunc 实现（Host 侧）
 *
 * 职责：
 *   1. 解析输入 shape / 转置属性 / 数据类型
 *   2. 选择 M/N/K 基本块（mTile, baseN, baseK）与 N 维并行段数 segS
 *   3. 用 MultiCoreMatmulTiling 生成 TCubeTiling（SetDim(1)：每个子单元
 *      单逻辑核完成一次 [mTile x baseN x K] 的小 GEMM，核间切分由 kernel
 *      侧按 (b, mt, s) round-robin 自主完成，规避多核 tiling 的分形顺序
 *      不确定性）
 *   4. 计算用户 workspace（跨核归约中转区）+ 系统 workspace，SetBlockDim
 */
#include "batch_matmul_max_sum_tiling.h"

#include "batchmatmulmaxsum.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <cstring>

namespace {
constexpr int64_t ALIGN_UP(int64_t x, int64_t a) { return (x + a - 1) / a * a; }

int32_t PickMTile(int64_t m)
{
    if (m >= 128) {
        return 128;
    }
    return static_cast<int32_t>(ALIGN_UP(m, 16));  // baseM 最小 16 且需 16 对齐
}

int32_t PickBaseN(int64_t n)
{
    if (n >= 64) {
        return 64;
    }
    if (n >= 32) {
        return 32;
    }
    return 16;
}

int32_t PickBaseK(int64_t k)
{
    if (k >= 256) {
        return 256;
    }
    if (k >= 128) {
        return 128;
    }
    if (k >= 64) {
        return 64;
    }
    return 32;  // 题目约束 K >= 32 且 K % 8 == 0
}
}  // namespace

namespace optiling {

ge::graphStatus BatchMatmulMaxSumTilingFunc(gert::TilingContext *context)
{
    /* ---------------- 1. 解析输入 ---------------- */
    const auto attrs = context->GetAttrs();
    const bool transX1 = *attrs->GetAttrPointer<bool>(0);
    const bool transX2 = *attrs->GetAttrPointer<bool>(1);

    const auto x1Shape = context->GetInputShape(0)->GetStorageShape();
    const auto x2Shape = context->GetInputShape(1)->GetStorageShape();
    /* 逻辑形状恒为 x1:(B,M,K)、x2:(B,K,N)；转置属性只影响存储布局：
     *   transposeX1=0: x1 存储 (B,M,K)；=1: x1 存储 (B,K,M)
     *   transposeX2=0: x2 存储 (B,K,N)；=1: x2 存储 (B,N,K) */
    const int64_t B = x1Shape.GetDim(0);
    const int64_t M = transX1 ? x1Shape.GetDim(2) : x1Shape.GetDim(1);
    const int64_t K = transX1 ? x1Shape.GetDim(1) : x1Shape.GetDim(2);
    const int64_t N = transX2 ? x2Shape.GetDim(1) : x2Shape.GetDim(2);

    const ge::DataType x1Dtype = context->GetInputDesc(0)->GetDataType();
    const bool isHalf = (x1Dtype == ge::DT_FLOAT16);

    /* ---------------- 2. 平台信息 ---------------- */
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const int32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    int32_t coreNum = aicNum;
    if (aivNum > 0 && (coreNum <= 0 || aivNum < coreNum)) {
        coreNum = aivNum;  // blockDim 同时不超过 AIC/AIV 资源
    }
    if (coreNum <= 0) {
        coreNum = 1;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const int64_t libWs = static_cast<int64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());

    /* ---------------- 3. 切分参数 ---------------- */
    int32_t mTile = PickMTile(M);
    int32_t baseN = PickBaseN(N);
    int32_t baseK = PickBaseK(K);
    const int32_t nSegs = static_cast<int32_t>((N + baseN - 1) / baseN);

    /* 多次尝试生成 TCubeTiling：失败则按 baseK -> mTile -> baseN 逐级降级重试 */
    AscendC::tiling::TCubeTiling cubeTiling;
    bool ok = false;
    for (int tryK = 0; tryK < 4 && !ok; ++tryK) {
        const int32_t tryBaseK = tryK == 0 ? baseK : (baseK >> tryK);
        if (tryK > 0 && tryBaseK < 32) {
            break;
        }
        for (int tryM = 0; tryM < 4 && !ok; ++tryM) {
            const int32_t tryMTile = tryM == 0 ? mTile : (mTile >> tryM);
            if (tryM > 0 && tryMTile < 16) {
                break;
            }
            for (int tryN = 0; tryN < 3 && !ok; ++tryN) {
                const int32_t tryBaseN = tryN == 0 ? baseN : (baseN >> tryN);
                if (tryN > 0 && tryBaseN < 16) {
                    break;
                }
                /* Vector 侧 UB 预算：C 双缓冲 + 行max缓冲 + Phase2 归约缓冲 */
                const int64_t usedUb = 2LL * tryMTile * tryBaseN * sizeof(float) +
                                       1LL * tryMTile * sizeof(float) +
                                       64LL * tryMTile * sizeof(float);
                matmul_tiling::MultiCoreMatmulTiling tilingApi(ascendcPlatform);
                tilingApi.SetDim(1);  // 单逻辑核；核间切分由 kernel 自主完成
                tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                   isHalf ? matmul_tiling::DataType::DT_FLOAT16 : matmul_tiling::DataType::DT_BF16,
                                   transX1);
                tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                                   isHalf ? matmul_tiling::DataType::DT_FLOAT16 : matmul_tiling::DataType::DT_BF16,
                                   transX2);
                tilingApi.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND,
                                   matmul_tiling::DataType::DT_FLOAT);
                tilingApi.SetOrgShape(M, N, K);
                tilingApi.SetShape(tryMTile, tryBaseN, K);
                tilingApi.SetFixSplit(tryMTile, tryBaseN, tryBaseK);
                tilingApi.SetBufferSpace(-1, -1,
                                         static_cast<int64_t>(ubSize) - usedUb);
                if (tilingApi.GetTiling(cubeTiling) != -1) {
                    mTile = tryMTile;
                    baseN = tryBaseN;
                    baseK = tryBaseK;
                    ok = true;
                }
            }
        }
    }
    if (!ok) {
        return ge::GRAPH_FAILED;
    }

    /* ---------------- 4. 并行度与子单元划分 ---------------- */
    const int32_t mtpb = static_cast<int32_t>((M + mTile - 1) / mTile);
    const int32_t totalGroups = static_cast<int32_t>(B) * mtpb;
    /* N 维并行段数：仅当 (b,mt) 组数不足以填满核时启用，避免额外归约开销 */
    int32_t segS = 1;
    if (totalGroups < coreNum) {
        const int32_t need = (coreNum + totalGroups - 1) / totalGroups;
        segS = need < nSegs ? need : nSegs;
    }
    const int32_t totalUnits = totalGroups * segS;
    const int32_t blockDim = totalUnits < coreNum ? totalUnits : coreNum;

    /* ---------------- 5. workspace ---------------- */
    /* 用户区：跨核归约中转 [B][mtpb][segS][mTile] fp32（行 max 结果） */
    const int64_t wsUser = static_cast<int64_t>(totalUnits) * mTile * static_cast<int64_t>(sizeof(float));
    const int64_t wsTotal = wsUser + libWs;
    size_t *currentWs = context->GetWorkspaceSizes(1);
    currentWs[0] = static_cast<size_t>(wsTotal);

    /* ---------------- 6. 序列化 tiling ---------------- */
    static_assert(sizeof(AscendC::tiling::TCubeTiling) <= BMMS_CUBE_TILING_RAW_LEN * sizeof(int32_t),
                  "TCubeTiling exceeds reserved raw buffer, enlarge BMMS_CUBE_TILING_RAW_LEN");
    auto *tiling = context->GetTilingData<BmmsTilingData>();
    (void)memset_s(tiling->cubeRaw, sizeof(tiling->cubeRaw), 0, sizeof(tiling->cubeRaw));
    (void)memcpy_s(tiling->cubeRaw, sizeof(tiling->cubeRaw), &cubeTiling, sizeof(cubeTiling));

    tiling->B = static_cast<int32_t>(B);
    tiling->M = static_cast<int32_t>(M);
    tiling->N = static_cast<int32_t>(N);
    tiling->K = static_cast<int32_t>(K);
    tiling->transX1 = transX1 ? 1 : 0;
    tiling->transX2 = transX2 ? 1 : 0;
    tiling->isHalf = isHalf ? 1 : 0;
    tiling->mTile = mTile;
    tiling->mTilesPerBatch = mtpb;
    tiling->baseN = baseN;
    tiling->nSegs = nSegs;
    tiling->segS = segS;
    tiling->totalUnits = totalUnits;
    tiling->blockDim = blockDim;
    tiling->wsElemCnt = totalUnits * mTile;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
IMPL_OP(BatchMatmulMaxSum).Tiling(optiling::BatchMatmulMaxSumTilingFunc);
}  // namespace ge
