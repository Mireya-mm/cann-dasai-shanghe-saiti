/**
 * Copyright 2026 The CANN Contest Authors. All Rights Reserved.
 *
 * BatchMatmulMaxSum 核函数实现（Ascend C，Atlas A2 / 910B）
 *
 * 计算语义：y[b] = sum_m max_n( sum_k X1[b,m,k] * X2[b,k,n] )
 *
 * 总体架构（单核函数，MIX 形态：Matmul 高阶 API + Vector 归约融合）：
 *
 *  Phase 1（Cube 为主）：
 *    子单元 = (batch b, M 块 mt, N 段 s)，round-robin 分发到各核；
 *    每个子单元一次 [mLen x nLen x K] 的小 GEMM（SetDim(1) + SetFixSplit，
 *    单次 Iterate 即产出一个完整 C 块，块位置完全可预测），
 *    C 块经 VECIN 队列进入 UB，Vector 侧做行内 max 归约，
 *    每行 max 结果写入 workspace 中转区：
 *        ws[u * mTile + m]  (u = (b*mtpb + mt)*segS + s)
 *
 *  SyncAll 核间硬同步
 *
 *  Phase 2（Vector）：
 *    batch b 由核 (b % blockNum) 负责：读回该 batch 的 [mtpb][segS][mTile]
 *    中转区，跨 N 段向量 max 合并 -> 每行 max -> 按固定顺序标量求和 -> y[b]
 *
 * 精度与一致性要点：
 *  - 点积在 Cube 内 FP32 累加，C 全程 FP32，无 fp16 中间量化；
 *  - 全负相似度场景安全：N 尾块行 max 以 -inf 初始化；
 *  - 所有归约顺序固定（无原子操作、无乱序累加），多次执行结果位级一致。
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "batch_matmul_max_sum_tiling.h"

namespace {
constexpr int32_t ALIGN_UP_I32(int32_t x, int32_t a) { return (x + a - 1) / a * a; }

__aicore__ inline int32_t IMin(int32_t a, int32_t b) { return a < b ? a : b; }

__aicore__ inline float ScalarMax(float a, float b) { return a > b ? a : b; }
}  // namespace

template <typename T>
class KernelBmms {
public:
    __aicore__ inline KernelBmms() = default;

    __aicore__ inline void Init(BmmsTilingData &td, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace)
    {
        td_ = td;
        /* 还原 TCubeTiling（按 int32 块拷贝，不依赖 libc memcpy） */
        {
            int32_t *raw = reinterpret_cast<int32_t *>(&tc_);
            constexpr int32_t cubeInts =
                static_cast<int32_t>(sizeof(AscendC::tiling::TCubeTiling) / sizeof(int32_t));
            static_assert(cubeInts <= BMMS_CUBE_TILING_RAW_LEN,
                          "TCubeTiling exceeds reserved raw buffer");
            for (int32_t i = 0; i < cubeInts; ++i) {
                raw[i] = td.cubeRaw[i];
            }
        }

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1),
                              static_cast<uint64_t>(td_.B) * td_.M * td_.K);
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2),
                              static_cast<uint64_t>(td_.B) * td_.K * td_.N);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), static_cast<uint64_t>(td_.B));
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace),
                              static_cast<uint64_t>(td_.wsElemCnt));

        REGIST_MATMUL_OBJ(&pipe_, GetSysWorkSpacePtr(), mmObj_, &tc_);

        pipe_.InitBuffer(cQue_, 2, static_cast<uint32_t>(td_.mTile) * td_.baseN * sizeof(float));
        pipe_.InitBuffer(rowMaxBuf_, static_cast<uint32_t>(td_.mTile) * sizeof(float));
        pipe_.InitBuffer(ph2Buf_,
                         static_cast<uint32_t>(td_.segS) * td_.mTile * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        Phase1Gemm();
        AscendC::SyncAll<false>();  // 核间硬同步（blockDim <= 物理核数）
        Phase2Reduce();
    }

private:
    /* ---------------- Phase 1：GEMM + 行内 max 归约，写 workspace ---------------- */
    __aicore__ inline void Phase1Gemm()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        for (int32_t u = static_cast<int32_t>(blockIdx); u < td_.totalUnits;
             u += static_cast<int32_t>(blockNum)) {
            const int32_t mt = u % td_.mTilesPerBatch;
            const int32_t grp = u / td_.mTilesPerBatch;
            const int32_t s = grp % td_.segS;
            const int32_t b = grp / td_.segS;

            const int32_t mStart = mt * td_.mTile;
            const int32_t mLen = IMin(td_.mTile, td_.M - mStart);
            const int32_t nStart = s * td_.baseN;
            const int32_t nLen = IMin(td_.baseN, td_.N - nStart);

            /* GM 子块起点：
             *   x1: trans=0 存储 (M,K) -> 偏移 mStart*K；trans=1 存储 (K,M) -> 偏移 mStart
             *   x2: trans=0 存储 (K,N) -> 偏移 nStart；  trans=1 存储 (N,K) -> 偏移 nStart*K */
            const int64_t b1 = static_cast<int64_t>(b);
            const int64_t aOff = td_.transX1 ? b1 * td_.K * td_.M + mStart
                                             : b1 * td_.M * td_.K + static_cast<int64_t>(mStart) * td_.K;
            const int64_t bOff = td_.transX2 ? b1 * td_.N * td_.K + static_cast<int64_t>(nStart) * td_.K
                                             : b1 * td_.K * td_.N + nStart;

            mmObj_.SetTensorA(x1Gm_[aOff], td_.transX1 != 0);
            mmObj_.SetTensorB(x2Gm_[bOff], td_.transX2 != 0);
            mmObj_.SetTail(mLen, nLen);  // 每单元显式设置，避免上一单元尾状态残留

            while (mmObj_.template Iterate<true>()) {
                auto cLocal = cQue_.AllocTensor<float>();
                mmObj_.template GetTensorC<true>(cLocal, false, true);
                /* EnQue/DeQue 建立 fixpipe 写入与 Vector 读取之间的依赖 */
                cQue_.EnQue(cLocal);
                cLocal = cQue_.template DeQue<float>();
                ReduceUnit(cLocal, mLen, nLen, u);
                cQue_.FreeTensor(cLocal);
            }
            mmObj_.End();
        }
    }

    /* 单个子单元：C 块 [mLen x nLen] (fp32) -> 每行 max -> 写 ws[u*mTile ...] */
    __aicore__ inline void ReduceUnit(const AscendC::LocalTensor<float> &cLocal, int32_t mLen,
                                      int32_t nLen, int32_t u)
    {
        auto rowMaxLocal = rowMaxBuf_.Get<float>();
        if (nLen == td_.baseN) {
            /* 主体段（N 对齐，无尾块）：逐行树归约 baseN -> 8 个候选（纯向量） */
            int32_t d = td_.baseN / 2;
            while (d >= 8) {
                for (int32_t r = 0; r < mLen; ++r) {
                    const int32_t row = r * td_.baseN;
                    AscendC::Max(cLocal[row], cLocal[row], cLocal[row + d], d);
                }
                d /= 2;
            }
            AscendC::PipeBarrier<AscendC::PIPE_V>();  // vector 写 -> scalar 读
            for (int32_t r = 0; r < mLen; ++r) {      // 8 候选 -> 1（标量，deterministic）
                const int32_t row = r * td_.baseN;
                float v = cLocal.GetValue(row);
                for (int32_t j = 1; j < 8; ++j) {
                    v = ScalarMax(v, cLocal.GetValue(row + j));
                }
                rowMaxLocal.SetValue(r, v);
            }
        } else {
            /* N 尾块：SetTail 后 GetTensorC(enSeq=true) 为顺序写入，
             * 数据按 tailM x tailN 紧凑排布（官方尾块示例语义），行内元素数少，
             * 直接标量归约（-inf 初始化，保证全负相似度正确） */
            for (int32_t r = 0; r < mLen; ++r) {
                const int32_t row = r * nLen;
                float v = -INFINITY;
                for (int32_t j = 0; j < nLen; ++j) {
                    v = ScalarMax(v, cLocal.GetValue(row + j));
                }
                rowMaxLocal.SetValue(r, v);
            }
        }
        AscendC::PipeBarrier<AscendC::PIPE_V>();  // scalar 写 -> MTE3 读
        /* 写 workspace 中转区（长度按 8 元素对齐满足 32B 搬运粒度；
         * 越过 mLen 的行为垃圾，Phase 2 只读取有效行） */
        const int32_t copyLen = ALIGN_UP_I32(mLen, 8);
        AscendC::DataCopy(wsGm_[static_cast<int64_t>(u) * td_.mTile], rowMaxLocal, copyLen);
    }

    /* ---------------- Phase 2：跨 N 段合并 + 确定性求和 -> y ---------------- */
    __aicore__ inline void Phase2Reduce()
    {
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();
        auto ph2Local = ph2Buf_.Get<float>();
        for (int32_t b = static_cast<int32_t>(blockIdx); b < td_.B; b += static_cast<int32_t>(blockNum)) {
            float sum = 0.0f;  // 固定顺序：mt 升序、行升序标量累加（多次执行一致）
            for (int32_t mt = 0; mt < td_.mTilesPerBatch; ++mt) {
                const int32_t mStart = mt * td_.mTile;
                const int32_t mLen = IMin(td_.mTile, td_.M - mStart);
                const int64_t srcBase =
                    (static_cast<int64_t>(b) * td_.mTilesPerBatch + mt) * td_.segS * td_.mTile;
                AscendC::PipeBarrier<AscendC::PIPE_V>();  // 前一轮 scalar 读 -> 本轮 MTE2 写
                AscendC::DataCopy(ph2Local, wsGm_[srcBase],
                                  static_cast<uint32_t>(td_.segS) * td_.mTile);
                AscendC::PipeBarrier<AscendC::PIPE_V>();  // MTE2 写 -> vector 读
                /* 跨 segS 段线性向量合并：ph2[0:mTile] <- max over s */
                for (int32_t sIdx = 1; sIdx < td_.segS; ++sIdx) {
                    AscendC::Max(ph2Local, ph2Local,
                                 ph2Local[static_cast<int32_t>(sIdx) * td_.mTile], td_.mTile);
                }
                AscendC::PipeBarrier<AscendC::PIPE_V>();  // vector 写 -> scalar 读
                for (int32_t m = 0; m < mLen; ++m) {
                    sum += ph2Local.GetValue(m);
                }
            }
            yGm_.SetValue(b, sum);
        }
    }

private:
    BmmsTilingData td_{};
    AscendC::tiling::TCubeTiling tc_{};

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> cQue_;  // Matmul C 输出（fp32）
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowMaxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ph2Buf_;

    AscendC::GlobalTensor<T> x1Gm_;
    AscendC::GlobalTensor<T> x2Gm_;
    AscendC::GlobalTensor<float> yGm_;
    AscendC::GlobalTensor<float> wsGm_;

    matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                   matmul::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>,
                   matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>>
        mmObj_;
};

/* 从 GM tiling 拷贝 POD（每核一次，标量读，量级 ~200B 可忽略） */
__aicore__ inline void LoadTilingData(GM_ADDR tiling, BmmsTilingData &td)
{
    __gm__ int32_t *p = reinterpret_cast<__gm__ int32_t *>(tiling);
    int32_t *dst = reinterpret_cast<int32_t *>(&td);
    constexpr int32_t total = static_cast<int32_t>(sizeof(BmmsTilingData) / sizeof(int32_t));
    for (int32_t i = 0; i < total; ++i) {
        dst[i] = p[i];
    }
}

extern "C" __global__ __aicore__ void batch_matmul_max_sum(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    /* MIX 分离形态：核函数于 AIV 侧执行，Matmul 由 AIV 发起、AIC 协同计算，
     * C 经 VECIN 回到本核做 Vector 归约（Atlas A2 标准融合模式） */
    AscendC::SetSysWorkspace(workspace);
    BmmsTilingData td;
    LoadTilingData(tiling, td);
    if (td.isHalf != 0) {
        KernelBmms<half> op;
        op.Init(td, x1, x2, y, workspace);
        op.Process();
    } else {
        KernelBmms<AscendC::bfloat16_t> op;
        op.Init(td, x1, x2, y, workspace);
        op.Process();
    }
}
