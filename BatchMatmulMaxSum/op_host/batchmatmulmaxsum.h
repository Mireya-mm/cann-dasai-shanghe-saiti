/**
 * Copyright 2026 The CANN Contest Authors. All Rights Reserved.
 *
 * BatchMatmulMaxSum 算子原型注册（上合赛区初赛算子赛题）
 *
 * 计算语义：y[b] = sum_m max_n( sum_k X1[b,m,k] * X2[b,k,n] )
 *   - X1: transposeX1=0 时存储布局 (B,M,K)，=1 时存储布局 (B,K,M)
 *   - X2: transposeX2=0 时存储布局 (B,K,N)，=1 时存储布局 (B,N,K)
 *   - 属性只声明存储布局，不参与数学计算（输入始终按逻辑 (M,K)x(K,N) 相乘）
 *   - 点积累加 / Max / Sum 均在 FP32 域进行；全负相似度时行 max 取 -inf 初始化
 *   - 输出 y: (B,) FP32
 */
#ifndef BATCH_MATMUL_MAX_SUM_H
#define BATCH_MATMUL_MAX_SUM_H

#include "register/op_impl_registry.h"

namespace ge {
REG_OP(BatchMatmulMaxSum)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(transposeX1, Bool)
    .REQUIRED_ATTR(transposeX2, Bool)
    .OP_END_FACTORY_REG(BatchMatmulMaxSum)
}  // namespace ge

#endif  // BATCH_MATMUL_MAX_SUM_H
