/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ccu context arg header file
 * Create: 2025-02-13
 */

#ifndef CCU_KERNEL_ARG_H
#define CCU_KERNEL_ARG_H

#include <vector>

#include "ccu_kernel_signature.h"
#include "hcomm_primitives.h"

namespace hcomm {

class CcuKernelArg {
public:
    explicit CcuKernelArg() = default;
    virtual ~CcuKernelArg() = default;
    virtual CcuKernelSignature GetKernelSignature() const = 0;
    
    std::vector<ChannelHandle> channels{};
};

}; // namespace hcomm

#endif // _CCU_KERNEL_ARG_H