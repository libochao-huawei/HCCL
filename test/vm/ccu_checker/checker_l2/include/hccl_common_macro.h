#ifndef HCCL_COMMON_MACRO_H
#define HCCL_COMMON_MACRO_H

#include <iostream>

#define UNLIKELY(x) __builtin_expect(!!(x), 0)

/* 检查函数返回值, 并返回指定错误码 */
#define HCCLVM_CHK_RET(call)                                                                    \
    do {                                                                                        \
        auto hcclRet = call;                                                                    \
        if (UNLIKELY(hcclRet != 0)) {                                                           \
            std::cout << "[" << __func__ << "]call trace: hcclRet -> " << hcclRet << std::endl; \
            return hcclRet;                                                                     \
        }                                                                                       \
    } while (0)

/* 检查指针, 若指针为NULL, 则记录日志, 并返回错误 */
#define HCCLVM_CHK_PTR(ptr)                                                                           \
    do {                                                                                              \
        if (UNLIKELY((ptr) == nullptr)) {                                                             \
            std::cout << "[" << __func__ << "]errNo[0x" << std::hex << HCCL_SIM_E_PTR << "]ptr [" << #ptr \
                      << "] is nullptr, return HCCL_E_PTR" << std::endl;                              \
            return HCCL_SIM_E_PTR;                                                                        \
        }                                                                                             \
    } while (0)

#endif //HCCL_COMMON_MACRO_H
