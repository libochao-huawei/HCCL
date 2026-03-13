
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: sal rts stub
 * Author:
 * Create: 2025-05-29
 */

#include <stdexcept>
#include "sal.h"
#include "log.h"

namespace Hccl {

/* 华为安全库函数 */
#if T_DESC("华为安全库函数适配", 1)

#include <securec.h>

/* 华为安全函数返回值转换 */
#define HUAWEI_SECC_RET_CHECK_AND_RETURN(ret) do { \
    switch (ret) {                        \
        case EOK:                         \
            return HCCL_SUCCESS;          \
        case EINVAL:                      \
            return HCCL_E_PARA;           \
        default:                          \
            return HCCL_E_INTERNAL;       \
    }                                     \
} while (0)
#define HUAWEI_SECC_RET_TRANSFORM(ret) ((ret == EOK) ? HCCL_SUCCESS : ((ret == EINVAL) ? HCCL_E_PARA : HCCL_E_INTERNAL))
#endif

std::string SalGetEnv(const char *name)
{
    if (name == nullptr || getenv(name) == nullptr) {
        return "EmptyString";
    }

    return getenv(name);
}

// 返回当前进程ID
s32 SalGetPid()
{
    return getpid();
}

// 返回当前线程ID
s32 SalGetTid()
{
    return syscall(SYS_gettid);
}

s32 sal_vsnprintf(char *strDest, size_t destMaxSize, size_t count, const char *format, va_list argList)
{
    /* 返回值不是错误码,无需转换 */
    CHK_PTR_NULL(strDest);
    CHK_PTR_NULL(format);
    return vsnprintf_s(strDest, destMaxSize, count, format, argList);
}

constexpr u32 INVALID_UINT = 0xFFFFFFFF;
// 字串符转换成无符号整型
HcclResult SalStrToULong(const std::string str, int base, u32 &val)
{
    try {
        u64 tmp = std::stoull(str, nullptr, base);
        if (tmp > INVALID_UINT) {
            HCCL_ERROR("[Transform][StrToULong]stoul out of range, str[%s] base[%d] val[%llu]", str.c_str(), base, tmp);
            return HCCL_E_PARA;
        } else {
            val = static_cast<u32>(tmp);
        }
    }
    catch (std::invalid_argument&) {
        HCCL_ERROR("[Transform][StrToULong]stoull invalid argument, str[%s] base[%d] val[%u]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (std::out_of_range&) {
        HCCL_ERROR("[Transform][StrToULong]stoull out of range, str[%s] base[%d] val[%u]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    catch (...) {
        HCCL_ERROR("[Transform][StrToULong]stoull catch errror, str[%s] base[%d] val[%u]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult sal_memset(void *dest, size_t destMaxSize, int c, size_t count)
{
    CHK_PTR_NULL(dest);
    s32 ret = memset_s(dest, destMaxSize, c, count);
    if (ret != EOK) {
        HCCL_ERROR("errNo[0x%016llx] In sal_memset, memset_s failed. errorno[%d], params: dest[%p], "\
            "destMaxSize[%d], c[%d], count[%d]", HCCL_ERROR_CODE(HUAWEI_SECC_RET_TRANSFORM(ret)), ret, dest, \
            destMaxSize, c, count);
    }
    HUAWEI_SECC_RET_CHECK_AND_RETURN(ret);
}

}
