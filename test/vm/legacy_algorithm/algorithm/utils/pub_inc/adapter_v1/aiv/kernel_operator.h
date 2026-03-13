#ifndef TEST_KERNEL_OPERATOR_H
#define TEST_KERNEL_OPERATOR_H

#include "hccl_types.h"
#include "aiv_base_stub.h"
#include "aiv_copy_stub.h"
#include "aiv_memory_stub.h"
#include "aiv_sync_stub.h"

// 解决部分编译问题，将inline定义为空
#define inline

enum KernelType {
    K_TYPE_AICORE = 1,              // c100/m200
    K_TYPE_AIC = 2,                 // v220-cube
    K_TYPE_AIV = 3,                 // v220-vec
    K_TYPE_MIX_AIC_MAIN = 4,        // v220 mix cube/vector 1:2
    K_TYPE_MIX_AIV_MAIN = 5,        // v220 mix vector/cube 1:2
    K_TYPE_AIC_ROLLBACK = 6,        // v220-cube，aic rollback
    K_TYPE_AIV_ROLLBACK = 7,        // v220-vec，aiv rollback
    K_TYPE_MAX
};

struct BaseTlv {  // TLV头部定义
    unsigned short type;
    unsigned short len;
};

enum FuncMetaType { // 函数级TLV类型
    F_TYPE_KTYPE = 1, // kernel type tlv
    F_TYPE_CROSS_CORE_SYNC = 2, // cross core sync
    F_TYPE_MIX_TASK_RATION = 3, // MIX CORE TYPE
    F_TYPE_L0_EXCEPTION_DFX = 4, // DFX tlv for header
    F_TYPE_L0_EXCEPTION_DFX_ARGSINFO = 5, // DFX tlv for args info
    F_TYPE_L0_EXCEPTION_DFX_IS_TIK = 6, // DFX tlv mark for TIK
    F_TYPE_MAX
};

struct FunMetaKType {
    BaseTlv head;
    unsigned int ktype;
};

struct FunLevelKType {
    struct FunMetaKType ktypeMeta;
};

#endif