#ifndef HCCL_ONE_SIDED_SERVICES_DL_H
#define HCCL_ONE_SIDED_SERVICES_DL_H

// #include "hccl_one_sided_services.h"   // 原始头文件，包含所有声明和类型定义
#include <hccl/hccl_types.h>
#include <hccl/base.h>

#ifdef __cplusplus
extern "C" {
#endif

// 需优化
typedef enum {
    HCCL_MEM_TYPE_DEVICE, ///< 设备侧内存（如NPU等）
    HCCL_MEM_TYPE_HOST,   ///< 主机侧内存
    HCCL_MEM_TYPE_NUM     ///< 内存类型数量
} HcclMemType;
/**
 * @struct HcclMem
 * @brief 内存段元数据描述结构体
 * @var type  - 内存物理位置类型，参见HcclMemType
 * @var addr  - 内存虚拟地址
 * @var size  - 内存区域字节数
 */
typedef struct {
    HcclMemType type;
    void *addr;
    uint64_t size;
} HcclMem;

const u32 HCCL_MEM_DESC_LENGTH = 511;

typedef struct {
    char desc[HCCL_MEM_DESC_LENGTH + 1]; // 具体内容对调用者不可见
} HcclMemDesc;

typedef struct {
    HcclMemDesc* array;
    u32 arrayLength;
} HcclMemDescs;

typedef struct {
    void* localAddr; // 本端VA
    void* remoteAddr; // 远端VA
    u64 count;
    HcclDataType dataType;
} HcclOneSideOpDesc;

typedef enum {
    HCCL_TOPO_FULLMESH = 0, // fullmesh连接
    HCCL_TOPO_NUM,
} HcclTopoType;

typedef struct {
    HcclTopoType topoType;
    u64 rsvd0;
    u64 rsvd1;
    u64 rsvd2;
} HcclPrepareConfig;

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcclRegisterMemPtr)(HcclComm, u32, int, void*, u64, HcclMemDesc*);
extern HcclResult (*hcclDeregisterMemPtr)(HcclComm, HcclMemDesc*);
extern HcclResult (*hcclExchangeMemDescPtr)(HcclComm, u32, HcclMemDescs*, int, HcclMemDescs*, u32*);
extern HcclResult (*hcclEnableMemAccessPtr)(HcclComm, HcclMemDesc*, HcclMem*);
extern HcclResult (*hcclDisableMemAccessPtr)(HcclComm, HcclMemDesc*);
extern HcclResult (*hcclBatchPutPtr)(HcclComm, u32, HcclOneSideOpDesc*, u32, rtStream_t);
extern HcclResult (*hcclBatchGetPtr)(HcclComm, u32, HcclOneSideOpDesc*, u32, rtStream_t);
extern HcclResult (*hcclRemapRegistedMemoryPtr)(HcclComm*, HcclMem*, u64, u64);
extern HcclResult (*hcclRegisterGlobalMemPtr)(const HcclMem*, void**);
extern HcclResult (*hcclDeregisterGlobalMemPtr)(void*);
extern HcclResult (*hcclCommBindMemPtr)(HcclComm, void*);
extern HcclResult (*hcclCommUnbindMemPtr)(HcclComm, void*);
extern HcclResult (*hcclCommPreparePtr)(HcclComm, const HcclPrepareConfig*, const int);

// 宏：将原始API名映射为函数指针调用
#define HcclRegisterMem                (*hcclRegisterMemPtr)
#define HcclDeregisterMem              (*hcclDeregisterMemPtr)
#define HcclExchangeMemDesc            (*hcclExchangeMemDescPtr)
#define HcclEnableMemAccess            (*hcclEnableMemAccessPtr)
#define HcclDisableMemAccess           (*hcclDisableMemAccessPtr)
#define HcclBatchPut                   (*hcclBatchPutPtr)
#define HcclBatchGet                   (*hcclBatchGetPtr)
#define HcclRemapRegistedMemory        (*hcclRemapRegistedMemoryPtr)
#define HcclRegisterGlobalMem           (*hcclRegisterGlobalMemPtr)
#define HcclDeregisterGlobalMem         (*hcclDeregisterGlobalMemPtr)
#define HcclCommBindMem                  (*hcclCommBindMemPtr)
#define HcclCommUnbindMem                (*hcclCommUnbindMemPtr)
#define HcclCommPrepare                  (*hcclCommPreparePtr)

// 查询函数声明
bool HcommIsSupportHcclRegisterMem(void);
bool HcommIsSupportHcclDeregisterMem(void);
bool HcommIsSupportHcclExchangeMemDesc(void);
bool HcommIsSupportHcclEnableMemAccess(void);
bool HcommIsSupportHcclDisableMemAccess(void);
bool HcommIsSupportHcclBatchPut(void);
bool HcommIsSupportHcclBatchGet(void);
bool HcommIsSupportHcclRemapRegistedMemory(void);
bool HcommIsSupportHcclRegisterGlobalMem(void);
bool HcommIsSupportHcclDeregisterGlobalMem(void);
bool HcommIsSupportHcclCommBindMem(void);
bool HcommIsSupportHcclCommUnbindMem(void);
bool HcommIsSupportHcclCommPrepare(void);

// 动态库管理接口
void HcclOneSidedServicesDlInit(void* libHcommHandle);
void HcclOneSidedServicesDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_ONE_SIDED_SERVICES_DL_H