/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: CcuErrorHandler
 * Create: 2025-04-13
 */

#ifndef CCU_ERROR_HANDLER_H
#define CCU_ERROR_HANDLER_H

#include <vector>
#include "hccl_types.h"
#include "task_param.h"
#include "ccu_error_info.h"
#include "ccu_rep_base.h"
#include "ccu_rep_context.h"

using namespace hcomm;
namespace Hccl {

struct CcuMissionContext {
    union {
        uint16_t value;
        uint16_t taskId;
    } part0;

    union {
        uint16_t value;
        uint16_t streamId;
    } part1;

    union {
        uint16_t value;
        struct {
            uint16_t taskKill : 1;
            uint16_t dieId : 2;
            uint16_t status : 13; // Status [12:0]
        };
    } part2;

    union {
        uint16_t value;
        struct {
            uint16_t status : 3; // Status [15:13]
            uint16_t counter : 8;
            uint16_t denyCnt : 5;
        };
    } part3;

    union {
        uint16_t value;
        struct {
            uint16_t denyCnt : 5;
            uint16_t currentIns : 11; // Current Ins [10:0]
        };
    } part4;

    union {
        uint16_t value;
        struct {
            uint16_t currentIns : 5; // Current Ins [15:11]
            uint16_t endIns : 11;
        };
    } part5;

    union {
        uint16_t value;
        struct {
            uint16_t endIns : 5;
            uint16_t startIns : 11;
        };
    } part6;

    union {
        uint16_t value;
        struct {
            uint16_t startIns : 5;
            uint16_t profileEn : 1;
            uint16_t missionVld : 1;
            uint16_t reserved : 9;
        };
    } part7;

    uint16_t reserved[24]; // part 8-31

    uint16_t GetStatus() const
    {
        return (part3.status << 13) | (part2.status);   // part3.status为[15:13]位
    }

    uint16_t GetCurrentIns() const
    {
        return (part5.currentIns << 11) | (part4.currentIns);   // part5.currentIns为[15:11]位
    }
};

struct CcuLoopContext {
    union {
        uint16_t value;
        uint16_t timestamp;
    } part0;

    union {
        uint16_t value;
        uint16_t timestamp;
    } part1;

    union {
        uint16_t value;
        struct {
            uint16_t timestamp : 4;
            uint16_t ckeOffset : 10;
            uint16_t msOffset : 2;
        };
    } part2;

    union {
        uint16_t value;
        struct {
            uint16_t msOffset : 9;
            uint16_t addrOffset : 7;
        };
    } part3;

    union {
        uint16_t value;
        uint16_t addrOffset;
    } part4;

    union {
        uint16_t value;
        struct {
            uint16_t addrOffset : 9;
            uint16_t ckBit : 7;
        };
    } part5;

    union {
        uint16_t value;
        uint16_t ckBit;
    } part6;

    union {
        uint16_t value;
        struct {
            uint16_t ckBit : 9;
            uint16_t perfMode : 1;
            uint16_t waitLoopCkbitValue : 6;
        };
    } part7;

    union {
        uint16_t value;
        uint16_t waitLoopCkbitValue;
    } part8;

    union {
        uint16_t value;
        struct {
            uint16_t waitLoopCkbitValue : 10;
            uint16_t currentIns : 6;    // Current_ins [5:0]
        };
    } part9;

    union {
        uint16_t value;
        struct {
            uint16_t currentIns : 10;   // Current_ins [15:6]
            uint16_t addrStride : 6;    // Addr_stride [5:0]
        };
    } part10;

    union {
        uint16_t value;
        uint16_t addrStride;            // Addr_stride [21:6]
    } part11;

    union {
        uint16_t value;
        struct {
            uint16_t addrStride : 10;   // Addr_stride [31:22]
            uint16_t denyCnt : 6;
        };
    } part12;

    union {
        uint16_t value;
        struct {
            uint16_t denyCnt : 4;
            uint16_t currentCnt : 12;   // Current_cnt [11:0]
        };
    } part13;

    union {
        uint16_t value;
        struct {
            uint16_t currentCnt : 1;    // Current_cnt [12]
            uint16_t totalCnt : 13;
            uint16_t endIns : 2;
        };
    } part14;

    union {
        uint16_t value;
        struct {
            uint16_t endIns : 14;
            uint16_t startIns : 2;
        };
    } part15;

    union {
        uint16_t value;
        struct {
            uint16_t startIns : 14;
            uint16_t missionId : 2;
        };
    } part16;

    union {
        uint16_t value;
        struct {
            uint16_t missionId : 2;
            uint16_t reserved : 14;
        };
    } part17;

    uint16_t reserved[14]; // part 18-31

    uint16_t GetCurrentIns() const
    {
        return (part10.currentIns << 6) | (part9.currentIns);   // part10.currentIns为[15:6]位
    }

    uint16_t GetCurrentCnt() const
    {
        return (part14.currentCnt << 12) | (part13.currentCnt); // part14.currentCnt为第[12]位
    }

    uint32_t GetAddrStride() const
    {
        const uint32_t low = static_cast<uint32_t>(part10.addrStride);
        const uint32_t mid = static_cast<uint32_t>(part11.addrStride) << 6;     // part11.addrStride为[21:6]位
        const uint32_t high = static_cast<uint32_t>(part12.addrStride) << 22;   // part12.addrStride为[31:22]位
        return high | mid | low;
    }
};

union LoopXm {
    uint64_t value;
    struct {
        uint64_t loopCnt : 13;
        uint64_t gsaStride : 32;
        uint64_t loopCtxId : 8;
        uint64_t reserved : 11;
    };
};

union LoopGroupXn {
    uint64_t value;
    struct {
        uint64_t reservedLow : 41;
        uint64_t loopInsCnt : 7;
        uint64_t expandOffset : 7;
        uint64_t expandCnt : 7;
        uint64_t reservedHigh : 2;
    };
};

union LoopGroupXm {
    uint64_t value;
    struct {
        uint64_t ckOffset : 10;
        uint64_t msOffset : 11;
        uint64_t gsaOffset : 32;
        uint64_t reserved : 11;
    };
};

struct ErrorInfoBase {
    int32_t  deviceId;
    uint8_t  dieId;
    uint8_t  missionId;
    uint16_t currentInsId;
    uint16_t status;
};

class CcuErrorHandler {
public:
    CcuErrorHandler() = delete;
    CcuErrorHandler(const CcuErrorHandler&) = delete;
    void operator=(const CcuErrorHandler&) = delete;

    static void GetCcuErrorMsg(int32_t deviceId, const ParaCcu &ccuTaskParam, std::vector<CcuErrorInfo> &errorInfo);

private:
    static void GenStatusInfo(const ErrorInfoBase &baseInfo, std::vector<CcuErrorInfo> &errorInfo);

    // LoopGroup
    static void GenErrorInfoLoopGroup(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                      CcuRep::CcuRepContext &ctx, std::vector<CcuErrorInfo> &errorInfo);
    // Loop
    static void GenErrorInfoLoop(const ErrorInfoBase &baseInfo, CcuRep::CcuRepContext &ctx,
                                 std::vector<CcuErrorInfo> &errorInfo);

    static void GenErrorInfoByRepType(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                      std::vector<CcuErrorInfo> &errorInfo);
    // Default
    static void GenErrorInfoDefault(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                    std::vector<CcuErrorInfo> &errorInfo);
    // WaitSignal
    static void GenErrorInfoLocPostSem(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                       std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoLocWaitSem(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                       std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoRemPostSem(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                       std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoRemWaitSem(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                       std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoRemPostVar(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                       std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoRemWaitGroup(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                         std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoPostSharedVar(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                          std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoPostSharedSem(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                          std::vector<CcuErrorInfo> &errorInfo);
    // TransMem
    static void GenErrorInfoRead(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                 std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoWrite(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                  std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoLocalCpy(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                     std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoLocalReduce(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                        std::vector<CcuErrorInfo> &errorInfo);
    // BufTransMem
    static void GenErrorInfoBufRead(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                    std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoBufWrite(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                     std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoBufLocRead(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                       std::vector<CcuErrorInfo> &errorInfo);
    static void GenErrorInfoBufLocWrite(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                        std::vector<CcuErrorInfo> &errorInfo);
    // BufReduce
    static void GenErrorInfoBufReduce(const ErrorInfoBase &baseInfo, std::shared_ptr<CcuRep::CcuRepBase> repBase,
                                      std::vector<CcuErrorInfo> &errorInfo);

    static CcuMissionContext GetCcuMissionContext(int32_t deviceId, uint32_t dieId, uint32_t missionId);
    static CcuLoopContext    GetCcuLoopContext(int32_t deviceId, uint32_t dieId, uint32_t loopCtxId);
    static uint64_t          GetCcuXnValue(int32_t deviceId, uint32_t dieId, uint32_t xnId);
    static uint64_t          GetCcuGSAValue(int32_t deviceId, uint32_t dieId, uint32_t gsaId);
    static uint16_t          GetCcuCKEValue(int32_t deviceId, uint32_t dieId, uint32_t ckeId);
};

} // namespace Hccl
#endif // CCU_ERROR_HANDLER_H