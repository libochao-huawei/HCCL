/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: ccu executor resource manager
 * Author: z00445483
 */

#ifndef HCCL_SIM_CCU_RESOURCE_MANAGER_H
#define HCCL_SIM_CCU_RESOURCE_MANAGER_H

#include <mutex>
#include <atomic>
#include <vector>
#include <map>
#include <memory>
#include <set>
#include "rts_stub.h"
#include "FakeStreamMgr.h"
#include "CcuResourceV1.h"

using namespace std;

enum CcuVersion {CCU_INVALID, CCU_V1, CCU_V2};

constexpr uint16_t U16_INVALID = UINT16_MAX;
constexpr uint32_t U32_INVALID = UINT32_MAX;
constexpr uint32_t S32_INVALID = INT32_MAX;
constexpr uint64_t U64_INVALID = UINT64_MAX;

struct CcuResData {
    CcuVersion version{CcuVersion::CCU_V1};
    std::unique_ptr<CcuResouceV1> v1Res;
};

class CcuResouceManager {
public:
    CcuResouceManager(const CcuResouceManager&) = delete;
    CcuResouceManager& operator=(const CcuResouceManager&) = delete;

    static CcuResouceManager& GetInstance() {
        static CcuResouceManager instance;
        return instance;
    }

    void Init(int rankId, int rankSize, CcuVersion version);
    void InitInstrInfo(const array<CcuInstrData, DIE_NUM> &ccuInstrInfo, std::map<int, std::vector<FakeSqe>> sqeQueues);
    void InitChannelId2RmtRankMap(int rankId, int dieId, uint16_t channelId, int rmtRank, uint16_t rmtDieId);

    void AddTaskInfo(int dieId, const rtCcuTaskInfo_t &ccuTaskInfo); // 收集SQE参数信息
    uint64_t GetXnValue(int rankId, int dieId, uint16_t xnId) const;
    uint64_t GetGsaValue(int rankId, int dieId, uint16_t gsaId) const;
    uint16_t GetCkeValue(int rankId, int dieId, uint16_t ckeId) const;
    char *GetMsAddr(int rankId, int dieId, uint16_t msId) const;
    std::pair<int, int> GetRmtCcu(int dieId, uint16_t channelId) const;

    void UpdateXnValue(int rankId, int dieId, uint16_t xnId, uint64_t value);
    void UpdateGsaValue(int rankId, int dieId, uint16_t gsaId, uint64_t value);
    void UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value);
    void TransMemToMem(void *srcBuf, void *dstBuf, uint64_t length);
    void TransMSToMS(int srcRank, int srcDie, int dstRank, int dstDie, uint16_t srcMsId, uint16_t dstMsId, uint16_t length);
    void TransMSToMem(int rankId, int dieId, uint16_t msId, void *buf, uint16_t length);
    void TransMemToMS(int rankId, int dieId, uint16_t msId, void *buf, uint16_t length);

    uint64_t GetSqeArgValue(int rankId, int dieId, uint16_t argId) const;
    uint16_t GetInstrCnt(int dieId) const;
    std::vector<Hccl::CcuRep::CcuInstr> GetInstrData(int dieId) const;
    std::string GetInstrDescribe(int dieId, int instrId) const;

    void DumpChannelId2RmtRank(int dieId) const;
    void DumpCcuInstructions() const;
    void DumpCcuAllResouceInfo() const;
    void DumpCcuXnResouceInfo() const;
    void DumpCcuGsaResouceInfo() const;
    void DumpCcuCkeResouceInfo() const;
    void DumpCcuChannelResouceInfo() const;
    void GeneCaModelFile(const array<CcuInstrData, DIE_NUM> &ccuInstrInfo, std::map<int, std::vector<FakeSqe>> sqeQueues) const;
    string& GetCcuInfoFilePath();
private:
    CcuResouceManager() = default;
    ~CcuResouceManager() = default;

private:
    bool enableDump_{false};
    int rankId_{0};
    CcuResData ccuResData_{}; // ccu资源数据
    std::mutex ccuExecutorMutex;
    string GeneCcuInfoFilePath; //ca model使用
};

#endif // HCCL_SIM_CCU_RESOURCE_MANAGER_H
