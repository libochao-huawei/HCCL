/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: 集合通信域头文件
 * Author:
 * Create:
 */
#ifndef ADAPTER_V2_COMMUNICATOR_IMPL_H
#define ADAPTER_V2_COMMUNICATOR_IMPL_H

#include <set>
#include <vector>
#include <unordered_map>
#include <string>
#include <atomic>
#include <chrono>
#include <memory>
#include "types.h"
#include "op_mode.h"
#include "hccl_params_pub.h"
#include "comm_type.h"
#include "coll_service_stub.h"
#include "coll_operator.h"
#include "dev_buffer.h"
#include "topo_info.h"
#include "ccu_ins_preprocessor.h"

using namespace std;
namespace Hccl {

using HcclUs = std::chrono::steady_clock::time_point;
class CommunicatorImpl {
public:
    HcclResult Init(const CommParams &commParams, const std::string &ranktableM,
        std::string& topoPath, const HcclCommConfig &config);
    HcclResult LoadOpbasedCollOp(CollOpParams &opParams, std::string& algName);
    HcclResult LoadOffloadCollOp(std::string &opTag, CollOpParams &opParams);

    RankId GetMyRank() const;
    u32 GetRankSize() const;
    u64 GetBufferSize() const;
    bool GetCcuFeatureFlag() const;
    void CovertToCurrentCollOperator(std::string &opTag, CollOpParams &opParams, OpMode opMode);
    CollOperator* GetCurrentCollOperator() const;

    DevId GetDevPhyId()
    {
        return devPhyId;
    }

    const shared_ptr<DevBuffer> GetCclBuffer() const
    {
        return cclBuffer;
    }
    ~CommunicatorImpl();

    RankGraph* GetRankGraph() const
    {
        HCCL_DEBUG("GetRankGraph ");
        return newVirtualTopo.get();
    }

    const DevType& GetDevType() const
    {
        return devType;
    }

    CollServiceStub* GetCollService() const;

    CcuInsPreprocessor* GetCcuInsPreprocessor()
    {
        return collService->GetCcuInsPreprocessor();
    }

    void TransformTask()
    {
        return collService->TransformTask();
    }
    bool IsOpUsingCcuMs() const;
    bool IsOpUsingCcuSched() const;
    bool IsCommUsingCcuMs() const;
    bool IsCommUsingCcuSched() const;
    HcclResult ReLoadCollOp();
    void AcceleratorFallback();
    bool IfAccStateConfigExplicitly() const;
private:
    std::string                                id;
    RankId                                     myRank;
    u32                                        rankSize;
    RankId                                     rankInParentComm;
    DevType                                    devType;
    DevId                                      devPhyId;
    DevId                                      devLogicId;
    HcclCommConfig                             config;
    std::unique_ptr<RankGraph>            newVirtualTopo;
    unique_ptr<CollServiceStub>                collService;
    unique_ptr<CollOperator>                   currentCollOperator;

    bool initFlag = false;
    u32  ccuFeatureFlag{0};
    bool devModeFlag = false;
    bool isWorldGroup = false;

    std::shared_ptr<DevBuffer> cclBuffer;
    u64                        cclBufferSize = 0;

    CommStatus status{CommStatus::COMM_IDLE};                  // 通信域状态
    std::unique_ptr<RankTableInfo>             ranktableInfo;  // 主通信域使用：序列化解析
    std::shared_ptr<TopoInfo>                  topoInfo;       // 主通信域使用：序列化解析

    void InitCcuInstance();
    void InitFeatureFlag();
    void InitCommonData(const CommParams &commParams);
    void InitCommonData(const CommParams &commParams, const HcclCommConfig &commConfig);
    void InitRankGraph(const std::string &ranktableM, std::string& topoPath);
    void AppendLocalDieIdForLinks();
    void CheckVirtualTopo() const;
    void InitDataBufferManager();
    void InitCollService();
    void ConvertCollOperatorA2A(CollOpParams &opParams);
    void ConvertCollOperatorMem(CollOpParams &opParams, u64 size);
    void CalcA2ASendRecvMem(CollOpParams &opParams, u64 &sendSize, u64 &recvSize) const;
};
} // namespace Hccl

#endif // HCCL_COMMUNICATOR_IMPL_H
