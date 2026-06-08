/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_OMNI_H
#define HCCL_CCU_TEMP_OMNI_H

#include "utils.h"
#include "ccu_alg_template_base.h"
#include "kernel/ccu_kernel_omni.h"

namespace ops_hccl {

using RankGroup = std::vector<RankId>;

class CcuTempOmni : public CcuAlgTemplateBase {
public:
    explicit  CcuTempOmni(const OpParam& param, 
                                        const u32 rankId, // 传通信域的rankId，userRank
                                        const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempOmni() override;

    std::string Describe() const override
    {
        return StringFormat("Template of omni with tempRankSize [%u].", tempRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest, const XmlInfo& xmlInfo);

    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         const TemplateResource& templateResource);

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    HcclResult CalcChannelRequestOmni(HcclComm comm, 
        const OpParam& param, 
        const TopoInfoWithNetLayerDetails* topoInfo,
        const std::vector<std::vector<u32>>& subcommInfo, 
        const std::vector<std::map<u32, OmniChannelInfo>>& mapchannelInfo, 
        std::vector<HcclChannelDesc> &channels);

    HcclResult PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs);

    HcclResult ProcessLinkForProtocol(HcclComm comm, const std::vector<CommProtocol>& expectedProtocols,
        const std::vector<CommLink>& linkList, u32 myRank, u32 remoteRank, uint32_t netLayer,
        std::vector<HcclChannelDesc>& channels, bool& protocolFound, const std::string& funcName);

    HcclResult CreateChannelFromLink(HcclComm comm, u32 myRank, u32 rank, uint32_t netLayer, u32 idx,
        const CommLink& link, const std::string& funcName, std::vector<HcclChannelDesc>& channels);

    uint32_t tempRankSize_ = 0;
    uint32_t dieNum_ = 0;
    std::map<uint32_t, std::vector<HcclChannelDesc>> channels_; // key is DieId
    std::map<uint32_t, RankGroup> rankGroup_; // key is DieId 

    uint32_t mySubCommRank_ = 0;
};

}// namespace ops_hccl

#endif// HCCL_CCU_TEMP_OMNI_H