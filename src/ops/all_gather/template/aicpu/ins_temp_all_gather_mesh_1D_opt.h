/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_ALL_GATHER_MESH_1D_OPT_H
#define INS_TEMP_ALL_GATHER_MESH_1D_OPT_H

#include "alg_v2_template_base.h"
#include "executor_base.h"

namespace ops_hccl {

class InsTempAllGatherMesh1DOpt : public InsAlgTemplateBase {
public:
    InsTempAllGatherMesh1DOpt() = default;
    explicit InsTempAllGatherMesh1DOpt(const OpParam &param, const u32 rankId,  // 传通信域的rankId，userRank
                                    const std::vector<std::vector<u32>> &subCommRanks);
    // Host侧调用
    ~InsTempAllGatherMesh1DOpt() override;

    std::string Describe() const override
    {
        std::string info = "Template of all gather Mesh with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }
    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &tempAlgParams,
                         TemplateResource &templateResource) override;
    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
                       AlgResourceRequest &resourceRequest) override;
    virtual HcclResult GetRes(AlgResourceRequest &resourceRequest) const;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    virtual u64 GetThreadNum() const;
    void GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain) override;
    
    void SetMeshDimensions(u32 rankSize, u32 myRank, u32 meshSize, u32 closSize)
    {
        rankSize_ = rankSize;
        myRank_ = myRank;
        meshSize_ = meshSize;
        closSize_ = closSize;
    }

    void SetRemoteWrite(bool remoteWriteFlag)
    {
        remoteWrite = remoteWriteFlag;
    }

    void SetTemplateDataParams1(const TemplateDataParams &params)
    {
        tempAlgParams1_ = params;
    }

protected:
    virtual HcclResult RunAllGatherMesh(const std::vector<ThreadHandle> &threads,
                                                        const std::map<u32, std::vector<ChannelInfo>> &channels);
    TemplateDataParams tempAlgParams_;
    TemplateDataParams tempAlgParams1_;

    bool remoteWrite = false; // 是否存在远端写，存在远端写则本地复制需要等远端写完成后才能进行
    u32 rankSize_{0};
    u32 myRank_{0};
    u32 meshSize_{0};
    u32 closSize_{0};
};

}  // namespace ops_hccl

#endif  // INS_TEMP_ALL_GATHER_MESH_1D_H