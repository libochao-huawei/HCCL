#ifndef CCU_TEMP_ALL_TO_ALL_V_MESH2DIE_H
#define CCU_TEMP_ALL_TO_ALL_V_MESH2DIE_H

#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_kernel_alg_base.h"
#include "template_utils.h"

namespace ops_hccl {

class CcuTempAlltoAllVMesh2Die : public CcuAlgTemplateBase {
public:
    CcuTempAlltoAllVMesh2Die() = default;
    explicit CcuTempAlltoAllVMesh2Die(const OpParam& param, const u32 rankId,
                                       const std::vector<std::vector<u32>> &subCommRanks);

    ~CcuTempAlltoAllVMesh2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of All to All V ccu mesh 2Die with rankSize[%u]", templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
                       AlgResourceRequest& resourceRequest) override;
    HcclResult KernelRun(const OpParam& param,
                         const TemplateDataParams& templateDataParams,
                         TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;

    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

    void SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo);

private:
    HcclResult PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channels);

    constexpr static uint32_t DIE_NUM = 2;

    bool withMyRank_{false};
    std::vector<std::vector<HcclChannelDesc>> channels_;
    std::vector<std::vector<uint32_t>> rankGroup_;
    A2ASendRecvInfo localSendRecvInfo_;
};

}

#endif
