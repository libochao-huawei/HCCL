// 完整修复后的代码
#include "channel.h"
#include "hccl_ccu_res.h"
#include "ccu_assist_pub.h"
#include "ccu_kernel_scatter_omnipipe_mesh_1D_mem2mem.h"
#include "ccu_temp_scatter_omnipipe_mesh_1D_mem2mem.h"
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

CcuTempScatterOmniPipeMesh1DMem2Mem::CcuTempScatterOmniPipeMesh1DMem2Mem(
    const OpParam &param, const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : CcuAlgTemplateBase(param, rankId, subCommRanks)
{
    std::vector<u32> ranks = subCommRanks[0];
    templateRankSize_ = ranks.size();
    auto it = std::find(ranks.begin(), ranks.end(), rankId);
    if (it != ranks.end()) {
        mySubCommRank_ = std::distance(ranks.begin(), it);
    }
    auto itRoot = std::find(ranks.begin(), ranks.end(), param.root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
    HCCL_DEBUG(
        "[%s] myRank[%u] mySubCommRank[%u] templateRankSize[%u] subCommRootId_[%d]",
        __func__, rankId, mySubCommRank_, templateRankSize_, subCommRootId_);
}

CcuTempScatterOmniPipeMesh1DMem2Mem::~CcuTempScatterOmniPipeMesh1DMem2Mem()
{
}

void CcuTempScatterOmniPipeMesh1DMem2Mem::SetRoot(u32 root)
{
    HCCL_INFO("[CcuTempScatterOmniPipeMesh1DMem2Mem][SetRoot] myRank_ [%u], set root [%u] ", myRank_, root);
    std::vector<u32> ranks = subCommRanks_[0];
    std::string ranksStr = "";
    for (auto r : ranks) { ranksStr += std::to_string(r) + " "; }
    HCCL_INFO("[CcuTempScatterOmniPipeMesh1DMem2Mem][SetRoot] ranks = subCommRanks[0] is: %s", ranksStr.c_str());
    auto itRoot = std::find(ranks.begin(), ranks.end(), root);
    if (itRoot != ranks.end()) {
        subCommRootId_  = std::distance(ranks.begin(), itRoot);
    }
}

u64 CcuTempScatterOmniPipeMesh1DMem2Mem::GetThreadNum() const
{
    return 1;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::GetRes(AlgResourceRequest& resourceRequest) const
{
    resourceRequest.slaveThreadNum = 0;
    resourceRequest.notifyNumOnMainThread = 0;
    resourceRequest.notifyNumPerThread.assign(resourceRequest.slaveThreadNum, 1);
    return HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::CalcRes(HcclComm comm, const OpParam &param,
    const TopoInfoWithNetLayerDetails *topoInfo, AlgResourceRequest &resourceRequest)
{
    GetRes(resourceRequest);
    resourceRequest.ccuKernelNum.push_back(1);

    HCCL_DEBUG("[%s]notifyNumOnMainThread[%u] slaveThreadNum[%u]", __func__,
        resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);

    CcuKernelInfo kernelInfo;
    kernelInfo.creator = [](const hcomm::CcuKernelArg &arg) {
        return std::make_unique<CcuKernelScatterOmniPipeMesh1DMem2Mem>(arg);
    };

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, channelDescs));

    std::set<uint32_t> mySet;
    std::vector<HcclChannelDesc> myChannels;
    for (auto &channel : channelDescs) {
        if (mySet.count(channel.remoteRank) == 0) {
            mySet.insert(channel.remoteRank);
            myChannels.push_back(channel);
        }
    }

    kernelInfo.kernelArg = std::make_shared<CcuKernelArgScatterOmniPipeMesh1DMem2Mem>(
        subCommRanks_[0].size(), mySubCommRank_, subCommRootId_, param, subCommRanks_, currentStep_, isSameAxisAsRoot_, totalStep_);
    kernelInfo.channels = myChannels;
    resourceRequest.ccuKernelInfos.push_back(kernelInfo);
    resourceRequest.channels.push_back(channelDescs);

    HCCL_DEBUG("[%s]channelDescs.size()=%llu, dimsize=%llu, ccuKernelInfos.size()=%llu", __func__,
        channelDescs.size(), subCommRanks_[0].size(), resourceRequest.ccuKernelInfos.size());

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::KernelRun(
    const OpParam &param, const TemplateDataParams &templateDataParams, TemplateResource &templateResource)
{
    HCCL_DEBUG("[%s] start", __func__);
    buffInfo_ = templateDataParams.buffInfo;
    auto stepSliceInfo = templateDataParams.stepSliceInfo;

    uint64_t inputAddrBase = PointerToAddr(buffInfo_.inputPtr);
    uint64_t outputAddrBase = PointerToAddr(buffInfo_.outputPtr);

    uint64_t inBuffBaseOff = templateDataParams.buffInfo.inBuffBaseOff;
    uint64_t outBuffBaseOff = templateDataParams.buffInfo.outBuffBaseOff;

    uint64_t scratchAddr = PointerToAddr(buffInfo_.hcclBuff.addr) + buffInfo_.hcclBuffBaseOff;
    uint64_t inputAddr = inputAddrBase + inBuffBaseOff;
    uint64_t token = CcuRep::GetTokenInfo(
        reinterpret_cast<uint64_t>(buffInfo_.inputPtr), static_cast<uint64_t>(buffInfo_.inputSize));

    uint64_t localCopyFlag = templateDataParams.localCopyFlag;
    if (localCopyFlag == 0) {
        uint64_t outputAddr = outputAddrBase + outBuffBaseOff;
        uint64_t inputSliceStride = templateDataParams.inputSliceStride + stepSliceInfo.stepInputSliceStride[mySubCommRank_];
        uint64_t outputSliceStride = stepSliceInfo.stepOutputSliceStride[mySubCommRank_];
        uint32_t repeatNum = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_].size();

        HCCL_DEBUG("[%s] myRank[%u] mySubCommRank[%u] repeatNum[%u] inputAddr[%llu] outputAddr[%llu] scratchAddr[%llu]", 
            __func__, myRank_, mySubCommRank_, repeatNum, inputAddr, outputAddr, scratchAddr);

        for (uint32_t rpt = 0; rpt < repeatNum; ++rpt) {
            uint64_t sliceSize = stepSliceInfo.stepSliceSize[mySubCommRank_][rpt];
            if (sliceSize == 0) {
                continue;
            }

            uint64_t inputOmniPipeSliceStride = stepSliceInfo.inputOmniPipeSliceStride[mySubCommRank_][rpt];
            uint64_t outputOmniPipeSliceStride = stepSliceInfo.outputOmniPipeSliceStride[mySubCommRank_][rpt];

            auto taskArg = std::make_unique<CcuTaskArgScatterOmniPipeMesh1DMem2Mem>(
                inputAddr, outputAddr, scratchAddr, sliceSize, 0, token, 0,
                inputSliceStride, outputSliceStride, inputOmniPipeSliceStride,
                outputOmniPipeSliceStride);
            void *taskArgPtr = static_cast<void *>(taskArg.get());

            CHK_RET(HcclCcuKernelLaunch(
                param.hcclComm, templateResource.threads[0], templateResource.ccuKernels[0], taskArgPtr));

            HCCL_DEBUG("[%s] myRank[%u] rpt[%u] sliceSize[%llu] inputOmniPipeSliceStride[%llu] outputOmniPipeSliceStride[%llu]",
                __func__, myRank_, rpt, sliceSize, inputOmniPipeSliceStride, outputOmniPipeSliceStride);
        }
    } else if (localCopyFlag == 1) {
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy start", __func__, myRank_);
        DataSlice srcSlice(buffInfo_.inputPtr, buffInfo_.inBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        DataSlice dstSlice(buffInfo_.outputPtr, buffInfo_.outBuffBaseOff, templateDataParams.sliceSize, templateDataParams.count);
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy inputAddrBase[%llu] inputAddrOffset[%llu] outputAddrBase[%llu] "
                   "outputAddrOffset[%llu] sliceSize[%llu]",
            __func__, myRank_, inputAddrBase, buffInfo_.inBuffBaseOff, outputAddrBase, buffInfo_.outBuffBaseOff,
            templateDataParams.sliceSize);
        CHK_RET(LocalCopy(templateResource.threads[0], srcSlice, dstSlice));
        HCCL_DEBUG("[%s] myRank[%u] TempLocalCopy end", __func__, myRank_);
    }

    HCCL_DEBUG("[%s] run success", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuTempScatterOmniPipeMesh1DMem2Mem::FastLaunch(const OpParam &param, const TemplateFastLaunchCtx& tempFastLaunchCtx)
{
    HCCL_DEBUG("[%s] end", __func__);
    return HcclResult::HCCL_SUCCESS;
}

u64 CcuTempScatterOmniPipeMesh1DMem2Mem::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    (void)inBuffType;
    (void)outBuffType;
    return templateRankSize_;
}

} // namespace ops_hccl