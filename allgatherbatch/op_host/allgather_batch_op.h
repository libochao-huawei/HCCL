#ifndef HCCL_ALLGATHERBATCH_OP_HOST_H
#define HCCL_ALLGATHERBATCH_OP_HOST_H

#include "allgather_batch.h"
#include "common.h"

namespace ops_hccl_allgatherbatch {

class AllGatherBatchOp {
public:
    HcclResult Exec(const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream);

private:
    HcclResult Validate(const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, aclrtStream stream) const;
    HcclResult PrepareTopoInfo(HcclComm comm, BatchTopoInfo &topoInfo) const;
    HcclResult PrepareOpParam(const HcclAllGatherItem *items, uint32_t itemCount, HcclComm comm, OpParam &param) const;
    HcclResult GetAlgRes(HcclComm comm, const OpParam &param, AlgResourceCtx **resCtx) const;
    HcclResult LoadAndLaunch(const OpParam &param, aclrtStream stream) const;
};

}  // namespace ops_hccl_allgatherbatch

#endif
