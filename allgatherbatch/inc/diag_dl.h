#ifndef HCCL_ALLGATHERBATCH_DIAG_DL_H
#define HCCL_ALLGATHERBATCH_DIAG_DL_H

#include <cstddef>

#include "hccl/hccl_types.h"

namespace ops_hccl_allgatherbatch {

typedef void (*HcommGetOpInfoCallback)(const void *opInfo, char *outPut, size_t size);

extern HcclResult (*hcommRegOpInfoPtr)(const char *, void *, size_t);
extern HcclResult (*hcommRegOpTaskExceptionPtr)(const char *, HcommGetOpInfoCallback);

#define HcommRegOpInfo (*ops_hccl_allgatherbatch::hcommRegOpInfoPtr)
#define HcommRegOpTaskException (*ops_hccl_allgatherbatch::hcommRegOpTaskExceptionPtr)

bool HcommIsSupportHcommRegOpInfo(void);
bool HcommIsSupportHcommRegOpTaskException(void);

void AllGatherBatchDiagDlInit(void *libHandle);
void AllGatherBatchDiagDlFini(void);

}  // namespace ops_hccl_allgatherbatch

#endif
