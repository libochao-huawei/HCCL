#ifndef OPS_HCCL_DOUBLE_ALLGATHER_HCCL_COMPAT_H
#define OPS_HCCL_DOUBLE_ALLGATHER_HCCL_COMPAT_H

#if __has_include("hccl/hccl.h")
#include "hccl/hccl.h"
#elif __has_include("hccl.h")
#include "hccl.h"
#else
#error "Cannot find hccl.h"
#endif

#if __has_include("hccl/hccl_types.h")
#include "hccl/hccl_types.h"
#elif __has_include("hccl_types.h")
#include "hccl_types.h"
#else
#error "Cannot find hccl_types.h"
#endif

#if __has_include("hccl/hccl_res.h")
#include "hccl/hccl_res.h"
#elif __has_include("hccl_res.h")
#include "hccl_res.h"
#else
#error "Cannot find hccl_res.h"
#endif

#if __has_include("hccl/hcomm_primitives.h")
#include "hccl/hcomm_primitives.h"
#elif __has_include("hcomm_primitives.h")
#include "hcomm_primitives.h"
#else
#error "Cannot find hcomm_primitives.h"
#endif

#endif
