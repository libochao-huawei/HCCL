#ifndef HCCL_DEPENDENCY_H
#define HCCL_DEPENDENCY_H

#include <cstdint>
#include "hccl_dependency.h"
#include "hccl_common_defs.h"

uint16_t g_rankId;
uint16_t GetRankIdFromTaskCid(uint64_t cid)
{
    HcclTaskCid taskCid;
    taskCid.value = cid;
    return taskCid.field.rankId;
}

/**
 * 获取当前的Rank ID。
 * 
 * @return 当前的Rank ID，类型为uint16_t。
 */
uint16_t GetCurrRankId()
{
    return g_rankId; // deprecated
}


uint16_t GetCurrServerId()
{
    return g_rankId;    // deprecated
}


#endif