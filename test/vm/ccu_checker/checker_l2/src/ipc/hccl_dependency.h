#ifndef HCCL_DEPENDENCY_H
#define HCCL_DEPENDENCY_H



/// 依赖
uint16_t GetRankIdFromTaskCid(uint64_t cid);

/**
 * 获取当前的Rank ID。
 * 
 * @return 当前的Rank ID，类型为uint16_t。
 */
uint16_t GetCurrRankId();


/**
 * 获取当前的ServerId
 * 
 * @return 当前的ServerID，类型为uint16_t。
 */
uint16_t GetCurrServerId();


#endif