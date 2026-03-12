#include "sim_context_manager.h"

namespace HcclProxy {

SimContextMgr::~SimContextMgr()
{
    for (auto& tagMap : contextMap_) {
        for (auto& pair : tagMap.second) {
            if (pair.second.addr != nullptr) {
                free(pair.second.addr);
            }
        }
    }
    contextMap_.clear();
    tagMap_.clear();
    engineMap_.clear();
}

HcclResult SimContextMgr::CreateCommEngineCtx(const std::string &tag, CommEngine engine, HcclMem *engineCtx)
{
    // CHK_PTR_NULL(engineCtx);
    if (engineCtx == nullptr) {
        return HCCL_E_PTR;
    }
    
    // 阻止重复创建
    if (contextMap_.find(tag) != contextMap_.end()) {
        auto engineCtxMap = contextMap_[tag];
        if (engineCtxMap.find(engine) == engineCtxMap.end()) {
            printf("[%s] already exist context with same key, tag[%s], engine[%d]", __func__, tag.c_str(), engine);
        }
        return HCCL_E_PARA;
        // CHK_PRT_RET(engineCtxMap.find(engine) != engineCtxMap.end(),
        //     HCCL_ERROR("[%s] already exist context with same key, tag[%s], engine[%d]", __func__, tag.c_str(), engine),
        //     HCCL_E_PARA);
    }

    void* ctxMem = nullptr;
    // 暂时只支持HOST与AICPU模式
    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS) {
        engineCtx->type = HCCL_MEM_TYPE_HOST;
        ctxMem = malloc(engineCtx->size);
        // CHK_PTR_NULL(ctxMem);
        // CHK_SAFETY_FUNC_RET(memset_s(ctxMem, engineCtx->size, 0, engineCtx->size));
        if (ctxMem == nullptr) {
            return HCCL_E_PTR;
        }
        memset_s(ctxMem, engineCtx->size, 0, engineCtx->size);
    } else if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        engineCtx->type = HCCL_MEM_TYPE_DEVICE;
        ctxMem = malloc(engineCtx->size);   // AICPU模式需要使用Device侧内存，但当前先用进程私有内存来实现
        // CHK_PTR_NULL(ctxMem);
        // CHK_SAFETY_FUNC_RET(memset_s(ctxMem, engineCtx->size, 0, engineCtx->size));
        if (ctxMem == nullptr) {
            return HCCL_E_PTR;
        }
        memset_s(ctxMem, engineCtx->size, 0, engineCtx->size);
    } else {
        // HCCL_ERROR("[%s] not support engine type[%d]", __func__, engine);
        printf("[ERROR] [%s] not support engine type[%d]", __func__, engine);
        return HCCL_E_PARA;
    }

    contextMap_[tag][engine] = {engineCtx->type, ctxMem, engineCtx->size};
    tagMap_[contextMap_[tag][engine]] = tag;
    engineMap_[contextMap_[tag][engine]] = engine;
    *engineCtx = contextMap_[tag][engine];

    // HCCL_INFO("[%s] create context success, tag[%s], engine[%d]", __func__, tag.c_str(), engine);
    printf("[INFO] [%s] create context success, tag[%s], engine[%d]", __func__, tag.c_str(), engine);
    return HCCL_SUCCESS;
}

HcclResult SimContextMgr::GetCommEngineCtx(const std::string &tag, CommEngine engine, HcclMem *engineCtx)
{
    // Ctx未创建返回
    if (contextMap_.find(tag) == contextMap_.end()) {
        // HCCL_INFO("[%s] not exist a context with tag[%s]", __func__, tag.c_str());
        printf("[INFO] [%s] not exist a context with tag[%s]", __func__, tag.c_str());
        return HCCL_E_PARA;
    } else {
        auto engineCtxMap = contextMap_[tag];
        if (engineCtxMap.find(engine) == engineCtxMap.end()) {
            // HCCL_INFO("[%s] not exist a context with tag[%s], engine[%d]", __func__, tag.c_str(), engine);
            printf("[INFO] [%s] not exist a context with tag[%s], engine[%d]", __func__, tag.c_str(), engine);
            return HCCL_E_PARA;
        }
    }

    *engineCtx = contextMap_[tag][engine];
    // HCCL_INFO("[%s] get context success, tag[%s], engine[%d]", __func__, tag.c_str(), engine);    
    printf("[INFO] [%s] get context success, tag[%s], engine[%d]", __func__, tag.c_str(), engine);
    return HCCL_SUCCESS;
}

HcclResult SimContextMgr::CopyCommEngineCtx(const std::string &tag, CommEngine engine, const void *srcCtx,
    uint64_t size, uint64_t dstCtxOffset)
{
    // Ctx未创建返回
    if (contextMap_.find(tag) == contextMap_.end()) {
        // HCCL_INFO("[%s] not exist a context with tag[%s]", __func__, tag.c_str());
        printf("[INFO] [%s] not exist a context with tag[%s]", __func__, tag.c_str());
        return HCCL_E_PARA;
    } else {
        auto engineCtxMap = contextMap_[tag];
        if (engineCtxMap.find(engine) == engineCtxMap.end()) {
            // HCCL_INFO("[%s] not exist a context with tag[%s], engine[%d]", __func__, tag.c_str(), engine);
            printf("[INFO] [%s] not exist a context with tag[%s], engine[%d]", __func__, tag.c_str(), engine);
            return HCCL_E_PARA;
        }
    }
    HcclMem dstCtx = contextMap_[tag][engine];
    // 暂时只支持HOST与AICPU模式
    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CPU_TS ||
        engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        (void)memcpy_s(reinterpret_cast<uint8_t*>(dstCtx.addr) + dstCtxOffset, size, srcCtx, size);
    } else {
        // HCCL_ERROR("[%s] not support engine type[%d]", __func__, engine);
        printf("[ERROR] [%s] not support engine type[%d]", __func__, engine);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult SimContextMgr::DestroyCommEngineCtx(const HcclMem *engineCtx)
{
    // CHK_PTR_NULL(engineCtx);
    if (engineCtx == nullptr) {
        return HCCL_E_PTR;
    }

    // Ctx不存在返回错误
    if (tagMap_.find(*engineCtx) == tagMap_.end()) {
        // HCCL_ERROR("[%s]The provided engineCtx does not exist.", __func__);
        printf("[ERROR] [%s]The provided engineCtx does not exist.", __func__);
        return HCCL_E_PARA;
    }

    // Ctx存在进行销毁
    contextMap_[tagMap_[*engineCtx]].erase(engineMap_[*engineCtx]);
    if (contextMap_[tagMap_[*engineCtx]].empty()) {
        contextMap_.erase(tagMap_[*engineCtx]);
    }
    tagMap_.erase(*engineCtx);
    engineMap_.erase(*engineCtx);

    free(engineCtx->addr); // 暂时只支持进程私有内存

    // HCCL_INFO("[%s] destroy context success", __func__);
    printf("[INFO] [%s] destroy context success", __func__);
    return HCCL_SUCCESS;
}

}