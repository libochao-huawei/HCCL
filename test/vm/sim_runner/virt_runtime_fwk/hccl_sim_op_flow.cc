/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccl sim interface
 */
#include <dlfcn.h>
#include "hccl_sim_op_flow.h"
#include "SimRunnerMgr.h"
#include "rts_stub.h"
#include "hccl.h"
#include "MC2_type_stub.h"
#include "hccl_ex.h"

using namespace std;
typedef HcclResult (*HcclGetCcuTaskInfoFunc)(HcclComm comm, void *fusionArgs, void *ccuTaskGroup);
HcclResult CcuMc2TilingTest(HcclComm &comm, void* &stream, OpType opType, HcclReduceOp ReduceOp, HcclDataType DataType)
{
    void *commContext = nullptr;
    rtCcuTaskGroup_t ccuTaskGroup;
    rtFusionArgsEx_t fusionArgs;
    fusionArgs.args = malloc(sizeof(void*) * 7 + sizeof(Mc2Tiling) + sizeof(sim_runner::HcclCommParamDesc));
    void* args = fusionArgs.args; 
    Mc2Tiling* tilingData = (Mc2Tiling*)((uint8_t *)args + sizeof(void*) * 7);

    *reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t *>(fusionArgs.args)+sizeof(void*) * 5) = reinterpret_cast<uint64_t>(tilingData);
    *reinterpret_cast<uint64_t*>(fusionArgs.args) = 2;
    fusionArgs.aicpuNum = 1;
    fusionArgs.aicpuArgs[0].kfcArgsFmtOffset = (sizeof(void*) * 7 + sizeof(Mc2Tiling)) / sizeof(void*);

    sim_runner::HcclCommParamDesc* commParamDesc = reinterpret_cast<sim_runner::HcclCommParamDesc*>(reinterpret_cast<uint8_t*>(fusionArgs.args) + sizeof(void*) * 7 + sizeof(Mc2Tiling));
    commParamDesc->groupNum = 1;
    commParamDesc->hasFfts = 0;
    commParamDesc->tilingDataPtrOff = 5;

    tilingData->version = 3;
    tilingData->commConfigNum = 1;
    tilingData->serverCfg = {0};
    tilingData->commConfig.opType = static_cast<uint32_t>(OP_TO_MC2_MAP.at(opType));
    tilingData->commConfig.reduceType = ReduceOp;
    tilingData->commConfig.dataType = DataType;
    tilingData->commConfig.outputDataType = DataType;

    stream = (void *)1;
    HcclResult ret = HcclAllocComResourceByTiling(comm, stream, tilingData, &commContext);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Start][MC2_test] failed to execute the HCCL MC2 HcclAllocComResourceByTiling operator, ret[%d].", ret);
        free(args);
        return ret;
    }
    std::unique_ptr<sim_runner::HcclCombinOpParam> contextData = std::make_unique<sim_runner::HcclCombinOpParam>();
    rtError_t rt_ret = memcpy_s(contextData.get(), sizeof(sim_runner::HcclCombinOpParam), commContext, sizeof(sim_runner::HcclCombinOpParam));
    if (ret != 0) {
        HCCL_ERROR("[Start][MC2_test] failed to memcpy_s commContext to contextData, ret[%d].", ret);
        free(args);
        return HCCL_E_MEMORY;
    }
    // 还有一个commContext出参的检查
   void* handle = dlopen("libhccl_v2.so", RTLD_LAZY);
    if (!handle) {
        printf("无法加载库: %s\n", dlerror());
        free(args);
        return HCCL_E_SYSCALL;
    }

    // 获取函数地址
    HcclGetCcuTaskInfoFunc HcclGetCcuTaskInfo = (HcclGetCcuTaskInfoFunc)dlsym(handle, "HcclGetCcuTaskInfo");
    if (!HcclGetCcuTaskInfo) {
        printf("无法找到函数: %s\n", dlerror());
        dlclose(handle);
        free(args);
        return HCCL_E_SYSCALL;
    }
    ret = HcclGetCcuTaskInfo(comm, (void *)&fusionArgs, (void *)&ccuTaskGroup);
    // 处理返回值
    if (ret != HCCL_SUCCESS) {
        std::cerr << "函数调用失败，返回值: " << ret << std::endl;
    } else {
        std::cout << "函数调用成功" << std::endl;
    }
    // 释放资源
    dlclose(handle);

    rtStreamCreateWithFlags(&stream, 0, 0);
    rt_ret = rtCCULaunch(&ccuTaskGroup.ccuTaskInfo[0], stream);
    free(args);
    return ret;
}

void InternalProcess(SimParams *params)
{
    try {
        // 1. 设置要使用的device
        aclrtSetDevice(params->myRank);
        if (!PrepareSimParams(params)) {
            HCCL_ERROR("[Start][InternalProcess] PrepareSimParams is error.");
            return;
        }
        std::cout << "子进程准备send/recv Buf结束" << std::endl;

        // 2. 创建通讯域并初始化
        void *comm;
        const char *filePath = "./ranktable.json";
        int rank = params->myRank;
        HcclResult ret = HcclCommInitClusterInfo(filePath, rank, &comm);
        if (ret != HCCL_SUCCESS || comm == nullptr) {
            HCCL_ERROR("[Start][HcclCommInitClusterInfo] init comm failed, rank[%d], ret[%d].", rank, ret);
            return;
        }

        // 3. 创建流
        rtStream_t stream = nullptr;
        // 4. 准备入参并执行集合通信
        OpType opType = params->situation.GetOpType();
        auto count = params->situation.GetCount();
        int root = params->situation.GetRoot();
        HcclDataType dataType = params->situation.GetDataType();
        HcclReduceOp reduceOp = params->situation.GetReduceOp();
        if (params->mc2Flag) {
            ret = CcuMc2TilingTest(comm, stream, opType, reduceOp, dataType);
        } else {
            rtStreamCreateWithFlags(&stream, 0, 0);
            switch (opType) {
                case OpType::ALLREDUCE:
                    ret = HcclAllReduce(params->sendBuf, params->recvBuf, count, dataType, reduceOp, comm, stream);
                    break;
                case OpType::REDUCE:
                    ret = HcclReduce(params->sendBuf, params->recvBuf, count, dataType, reduceOp, root, comm, stream);
                    break;
                case OpType::BROADCAST:
                    ret = HcclBroadcast(params->sendBuf, count, dataType, root, comm, stream);
                    break;
                case OpType::ALLGATHER:
                    ret = HcclAllGather(params->sendBuf, params->recvBuf, params->sendCount, dataType, comm, stream);
                    break;
                case OpType::ALLTOALL:
                    ret = HcclAlltoAll(params->sendBuf, params->sendCount, dataType,
                                        params->recvBuf, params->recvCount, dataType,
                                        comm, stream);
                    break;
                case OpType::REDUCESCATTER:
                    ret = HcclReduceScatter(
                        params->sendBuf, params->recvBuf, params->recvCount, dataType, reduceOp, comm, stream);
                    break;
                default:
                    break;
            }
        }

        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[Start][InternalProcess] failed to execute the HCCL operator, ret[%d].", ret);
            return;
        }
        // 5.执行流同步(按rank触发)
        rtStreamSynchronize(stream);
        // 6.销毁通信域
        ret = HcclCommDestroy(comm);
        if (ret != HCCL_SUCCESS) {
            std::cout << "销毁通信域失败" << std::endl;
            return;
        }

        // 7.结果校验并更新共享内存数据
        bool res = VerifySimResult(params);
        SimRunnerMgr::GetInstance().GetShmPoolMgr()->SetVerifyResult(rank, res);
    } catch (const std::exception &e) {
        HCCL_ERROR("[Start][InternalProcess] some exception occurs and was catched by test fwk:[%s].", e.what());
    }
}