/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: 1.0适配器对外提供的编排接口
 * Author: huangweihao
 * Create: 2025-05-06
 */

#include "alg_adapter_v1_interface.h"
#include "orchestrate.h"
#include "link_type_recorder.h"
#include "device_info_recorder.h"
#include "mem_layout.h"
#include "externalinput.h"
#include "env_config.h"

using namespace checker;
namespace hccl {

TaskQuesGenerator::~TaskQuesGenerator()
{
    AllTransport_.clear();
    CreatedLinksDict_.clear();
    links2TransportCompare_.clear();
    DeviceInfoRecorder::Global()->Reset();
}

HcclResult TaskQuesGenerator::Run(CheckerOpParam &checkerOpParam, TopoMeta &topoMeta)
{
    hccl::RankTable_t rankTable;
    GenRankTable(rankTable, topoMeta);

    // 混合组网场景
    if (checkerOpParam.devTypes.size() != 0) {
        for (auto &rankInfo : rankTable.rankList) {
            rankInfo.deviceInfo.deviceType = g_CheckerDevType2HcclDevType[checkerOpParam.devTypes[rankInfo.superPodIdx]];
        }
    }
    // 为本次执行初始化状态
    u32 rankNum = rankTable.rankNum;
    LinkTypeRecorder::Global()->SetIs310P3V(checkerOpParam.is310P3V);
    if (checkerOpParam.devTypes.size() == 0) {
        checkerOpParam.devTypes.push_back(checkerOpParam.devtype);
    }
    LinkTypeRecorder::Global()->SetLinkTypeMap(checkerOpParam.devTypes);
    RankInfoRecorder::Global()->InitRankInfo(topoMeta, checkerOpParam.devtype);
    DeviceInfoRecorder::Global()->InitDeviceInfo(topoMeta, rankTable, checkerOpParam.devtype);
    MemLayout::Global()->Init(checkerOpParam);

    // 根据环境变量初始化g_externalInput
    CHK_RET(InitExternalInput());
    CHK_RET(InitEnvConfig());
    bool isIOSameAddr = (g_CheckerOpType2HcclCMDType[checkerOpParam.opType] == HcclCMDType::HCCL_CMD_BROADCAST);
    // 申请内存资源
    vector<shared_ptr<hccl::HcclCommunicator>> communicators;
    CHK_RET(OrchestraTask(checkerOpParam, rankTable, rankNum, false, communicators, isIOSameAddr));

    // transportLink匹配
    CHK_RET(CheckTransportLink());
    // 实际跑代码
    HcclResult ret = OrchestraTask(checkerOpParam, rankTable, rankNum, true, communicators, isIOSameAddr);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[Checker][RunTask]errNo[0x%016llx] run task failed", HCCL_ERROR_CODE(ret));
    }

    // 清理g_externalInput相关的配置
    CHK_RET(ResetInitState());
    // 清理g_envConfig环境变量
    CHK_RET(ResetEnvConfigInitState());
    CHK_RET(ResetAlgEnvConfigInitState());
    return ret;
}

}