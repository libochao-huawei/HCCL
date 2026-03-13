/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <vector>
#include <atomic>
#include <iostream>
#include "sim_runner_ops.h"
#include <securec.h>
#include "hccp_ctx.h"
#include "hccp_common.h"
#include <fstream>
#include <filesystem>

#include "ccu_microcode_v1.h"
#include "ccu_channel_ctx_mgr_v1.h"
#include "ccu_dev_mgr_imp.h"
#include "rt_external_kernel.h"
#include "ccu_common.h"
#include "sim_runner_common.h"
#include "ccu_jetty_ctx_mgr.h"
#include "hccl_vm_log.h"

extern uint64_t g_cur_server_key;

using namespace HcclSim;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

int GetEnableCcuDie(hcomm::CustomChannelInfoOut *output, uint8_t dieId)
{
    const char* hcclvmTopoType = std::getenv("HCCLVM_TOPO_TYPE");
    if (hcclvmTopoType != nullptr && std::string(hcclvmTopoType) == "HF") {
        // HF需要使能两个DIE
        output->data.dataInfo.dataArray[0].dieinfo.enableFlag = 1;
        return 0;
    }

    // 多server场景，默认使能2Die
    if (sim::GetHostSize() > 1) {
        output->data.dataInfo.dataArray[0].dieinfo.enableFlag = 1;
        return 0;
    }

    const char* dieNum = std::getenv("HCCL_IODIE_NUM");
    if (dieNum != nullptr) {
        if (std::string(dieNum) == "2") {
            output->data.dataInfo.dataArray[0].dieinfo.enableFlag = 1;
            return 0;
        }
        HCCL_VM_INFO("[CustomChannelInfoOut]Env variable HCCL_IODIE_NUM=[{:d}]", dieNum);
    }
    output->data.dataInfo.dataArray[0].dieinfo.enableFlag = (dieId == 0) ? 1 : 0;
    return 0;
}

int SetCcuResourceBasicInfo(hcomm::CustomChannelInfoOut* output, uint8_t dieId, uint32_t devId)
{
    if (dieId == 0) {
        output->data.dataInfo.dataArray[0].baseinfo.resourceAddr = 0x123123123;
    } else {
        output->data.dataInfo.dataArray[0].baseinfo.resourceAddr = 0x456456456;
    }
    output->data.dataInfo.dataArray[0].baseinfo.missionKey = 0;
    output->data.dataInfo.dataArray[0].baseinfo.msId = 3;  //
    uint32_t instructionNum = 0x8000;                      // Instruction 32k
    uint32_t missionNum = 16;                              // Mission ctx 16
    uint32_t loopEngineNum = 200;                          // Loop ctx 200
    output->data.dataInfo.dataArray[0].baseinfo.caps.cap0 =
        (instructionNum - 1) | ((missionNum - 1) << MOVE_TOW_BYTES) | ((loopEngineNum - 1) << MOVE_THREE_BYTES);
    uint32_t gsaNum = 3072;     // GSA 3072
    uint32_t xnNum = 3072;      // Xn 3072
    output->data.dataInfo.dataArray[0].baseinfo.caps.cap1 = ((xnNum - 1) << MOVE_TOW_BYTES) | (gsaNum - 1);
    uint32_t ckeNum = 1024;     // Checlist Entry(CKE) 1024
    uint32_t msNum = 1536;      // MemorySlice(MS) 1536
    output->data.dataInfo.dataArray[0].baseinfo.caps.cap2 = ((msNum - 1) << MOVE_TOW_BYTES) | (ckeNum - 1);
    uint32_t channelNum = 128;  // Channel ӳ��� 128
    uint32_t jettyNum = 128;    // Jetty context 128
    output->data.dataInfo.dataArray[0].baseinfo.caps.cap3 = ((jettyNum - 1) << MOVE_TOW_BYTES) | (channelNum - 1);
    uint32_t pfeNum = 16;       // PFE���ñ� 16
    output->data.dataInfo.dataArray[0].baseinfo.caps.cap4 = (pfeNum - 1) & 0x000000FF;
    return 0;
}
extern void* GetRealPtrByAddr(const void *devPtr);

void DumpInstrDecToFile(uint32_t instrCnt, const hcomm::CcuRep::CcuInstr* instrData, const std::string &fileName)
{
    std::ofstream ofs(fileName, std::ios::out | std::ios::trunc);
    ofs <<"ccu total instruction number: "<<instrCnt <<"\n";
    for (uint32_t idx = 0; idx < instrCnt; idx++) {
        ofs << "[InstrData][ " + std::to_string(idx) + "]" + hcomm::CcuRep::ParseInstr(&instrData[idx]) + "\n";
    }
    return;
}

void DumpCcuSqeToFile(uint32_t startId, uint32_t instrCnt, uint32_t argSize, uint64_t args[], const std::string &fileName)
{
    std::ofstream ofs(fileName, std::ios::out | std::ios::trunc);
    ofs <<"ccu sqe info: startInstrId= "<<startId<<", instrCnt= "<<instrCnt<<", argSize= "<<argSize <<"\n";
    for (uint32_t idx = 0; idx < argSize; idx++) {
        ofs << "[SQE Arg][" << idx << "]: " << args[idx] << "\n";
    }
    return;
}

namespace fs = std::filesystem;

bool write_or_overwrite_in_cwd(const std::string& filename, const std::string &data) {
    fs::path target = fs::current_path() / filename;

    std::error_code ec;
    bool exists = fs::exists(target, ec);
    if (ec) {
        // 读取状态出错，视为失败
        return false;
    }

    std::ofstream ofs;
    if (exists) {
        ofs.open(target, std::ios::out | std::ios::app);
    } else {
        ofs.open(target, std::ios::out | std::ios::trunc);
    }

    if (!ofs.is_open()) {
        return false;
    }

    if (exists) {
        ofs << "\n\n";
    }

    ofs <<data;
    ofs.flush();
    return true;
}

// input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION
int LoadMicrocodeInstructionStub(uint32_t devId, uint8_t dieId, const hcomm::CustomChannelInfoIn *input)
{
    HCCL_VM_INFO("zhf-enter LoadMicrocodeInstruction .....");
    if (dieId >= DIE_NUM) {
        HCCL_VM_ERROR("[LoadMicrocodeInstruction] wrong param of die id: {}", dieId);
        return -1;
    }

    sim::Device device{};
    if (GetDeviceByPhysicalId(devId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("[LoadMicrocodeInstruction] get device by logic id {} failed.", devId);
        return -1;
    }
    sim::Ccu ccu{};
    if (GetCcuFromDeviceByDieId(device.id, dieId, ccu) != ACL_SUCCESS) {
        HCCL_VM_ERROR("[LoadMicrocodeInstruction] get ccu from device by die id {} failed.", dieId);
        return -1;
    }
    sim::CcuResource ccuRes;
    if (GetCcuResourceByCcu(ccu.id, ccuRes) != ACL_SUCCESS) {
        HCCL_VM_ERROR("[LoadMicrocodeInstruction] get ccu resource by ccu {} failed.", ccu.id);
        return -1;
    }

    auto devKey = device.id;
    auto rank = RunnerDB::GetOneByPred<sim::Rank>([devKey](const sim::Rank& r) {
        return r.device_id == devKey;
    });
    if (!rank.second) {
        HCCL_VM_ERROR("[LoadMicrocodeInstruction] can not find any rank");
        return -1;
    }
    auto rankId = rank.first.rank_id;

    auto ccuDataTmp = (hcomm::CcuDataTypeUnion)(input->data.dataInfo.dataArray[0]);
    auto instrPtr = reinterpret_cast<hcomm::CcuRep::CcuInstr*>(GetRealPtrByAddr((void *)ccuDataTmp.insinfo.resourceAddr));
    if (instrPtr == nullptr) {
        HCCL_VM_ERROR("[LoadMicrocodeInstruction] get ccu instrPtr by resourceAddr failed  addr:0x{:x}", ccuDataTmp.insinfo.resourceAddr);
        return -1;
    }
    auto startId  = input->offsetStartIdx;
    auto instrInfoSize = input->data.dataInfo.dataLen;
    auto instrCnt = instrInfoSize / sizeof(hcomm::CcuRep::CcuInstr);
    
    ccuRes.instr_cnt += instrCnt;
    HCCL_VM_INFO("Get Ccu {} of rank {} instructions, count= {}, startId= {}, newCnt={}", dieId, rankId, instrCnt, startId, ccuRes.instr_cnt);
    for (uint32_t i = 0; i < instrCnt; i++) {
        memcpy(ccuRes.instr_space[startId + i], &instrPtr[i], sizeof(hcomm::CcuRep::CcuInstr));
    }
    std::ostringstream fileName;
    fileName << "mc_instr_info_rank_" << rankId<<"_die_"<<static_cast<uint32_t>(dieId)<<".txt";
    std::ostringstream mcData;
    mcData <<"ccu total instruction number: "<<instrCnt <<"\n";
    for (uint32_t idx = 0; idx < instrCnt; idx++) {
        mcData << "[InstrData][ " + std::to_string(startId + idx) + "]" + hcomm::CcuRep::ParseInstr(&instrPtr[idx]) + "\n";
    }
    auto status = write_or_overwrite_in_cwd(fileName.str(), mcData.str());

    auto ccuResId = ccuRes.id;
    RunnerDB::Update<sim::CcuResource>(ccuResId, [ccuResId, ccuRes](sim::CcuResource &cr) {
        cr.instr_cnt = ccuRes.instr_cnt;
        cr.state = 1;
        memcpy(cr.instr_space, ccuRes.instr_space, sizeof(ccuRes.instr_space));
    });
    return 0;
}

uint64_t GetRemoteCcuVa(const hcomm::ChannelCtxDataV1 &chDataTmp)
{
    uint64_t dstVa = 0;
    dstVa |= (uint64_t)(chDataTmp.dstVaLow & hcomm::MASK_VA_LOW);           // 低 8 位
    dstVa |= ((uint64_t)(chDataTmp.dstVaMiddle & hcomm::MASK_VA_MID) << hcomm::SHIFT_8BITS);   // 位 8-23
    dstVa |= ((uint64_t)(chDataTmp.dstVaHigh & hcomm::MASK_VA_HIGH) << hcomm::SHIFT_24BITS);  // 位 24-39
    dstVa |= ((uint64_t)(chDataTmp.dstVaHigher & hcomm::MASK_VA_HIGHER) << hcomm::SHIFT_40BITS);  // 位 40+

    return dstVa << hcomm::REMOTE_CCU_VA_RIGHT_SHIFT_NUM;
}

// 配置channel信息：input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_CHANNEL
int ConfigChannelInfo(hcomm::CustomChannelInfoIn *input, uint32_t deviceId)
{
    sim::Device locDevice{};
    if (GetDeviceByPhysicalId(deviceId, locDevice) != ACL_SUCCESS) {
        HCCL_VM_ERROR("[ConfigChannelInfo] get device by physic id {} failed.", deviceId);
        return -1;
    }
    uint8_t dieId = input->data.dataInfo.udieIdx;
    uint32_t chId = input->offsetStartIdx;

    // 配置channel信息：input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_CHANNEL
    hcomm::ChannelCtxDataV1 chDataTmp;
    (void)memcpy(&chDataTmp, input->data.dataInfo.dataArray,
        sizeof(struct hcomm::ChannelCtxDataV1));
    Hccl::Eid eid;
    for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
        eid.raw[i] = chDataTmp.eidRaw[URMA_EID_LEN - i - 1];
    }
    auto ipAddr = Hccl::IpAddress(eid).GetIpStr().substr(2);

    // 环回链路
    if (ipAddr == "") {
        return 0;
    }

    sim::Port dstPort{};
    if (GetPortByIpAddr(ipAddr, dstPort) != 0) {
        HCCL_VM_ERROR("[ConfigChannelInfo]Get dst port failed. ip= {}", ipAddr.c_str());
        return -1;
    }

    auto rmtDevice = RunnerDB::GetById<sim::Device>(dstPort.device_id);
    if (!rmtDevice.has_value()) {
        printf("[ERROR][ConfigChannelInfo] can not find end point pair by key:%lu\n", dstPort.device_id);
        return -1;
    }

    auto rmtCcuVa = GetRemoteCcuVa(chDataTmp);

    HCCL_VM_INFO(
        "[ConfigChannelInfo] channel info: loc phyId: {:d}, loc devKey: {:d}, loc dieId: {:d}, chId: {:d}, rmt devKey: {:d}, rmt dieId: {:d}, rmt ip: {}, rmt ccu va: {:x}",
        deviceId,
        locDevice.id,
        static_cast<uint32_t>(dieId),
        chId,
        dstPort.device_id,
        static_cast<uint32_t>(chDataTmp.ioDieId),
        ipAddr,
        rmtCcuVa);

    // 3. 根据dstPort获取EndPointPiar
    sim::EndPointPair endPointPair{};
    if (GetEndPointPairByDstPort(dstPort.id, endPointPair) != 0) {
        HCCL_VM_ERROR("[ConfigChannelInfo]Get end point pair failed.");
        return -1;
    }

    auto locDevKey = locDevice.id;
    auto locRank = RunnerDB::GetOneByPred<sim::Rank>([locDevKey](const sim::Rank& r) {
        return r.device_id == locDevKey;
    });
    if (!locRank.second) {
        HCCL_VM_ERROR("[ConfigChannelInfo] can not find loc rank by device key: {}", locDevKey);
        return -1;
    }

    auto rmtDevKey = rmtDevice->id;
    auto rmtRank = RunnerDB::GetOneByPred<sim::Rank>([rmtDevKey](const sim::Rank& r) {
        return r.device_id == rmtDevKey;
    });
    if (!rmtRank.second) {
        HCCL_VM_ERROR("[ConfigChannelInfo] can not find rmt rank by device key: {}", rmtDevKey);
        return -1;
    }

    // 3. 根据dstPort获取EndPointPiar
    sim::CcuChannel ccuChannel{};
    // 框内场景，src和dst endpointPiar是同一个
    ccuChannel.end_point_pair_id = endPointPair.id;
    ccuChannel.channel_id = chId;
    ccuChannel.src_rank   = locRank.first.rank_id;
    ccuChannel.dst_rank   = rmtRank.first.rank_id;
    ccuChannel.src_die    = dieId;
    if (rmtCcuVa == 0x123000000) {
        ccuChannel.dst_die    = 0;
    } else {
        ccuChannel.dst_die    = 1;
    }
    
    
    HCCL_VM_INFO(
        "[ConfigChannelInfo] channel table entry: loc phyId: {:d}, loc rank: {:d}, loc dieId: {:d}, chId: {:d}, rmt rank: {:d}, rmt dieId: {:d}, rmt ip: {}",
        deviceId,
        locDevice.logic_id,
        static_cast<uint32_t>(dieId),
        chId,
        rmtDevice->logic_id,
        static_cast<uint32_t>(ccuChannel.dst_die),
        ipAddr);
    auto channelKey = RunnerDB::Add<sim::CcuChannel>(ccuChannel);

    return 0;
}

int ConfigJettyInfo(hcomm::CustomChannelInfoIn *input, uint32_t deviceId)
{
    HCCL_VM_INFO("[ConfigJettyInfo] Enter into config jetty info...");
    uint8_t dieId      = input->data.dataInfo.udieIdx;
    uint32_t jettyNum  = input->data.dataInfo.dataArraySize;
    uint32_t startJettyCtxId = input->offsetStartIdx;

    std::vector<hcomm::LocalJettyCtxData> jettyCtxData;
    jettyCtxData.resize(jettyNum);
    for (size_t i = 0; i < jettyNum; i++) {
        (void)memcpy(&jettyCtxData[i],
            &input->data.dataInfo.dataArray[i], sizeof(hcomm::LocalJettyCtxData));
    }

    for (auto &tmp : jettyCtxData) {
        HCCL_VM_DEBUG("[{}] doorbellAddr: [3]0x{:04x}, [2]0x{:04x}, [1]0x{:04x}, [0]0x{:04x}", __func__,
            tmp.doorbellAddr[3],  // 3: doorbell ��ַ����
            tmp.doorbellAddr[2],  // 2: doorbell ��ַ����
            tmp.doorbellAddr[1],
            tmp.doorbellAddr[0]);

        // ��ȫ���⣺��ֹ��ӡtoken�����Ϣ
        HCCL_VM_DEBUG("[{}] pfeIdx: 0x{:04x}, ioDieId: 0x{:04x}, doorbellAddrType: 0x{:04x}, tokenValueIsValid: 0x{:04x}", __func__,
            static_cast<uint16_t>(tmp.pfeIdx),
            static_cast<uint16_t>(tmp.ioDieId),
            static_cast<uint16_t>(tmp.doorbellAddrType),
            static_cast<uint16_t>(tmp.tokenValueIsValid));

        HCCL_VM_DEBUG("[{}] sqeBasicBlockLeftShifts: 0x{:04x}, pi: 0x{:04x}, ci: 0x{:04x}, "
            "maxCi: 0x{:04x}, oooCqeCnt: 0x{:04x}, startWqeBasicBlockIdxLow: 0x{:04x}, "
            "startWqeBasicBlockIdxHigh: 0x{:04x}, doorbellSendState: 0x{:04x}", __func__,
            static_cast<uint16_t>(tmp.sqeBasicBlockLeftShifts),
            tmp.pi,
            tmp.ci,
            tmp.maxCi,
            static_cast<uint16_t>(tmp.oooCqeCnt),
            static_cast<uint16_t>(tmp.startWqeBasicBlockIdxLow),
            static_cast<uint16_t>(tmp.startWqeBasicBlockIdxHigh),
            static_cast<uint16_t>(tmp.doorbellSendState));
    }

    return 0;
}

int RaCustomChannel(struct RaInfo info, struct CustomChanInfoIn *in, struct CustomChanInfoOut *out)
{
    hcomm::CustomChannelInfoIn *input = reinterpret_cast<hcomm::CustomChannelInfoIn *>(in);
    hcomm::CustomChannelInfoOut *output = reinterpret_cast<hcomm::CustomChannelInfoOut *>(out);
    uint8_t  dieId = input->data.dataInfo.udieIdx;
    uint32_t devId = info.phyId;

    switch (input->op) {
        case hcomm::CcuOpcodeType::CCU_U_OP_GET_DIE_WORKING:
            return GetEnableCcuDie(output, dieId);
        case hcomm::CcuOpcodeType::CCU_U_OP_GET_BASIC_INFO:
            return SetCcuResourceBasicInfo(output, dieId, devId);
        case hcomm::CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION:
            return LoadMicrocodeInstructionStub(devId, dieId, input);
        case hcomm::CcuOpcodeType::CCU_U_OP_SET_CHANNEL:
            return ConfigChannelInfo(input, devId);
        case hcomm::CcuOpcodeType::CCU_U_OP_SET_JETTY_CTX:
            return ConfigJettyInfo(input, devId);
        default:
            break;
    }

    return 0;
}

int RaCtxInit(struct CtxInitCfg *cfg, struct CtxInitAttr *attr, void **ctxHandle)
{
    Hccl::Eid simEid;
    memcpy(simEid.raw, attr->ub.eid.raw, sizeof(simEid));
    auto ipAddr = Hccl::IpAddress(simEid).GetIpStr().substr(2);
    std::cout<<"zhf-[RaGetSockets] get ip addr = "<<ipAddr<<std::endl;
    HCCL_VM_INFO("[RaCtxInit] Get ip addr {}", ipAddr);

    if (ipAddr == "0.0.0.0") {
        *ctxHandle = (void*)0x80000000;
        return 0;
    }

    sim::Port port{};
    if (GetPortByIpAddr(ipAddr, port) != 0) {
        HCCL_VM_ERROR("[RaCtxInit]Get dst port failed.");
        return -1;
    }

    *ctxHandle = (void*)port.rdma_handle;
    return 0;
}

int RaCtxGetAsyncEvents(void *ctxHandle, struct AsyncEvent events[], unsigned int *num)
{
    HCCL_VM_ERROR("[RaCtxGetAsyncEvents]Not support for now.");
    return -1;
}

int RaGetEidByIp(void *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num)
{
    HCCL_VM_ERROR("[RaGetEidByIp]Not support for now.");
    return -1;
}

int RaCtxDeinit(void *ctxHandle)
{
    return 0;
}

int RaGetDevBaseAttr(void *ctxHandle, struct DevBaseAttr *attr)
{
    sim::Port port{};
    if (GetPortByCtxHandle(reinterpret_cast<uint64_t>(ctxHandle), port) != 0) {
        HCCL_VM_ERROR("[RaGetDevBaseAttr]Get port by ctxHandle failed.");
        return -1;
    }

    auto ccu = RunnerDB::GetById<sim::Ccu>(port.ccu_id);
    if (!ccu.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not find ccu by key:{:d}", __func__, port.ccu_id);
        return -1;
    }

    attr->ub.dieId  = static_cast<uint32_t>(ccu->die_id);
    attr->ub.funcId = port.func_id;

    return 0;
}

int GetAllUsedPorts(uint32_t phyDevId, std::vector<sim::Port> &allUsedPorts)
{
    sim::Device device{};
    if (GetDeviceByPhysicalId(phyDevId, device) != 0) {
        HCCL_VM_ERROR("[{}] can not find device by id:{:d}", __func__, phyDevId);
        return -1;
    }

    auto deviceKey = device.id;
    HCCL_VM_INFO("zhf-Find all used port start: {}", phyDevId);
    allUsedPorts = RunnerDB::GetByPred<sim::Port>([deviceKey, phyDevId](const sim::Port& port) {
        if ((port.device_id == deviceKey && port.status == 1)) {
            HCCL_VM_INFO("zhf-find all used port -- new: phyDevId: {:d}, devKey: {:d}, ip: {}, serKey: {:d}", phyDevId, deviceKey, port.ip_addr, g_cur_server_key);
        }
        return (port.device_id == deviceKey && port.status == 1);
    });
    HCCL_VM_INFO("zhf-Find all used port end: {}", allUsedPorts.size());
    return 0;
}

int RaGetDevEidInfoNum(struct RaInfo info, unsigned int *num)
{
    std::vector<sim::Port> allUsedPorts;
    if (GetAllUsedPorts(info.phyId, allUsedPorts) != 0) {
        return -1;
    }
    HCCL_VM_INFO("zhf-[ra_get_dev_eid_info_num] return success {:d}", allUsedPorts.size());
    *num = allUsedPorts.size();
    return 0;
}

int RaGetDevEidInfoList(struct RaInfo info, struct HccpDevEidInfo infoList[], unsigned int *num)
{
    HCCL_VM_INFO("[RaGetDevEidInfoList] enter into RaGetDevEidInfoList: {:d}", info.phyId);
    std::vector<sim::Port> allUsedPorts;
    if (GetAllUsedPorts(info.phyId, allUsedPorts) != 0) {
        return -1;
    }

    std::cout<<"[RaGetDevEidInfoList] Get port num: "<<allUsedPorts.size()<<std::endl;
    for (uint32_t idx = 0; idx < *num; idx++) {
        infoList[idx].type = 0;
        infoList[idx].eidIndex = 0;
        infoList[idx].funcId = allUsedPorts[idx].func_id;
        infoList[idx].chipId = info.phyId; // todo: 单server, logic id与rank id相等，但多server此处有问题。

        auto ccu = RunnerDB::GetById<sim::Ccu>(allUsedPorts[idx].ccu_id);
        if (!ccu.has_value()) {
            // not find
            HCCL_VM_ERROR("[{}] can not find ccu by key:{:d}", __func__, allUsedPorts[idx].ccu_id);
            return -1;
        }
        infoList[idx].dieId = ccu->die_id;

        auto ipAddress = Hccl::IpAddress(allUsedPorts[idx].ip_addr);
        auto eid = ipAddress.GetEid();
        HCCL_VM_INFO("zhf- xxxx ip addr: ip= {}, funcId= {:d}, dieId= {:d}", ipAddress.GetIpStr(), infoList[idx].funcId, static_cast<uint32_t>(ccu->die_id));
        for (uint32_t i = 8; i < 16; i++) {
            infoList[idx].eid.raw[i] = eid.raw[i];
            if (info.phyId == 0) {
                HCCL_VM_INFO("zhf-cxxx: {:d}", static_cast<int>(infoList[idx].eid.raw[i]));
            }
        }
    }

    return 0;
}

int rtCCULaunch(rtCcuTaskInfo_t *taskInfo, rtStream_t const stream)
{
    sleep(2);
    uint64_t streamId = sim::GetCurrentStreamId((uint64_t)(uintptr_t)stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::CCU_GRAPH;
    taskMetaData.commId   = 0;
    taskMetaData.streamId = streamId;
    taskMetaData.rankId   = curRank;

    memcpy(&taskMetaData.taskData.ccu, taskInfo, sizeof(rtCcuTaskInfo_t));

    HCCL_VM_INFO("[{}]Get rank {:d} die {:d} streamId {:d} mission {:d} Sqe info, startId= {:d}, cnt= {:d}, argSize= {:d}",
        __func__,
        curRank,
        static_cast<uint32_t>(taskInfo->dieId),
        streamId,
        static_cast<uint32_t>(taskInfo->missionId),
        taskInfo->instStartId,
        taskInfo->instCnt,
        taskInfo->argSize);
    for (uint32_t idx = 0; idx < taskInfo->argSize; idx++) {
        HCCL_VM_INFO("[{}]sqe arg {:d}: {:x}", __func__, idx, taskInfo->args[idx]);
    }
    std::ostringstream fileName;
    fileName << "sqe_info_rank_" << curRank << "_die_" << static_cast<uint32_t>(taskInfo->dieId) << "_mission_"
             << static_cast<uint32_t>(taskInfo->missionId) << "_startId_" << taskInfo->instStartId << ".txt";
    std::ostringstream sqeData;
    sqeData <<"ccu sqe info: startInstrId= "<<taskInfo->instStartId<<", instrCnt= "<< taskInfo->instCnt<<", argSize= "<<taskInfo->argSize <<"\n";
    for (uint32_t idx = 0; idx < taskInfo->argSize; idx++) {
        sqeData << "[SQE Arg][" << idx << "]: " << taskInfo->args[idx] << "\n";
    }
    auto status = write_or_overwrite_in_cwd(fileName.str(), sqeData.str());

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[{}] InsertTaskToCollection fail", __func__);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // �·�cid
    HcclTaskCid taskCid{0, curRank, index};
    sim::Task task{};
    task.stream_id  = streamId;
    task.cid        = taskCid.value;
    task.type       = (uint8_t)HccLTaskMetaType::CCU_GRAPH;

    auto taskId = RunnerDB::Add<sim::Task>(task);

    // �·�cid
    //TaskVentilator::GetInstance().AddTaskCid(streamId, taskCid);
    // ��¼״̬
    //TaskStatusCache::GetInstance().AddTaskCid(streamId, taskCid);
    return 0;
}

#ifdef __cplusplus
}
#endif  // __cplusplus