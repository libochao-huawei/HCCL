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
#include "hccp_common.h"
#include "hccp_ctx.h"
#include <fstream>

#include "ccu_microcode_v1.h"
#include "ccu_channel_ctx_mgr_v1.h"
#include "rt_external_kernel.h"
#include "ccu_common.h"
#include "sim_runner_common.h"
#include "ccu_jetty_ctx_mgr.h"
#include "hccl_vm_log.h"

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

// ����ccuӲ�������Ϣ��input->op == Hccl::CcuOpcodeType::CCU_U_OP_GET_BASIC_INFO
int SetCcuResourceBasicInfo(hcomm::CustomChannelInfoOut* output, uint8_t dieId, uint32_t devId)
{
    // todo: ��ȡccu�汾��Ϣ
    output->data.dataInfo.dataArray[0].baseinfo.resourceAddr = 0x123456789;
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

// ����΢��ָ�input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION
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
    HCCL_VM_INFO("Get Ccu {} of device {} instructions, count= {}, startId= {}, newCnt={}", dieId, devId, instrCnt, startId, ccuRes.instr_cnt);
    for (uint32_t i = 0; i < instrCnt; i++) {
        memcpy(ccuRes.instr_space[startId + i], &instrPtr[i], sizeof(hcomm::CcuRep::CcuInstr));
    }
    std::ostringstream fileName;
    fileName << "mc_instr_info_rank_" << devId<<"_die_"<<static_cast<uint32_t>(dieId)<<".txt";
    DumpInstrDecToFile(instrCnt, instrPtr, fileName.str());

    auto ccuResId = ccuRes.id;
    RunnerDB::Update<sim::CcuResource>(ccuResId, [ccuResId, ccuRes](sim::CcuResource &cr) {
        cr.instr_cnt = ccuRes.instr_cnt;
        memcpy(cr.instr_space, ccuRes.instr_space, sizeof(ccuRes.instr_space));
    });
    return 0;
}

// 配置channel信息：input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_CHANNEL
int ConfigChannelInfo(hcomm::CustomChannelInfoIn *input, uint32_t deviceId)
{
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
    // SimEid simEid;
    // std::copy(std::begin(eid.raw), std::end(eid.raw), simEid.begin());
    auto ipAddr = Hccl::IpAddress(eid).GetIpStr().substr(2);
    std::cout<<"zhf-[ConfigChannelInfo] get ip addr = "<<ipAddr<<std::endl;

    HCCL_VM_INFO("zhf-config channel..{}", ipAddr);
    if (ipAddr == "") {
        return 0;
        std::string rmtPortName;
        uint32_t rmtDeviceId = deviceId;
        uint8_t  rmtDieId = 0;
        if ((dieId == 0 && chId == 0) || (dieId == 1 && chId == 1)) {
            // die�ڻ��أ�channel 0��die0 -> die0 �� channel 1: die1 -> die1
            rmtDieId = dieId;
            rmtPortName = "0/0";
        } else {
            // rank��die�价�أ�channel 1��die0 -> die1 �� channel 0: die1 -> die0
            rmtDieId = (dieId == 0 ? 1 : 0);
            rmtPortName = "0/0";
        }
        // ����device id��ȡdevice key
        sim::Device device{};
        if (GetDeviceByLogicId(rmtDeviceId, device) != 0) {
            HCCL_VM_ERROR("[{}] can not find device by id:{:d}", __func__, deviceId);
            return -1;
        }
        // ����deviceKey + dieId �ҵ�ccu key
        sim::Ccu ccu{};
        if (GetCcuFromDeviceByDieId(device.id, dieId, ccu) != ACL_SUCCESS) {
            HCCL_VM_ERROR("[ConfigChannelInfo] get ccu from device by die id {} failed.", dieId);
            return -1;
        }
        // device id + ccu id + port name ---> port
        sim::Port dstPort{};
        if (GetPortFromSpecCcuByName(ccu.id, rmtPortName, dstPort) != 0) {
            HCCL_VM_ERROR("[ConfigChannelInfo]Get dst port failed.{}", rmtPortName);
            return -1;
        }
        // 3. ����dstPort��ȡEndPointPiar
        sim::EndPointPair endPointPair{};
        if (GetEndPointPairByDstPort(dstPort.id, endPointPair) != 0) {
            HCCL_VM_ERROR("[ConfigChannelInfo]Get end point pair failed.{}", rmtPortName);
            return -1;
        }
        // 4. ����channel��Ϣ
        sim::CcuChannel ccuChannel{};
        // ���ڳ�����src��dst endpointPiar��ͬһ��
        ccuChannel.end_point_pair_id = endPointPair.id;
        ccuChannel.channel_id = chId;
        ccuChannel.src_rank   = deviceId;
        ccuChannel.dst_rank   = device.logic_id;
        ccuChannel.src_die    = dieId;
        ccuChannel.dst_die    = ccu.die_id;
        std::cout << "[INFO][ConfigChannelInfo][Loop] Success: channelId= " << chId << ", srcRank= " << deviceId
                  << ", srcDie= " << static_cast<uint32_t>(dieId) << ", dstRank= " << device.logic_id
                  << ", dstDie= " << static_cast<uint32_t>(ccu.die_id) << std::endl;
        auto channelKey = RunnerDB::Add<sim::CcuChannel>(ccuChannel);
        return 0;
    }

    // 2. ����EID/IP��ȡ��ӦԶ��dstPort
    sim::Port dstPort{};
    if (GetPortByIpAddr(ipAddr, dstPort) != 0) {
        HCCL_VM_ERROR("[ConfigChannelInfo]Get dst port failed. ip= {}", ipAddr.c_str());
        return -1;
    }

    auto deviceT = RunnerDB::GetById<sim::Device>(dstPort.device_id);
    if (!deviceT.has_value()) {
        printf("[ERROR][DumpSimSynData] can not find end point pair by key:%lu\n", dstPort.device_id);
        return -1;
    }
    // todo: 出框port为"0/7"、"0/8"或"1/7"、"1/8"
    // 3. 根据dstPort获取EndPointPiar
    sim::EndPointPair endPointPair{};
    if (GetEndPointPairByDstPort(dstPort.id, endPointPair) != 0) {
        HCCL_VM_ERROR("[ConfigChannelInfo]Get end point pair failed.");
        return -1;
    }

    sim::Device device{};
    if (GetDeviceByPhysicalId(deviceId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("[LoadMicrocodeInstruction] get device by logic id {} failed.", deviceId);
        return -1;
    }

    // 3. 根据dstPort获取EndPointPiar
    sim::CcuChannel ccuChannel{};
    // 框内场景，src和dst endpointPiar是同一个
    ccuChannel.end_point_pair_id = endPointPair.id;
    ccuChannel.channel_id = chId;
    ccuChannel.src_rank   = device.logic_id;
    ccuChannel.dst_rank   = deviceT->logic_id;
    ccuChannel.src_die    = dieId;
    ccuChannel.dst_die    = dieId;
    std::cout << "[INFO][ConfigChannelInfo][NoLoop] Success:addr="<<ipAddr<<", channelId= " << chId << ", srcRank= " << deviceId
              << ", srcDie= " << static_cast<uint32_t>(dieId) << ", dstRank= " << deviceT->logic_id
              << ", dstDie= " << static_cast<uint32_t>(dieId) << std::endl;
    auto channelKey = RunnerDB::Add<sim::CcuChannel>(ccuChannel);

    return 0;
}

// ����channel��Ϣ��input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_JETTY_CTX
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

int ra_custom_channel(struct RaInfo info, struct custom_chan_info_in *in, struct custom_chan_info_out *out)
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

int ra_ctx_init(struct ctx_init_cfg *cfg, struct ctx_init_attr *attr, void **ctx_handle)
{
    // ����EID��ȡrdma handle
    // SimEid simEid;
    // std::copy(std::begin(attr->ub.eid.raw), std::end(attr->ub.eid.raw), simEid.begin());
    // hccp_eid ipEid;
    // for (uint32_t i = 0; i < Hccl::URMA_EID_LEN; i++) {
    //     ipEid.raw[i] = attr->ub.eid.raw[Hccl::URMA_EID_LEN - i - 1];
    // }
    // // auto ipAddr = Hccl::IpAddress(ipEid.in4.addr).GetIpStr();
    // auto ipAddr = Hccl::IpAddress(attr->ub.eid.in4.addr).GetIpStr();
    Hccl::Eid simEid;
    memcpy(simEid.raw, attr->ub.eid.raw, sizeof(simEid));
    auto ipAddr = Hccl::IpAddress(simEid).GetIpStr().substr(2);
    std::cout<<"zhf-[RaGetSockets] get ip addr = "<<ipAddr<<std::endl;
    HCCL_VM_INFO("[ra_ctx_init] Get ip addr {}", ipAddr);

    if (ipAddr == "0.0.0.0") {
        *ctx_handle = (void*)0x80000000;
        return 0;
    }

    sim::Port port{};
    if (GetPortByIpAddr(ipAddr, port) != 0) {
        HCCL_VM_ERROR("[ra_ctx_init]Get dst port failed.");
        return -1;
    }

    *ctx_handle = (void*)port.rdma_handle;
    return 0;
}

int ra_get_dev_base_attr(void *ctx_handle, struct dev_base_attr *attr)
{
    // ����ctx handle����ȡ��ӦdieId��funcId��Ϣ��ctx handle��EIDһһ��Ӧ��
    sim::Port port{};
    if (GetPortByCtxHandle(reinterpret_cast<uint64_t>(ctx_handle), port) != 0) {
        HCCL_VM_ERROR("[ra_get_dev_base_attr]Get port by ctx_handle failed.");
        return -1;
    }

    // ����ccu id��ȡccu
    auto ccu = RunnerDB::GetById<sim::Ccu>(port.ccu_id);
    if (!ccu.has_value()) {
        // not find
        HCCL_VM_ERROR("[{}] can not find ccu by key:{:d}", __func__, port.ccu_id);
        return -1;
    }

    attr->ub.die_id  = static_cast<uint32_t>(ccu->die_id);
    attr->ub.func_id = port.func_id;

    return 0;
}

int GetAllUsedPorts(uint32_t deviceId, std::vector<sim::Port> &allUsedPorts)
{
    // ����device id��ȡdevice key
    sim::Device device{};
    if (GetDeviceByLogicId(deviceId, device) != 0) {
        HCCL_VM_ERROR("[{}] can not find device by id:{:d}", __func__, deviceId);
        return -1;
    }
    // ����info.phy_id��Ӧdevice��ʹ�õ�IP��������ranktable��ʹ�õģ� �������� ����device id��ip��ַɸѡ
    auto deviceKey = device.id;
    HCCL_VM_INFO("zhf-Find all used port start: {}", deviceId);
    allUsedPorts = RunnerDB::GetByPred<sim::Port>([deviceKey](const sim::Port& port) {
        if ((port.device_id == deviceKey && port.status == 1)) {
            HCCL_VM_INFO("zhf-find all used port -- new: {}, {}, {}", port.device_id, deviceKey, port.ip_addr);
        }
        return (port.device_id == deviceKey && port.status == 1);
    });
    HCCL_VM_INFO("zhf-Find all used port end: {}", allUsedPorts.size());
    return 0;
}

int ra_get_dev_eid_info_num(struct RaInfo info, unsigned int *num)
{
    std::vector<sim::Port> allUsedPorts;
    if (GetAllUsedPorts(info.phyId, allUsedPorts) != 0) {
        return -1;
    }
    HCCL_VM_INFO("zhf-[ra_get_dev_eid_info_num] return success {:d}", allUsedPorts.size());
    *num = allUsedPorts.size();
    return 0;
}

int ra_get_dev_eid_info_list(struct RaInfo info, struct dev_eid_info info_list[], unsigned int *num)
{
    HCCL_VM_INFO("[ra_get_dev_eid_info_list] enter into ra_get_dev_eid_info_list");
    // ����info.phy_id��Ӧdevice��ʹ�õ�����IP��Ӧ��eid��Ϣ��func_id, chip_id, die_id�ȣ�
    std::vector<sim::Port> allUsedPorts;
    if (GetAllUsedPorts(info.phyId, allUsedPorts) != 0) {
        return -1;
    }

    for (uint32_t idx = 0; idx < *num; idx++) {
        info_list[idx].type = 0;
        info_list[idx].eid_index = 0;
        info_list[idx].func_id = allUsedPorts[idx].func_id;
        info_list[idx].chip_id = info.phyId; // todo: ��server, logic id��rank id��ȣ�����server�˴������⡣
        // ����ccu id��ȡccu
        auto ccu = RunnerDB::GetById<sim::Ccu>(allUsedPorts[idx].ccu_id);
        if (!ccu.has_value()) {
            // not find
            HCCL_VM_ERROR("[{}] can not find ccu by key:{:d}", __func__, allUsedPorts[idx].ccu_id);
            return -1;
        }
        info_list[idx].die_id = ccu->die_id;

        // ��IP��ַ�ַ���ת��ΪEID
        auto ipAddress = Hccl::IpAddress(allUsedPorts[idx].ip_addr);
        auto eid = ipAddress.GetEid();
        HCCL_VM_INFO("zhf- xxxx ip addr: ");
        for (uint32_t i = 8; i < 16; i++) {
            info_list[idx].eid.raw[i] = eid.raw[i];
            if (info.phyId == 0) {
                HCCL_VM_INFO("zhf-cxxx: {:d}", static_cast<int>(info_list[idx].eid.raw[i]));
            }
        }
    }

    return 0;
}

// ��ȡCCU SQE���ݣ���΢��ָ����Σ�
int rtCCULaunch(rtCcuTaskInfo_t *taskInfo, rtStream_t const stream)
{
    uint64_t streamId = (uint64_t)(uintptr_t)stream;

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::CCU_GRAPH;
    taskMetaData.commId   = 0;
    taskMetaData.rankId   = curRank;
    taskMetaData.streamId = streamId;
    memcpy(&taskMetaData.taskData.ccu, taskInfo, sizeof(rtCcuTaskInfo_t));
    std::cout<<"zhf-rank "<<curRank<<", dieId= "<<static_cast<uint32_t>(taskInfo->dieId)<<std::endl;
    std::cout << "[rtCCULaunch-1] Get sqe info: " << taskInfo->instStartId << ", cnt= " << taskInfo->instCnt
              << ", argSize= " << taskInfo->argSize << std::endl;
    std::cout << "[rtCCULaunch-2] Get sqe info: " << taskMetaData.taskData.ccu.instStartId
              << ", cnt= " << taskMetaData.taskData.ccu.instCnt << ", argSize= " << taskMetaData.taskData.ccu.argSize
              << std::endl;
    for (uint32_t idx = 0; idx < taskInfo->argSize; idx++) {
        printf("zhf-ccu mc args: %u: 0x%lx\n", idx, taskInfo->args[idx]);
    }
    std::ostringstream fileName;
    fileName << "sqe_info_rank_" << curRank << "_die_" << static_cast<uint32_t>(taskInfo->dieId) << "_mission_"
             << static_cast<uint32_t>(taskInfo->missionId) << "_startId_" << taskInfo->instStartId << ".txt";
    DumpCcuSqeToFile(taskInfo->instStartId, taskInfo->instCnt, taskInfo->argSize, taskInfo->args, fileName.str());

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