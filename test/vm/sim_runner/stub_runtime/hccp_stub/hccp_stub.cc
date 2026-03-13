/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * Description: hccp stub
 */

#include "fake_socket.h"
#include <iostream>
#include "hccl_sim_pub_stub.h"
#include "ccu_device_manager.h"
#include "ip_address.h"
#include "hccp_ctx.h"
#include "ccu_channel_mgr.h"
#include "ccu_channel_mgr_v1.h"
#include "SimRunnerMgr.h"

#ifdef __cplusplus
extern "C" {
constexpr uint32_t MOVE_TOW_BYTES   = 16;
constexpr uint32_t MOVE_THREE_BYTES = 24;
constexpr uint32_t SOCKET_VNIC_IP_INFOS_INTERFACE = 55;
constexpr uint32_t GET_NOTIFY_BA = 14;
constexpr uint32_t MOVE_20_BITS = 20;
constexpr uint32_t MOVE_16_BITS = 16;

int ra_socket_init(int mode, struct rdev rdev_info, void **socket_handle)
{
    *socket_handle = SimRunnerMgr::GetInstance().GetFakeSocket()->GetSocketHandle();
    return 0;
}

int ra_socket_listen_start(struct socket_listen_info_t conn[], unsigned int num)
{
    return 0;
}

int ra_socket_listen_start_async(struct socket_listen_info_t conn[], unsigned int num, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_socket_deinit(void *socket_handle)
{
    return 0;
}

int ra_get_sockets(unsigned int role, struct socket_info_t conn[], unsigned int num, unsigned int *connected_num)
{
    // 只打桩num=1的情况
    auto ret = SimRunnerMgr::GetInstance().GetFakeSocket()->Get(role, conn[0]);

    if(!ret) {
        return 1;
    }
    *connected_num = num;
    return 0;
}

int ra_socket_white_list_del(void *socket_handle, struct socket_wlist_info_t white_list[], unsigned int num)
{
    return 0;
}

int ra_socket_send(const void *fd_handle, const void *data, unsigned long long size, unsigned long long *sent_size)
{
    auto ret = SimRunnerMgr::GetInstance().GetFakeSocket()->Send((int *)fd_handle, data, size, sent_size);
    if(!ret) {
        return 1;
    }
    return 0;
}

int ra_socket_recv(const void *fd_handle, void *data, unsigned long long size, unsigned long long *received_size)
{
    auto ret = SimRunnerMgr::GetInstance().GetFakeSocket()->Recv((int *)fd_handle, data, size, received_size);

    //可能先发起收，需要重试多次

    int retryCnt = 0;
    while(retryCnt++ <= 20) {
        if (!ret) {
            sleep(1);
            ret = SimRunnerMgr::GetInstance().GetFakeSocket()->Recv((int *)fd_handle, data, size, received_size);
        } else {
            return 0;
        }
    }

    if(!ret) {
        return 1;
    }
    return 0;
}

int ra_socket_get_white_list_status(unsigned int *enable)
{
    return 0;
}

int ra_socket_batch_close(struct socket_close_info_t conn[], unsigned int num)
{
    return 0;
}

int ra_socket_batch_close_async(struct socket_close_info_t conn[], unsigned int num, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_socket_batch_connect(struct socket_connect_info_t conn[], unsigned int num)
{
    // TODO 当前只打桩num=1情况
    if (num != 1) {
        return 1;
    }

    auto ret = SimRunnerMgr::GetInstance().GetFakeSocket()->Connect(conn[0]);

    if(!ret) {
        return 1;
    }

    return 0;
}

int ra_socket_batch_connect_async(struct socket_connect_info_t conn[], unsigned int num, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return ra_socket_batch_connect(conn, num);
}

int ra_socket_listen_stop(struct socket_listen_info_t conn[], unsigned int num)
{
    return 0;
}

int ra_socket_listen_stop_async(struct socket_listen_info_t conn[], unsigned int num, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_socket_white_list_add(void *socket_handle, struct socket_wlist_info_t white_list[], unsigned int num)
{
    return 0;
}

int ra_init(struct ra_init_config *config)
{
    return 0;
}

int ra_deinit(struct ra_init_config *config)
{
    return 0;
}

int ra_socket_set_white_list_status(unsigned int enable)
{
    return 0;
}

int ra_rdev_init(int mode, unsigned int notify_type, struct rdev rdev_info, void **rdma_handle)
{
    *rdma_handle = SimRunnerMgr::GetInstance().GetNetDeviceMgr()->GetRdmaHandle(rdev_info.local_ip.addr.s_addr);
    return 0;
}

int ra_get_qp_status(void *qp_handle, int *status)
{
    return 0;
}

int ra_get_ifnum(struct ra_get_ifattr *config, unsigned int *num)
{
    return 0;
}

int ra_get_ifaddrs(struct ra_get_ifattr *config, struct interface_info interface_infos[], unsigned int *num)
{
    return 0;
}

int ra_qp_connect_async(void *qp_handle, const void *fd_handle)
{
    return 0;
}

int ra_qp_destroy(void *qp_handle)
{
    return 0;
}

int ra_qp_create(void *rdev_handle, int flag, int qp_mode, void **qp_handle)
{
    return 0;
}

int ra_rdev_deinit(void *rdma_handle, unsigned int notify_type)
{
    return 0;
}

int ra_mr_reg(void *qp_handle, struct mr_info *info)
{
    return 0;
}

int ra_mr_dereg(void *qp_handle, struct mr_info *info)
{
    return 0;
}

int ra_get_notify_base_addr(void *rdev_handle, unsigned long long *va, unsigned long long *size)
{
    return 0;
}

int ra_socket_init_v1(int mode, struct socket_init_info_t socket_init, void **socket_handle)
{
    return 0;
}

int ra_send_wr(void *qp_handle, struct send_wr *wr, struct send_wr_rsp *op_rsp)
{
    return 0;
}

int ra_ctx_init(struct ctx_init_cfg *cfg, struct ctx_init_attr *attr, void **ctx_handle)
{
    auto netDeviceMgr = SimRunnerMgr::GetInstance().GetNetDeviceMgr();
    // 在fakeUb对象中打桩了一个32位数组，每次Init返回不同的地址，作为CCU key值。
    auto ipAddr = Hccl::IpAddress(attr->ub.eid.in4.addr);
    *ctx_handle = netDeviceMgr->GetRdmaHandleNew(attr->ub.eid.in4.addr);
    HCCL_INFO("ra_ctx_init *ctx_handle[%p] ipAddr[%s]", *ctx_handle, ipAddr.GetIpStr().c_str());
    netDeviceMgr->SetHandle2AddrMap(*ctx_handle, attr->ub.eid.in4.addr);  // 将ctx_handle和ipAddr做一个映射，用于后续的查找
    return 0;
}

int ra_ctx_deinit(void *ctx_handle)
{
    return 0;
}

int ra_ctx_qp_create(void *ctx_handle, struct qp_create_attr *attr, struct qp_create_info *info, void **qp_handle)
{
    return 0;
}

int ra_ctx_qp_bind(void *qp_handle, void *rem_qp_handle)
{
    return 0;
}

int ra_ctx_qp_unimport(void *ctx_handle, void *rem_qp_handle)
{
    return 0;
}

int ra_ctx_rmem_import(void *ctx_handle, struct mr_import_info_t *rmem_info, void **rmem_handle)
{
    return 0;
}

int ra_get_dev_base_attr(void *ctx_handle, struct dev_base_attr *attr)
{
    auto netDeviceMgr = SimRunnerMgr::GetInstance().GetNetDeviceMgr();
    NetDeviceInfo rmtDevice;
    auto ret = netDeviceMgr->GetRmtDevInfo(ctx_handle, rmtDevice);
    if (!ret) {
        std::cout<<"[ra_get_dev_base_attr][ERROR] Get remote device info failed. ctx_handle="<<std::hex<<ctx_handle<<std::endl;
        return -1;
    }
    //将id信息赋值到RdmaHandleManager中
    attr->ub.die_id = rmtDevice.dieId;
    attr->ub.func_id = rmtDevice.funcId;
    return 0;
}

int ra_ctx_cq_create(void *ctx_handle, struct cq_info_t *info, void **cq_handle)
{
    return 0;
}

int ra_ctx_cq_destroy(void *ctx_handle, void *cq_handle)
{
    return 0;
}

int ra_ctx_update_ci(void *qp_handle, uint16_t ci)
{
    return 0;
}

int ra_get_dev_eid_info_num(struct ra_info info, unsigned int *num)
{
    *num = SimRunnerMgr::GetInstance().GetNetDeviceMgr()->GetIpNum(info.phy_id);  // 由die_info.json文件解析出来的ip个数赋值给num
    return 0;
}

int ra_get_dev_eid_info_list(struct ra_info info, struct dev_eid_info info_list[], unsigned int *num)
{
    auto dieInfoMap = SimRunnerMgr::GetInstance().GetNetDeviceMgr()->GetDieInfoMap(info.phy_id);  //由die_info.json文件解析出的map信息
    int count = 0;
    for (auto it = dieInfoMap.begin(); it != dieInfoMap.end(); it++, count++) {
        // 映射表信息得到对应的EidInfo
        info_list[count].type = 0;
        info_list[count].eid_index = 0;
        info_list[count].func_id = it->second.funcId;
        info_list[count].chip_id = it->second.rankId;
        info_list[count].die_id = it->second.dieId;
    }
    return 0;
}

int ra_ctx_qp_unbind(void *qp_handle)
{
    return 0;
}

int ra_ctx_qp_destroy(void *qp_handle)
{
    return 0;
}

void LaunchCcuInstrs(int devId, int dieId, Hccl::CustomChannelInfoIn &instr)
{
    auto fakeStreamMgr = SimRunnerMgr::GetInstance().GetFakeStreamMgr();
    if (fakeStreamMgr != nullptr) {
        fakeStreamMgr->SaveInstr(dieId, instr);
    }
}

uint64_t GetResourceAddr(uint8_t devid, uint16_t dieId)
{
    uint64_t baseAddr = (dieId == 0) ? 0x000008e00000000 : 0x000008e40000000;
    baseAddr += devid * 0x10000000000;
    std::cout << "GetResourceAddr devid " << (uint32_t)devid << " dieId " << (uint32_t)dieId << " baseAddr " << baseAddr << std::endl;
    return baseAddr;
}

void SetCcuResourceBasicInfo(Hccl::CustomChannelInfoOut* output, uint8_t dieId, uint16_t devId)
{
    auto &simRunnerMgr = SimRunnerMgr::GetInstance();
    if (simRunnerMgr.GetCcuVersionFlag() == CcuVersion::CCU_V1) {
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
        uint32_t channelNum = 128;  // Channel 映射表 128
        uint32_t jettyNum = 128;    // Jetty context 128
        output->data.dataInfo.dataArray[0].baseinfo.caps.cap3 = ((jettyNum - 1) << MOVE_TOW_BYTES) | (channelNum - 1);
        uint32_t pfeNum = 16;       // PFE配置表 16
        output->data.dataInfo.dataArray[0].baseinfo.caps.cap4 = (pfeNum - 1) & 0x000000FF;
    } else if (simRunnerMgr.GetCcuVersionFlag() == CcuVersion::CCU_V2) {
        if (simRunnerMgr.GetCaModelFlag() == 1) {
            output->data.dataInfo.dataArray[0].baseinfo.resourceAddr = GetResourceAddr(devId, dieId);
        } else {
            output->data.dataInfo.dataArray[0].baseinfo.resourceAddr = 0x123456789;
        }
        uint32_t instructionNum = 0x8000;                      // Instruction 32k
        uint32_t missionNum = 16;                              // Mission ctx 16
        uint32_t loopEngineNum = 512;                          // Loop ctx 512 for V121
        output->data.dataInfo.dataArray[0].baseinfo.caps.cap0 = 
            ((instructionNum - 1) & 0x0000FFFF)           // 低 16 位 (instructionNum)
            | (((missionNum - 1) & 0x0000000F) << MOVE_16_BITS)      // 中间 4 位 (missionNum)
            | (((loopEngineNum - 1) & 0x00000FFF) << MOVE_20_BITS);  // 高 12 位 (loopEngineNum)
        output->data.dataInfo.dataArray[0].baseinfo.missionKey = 0;
        output->data.dataInfo.dataArray[0].baseinfo.msId = 3;
        uint32_t gsaNum = 0;     // GSA 0 for V121
        uint32_t xnNum = 16384;      // Xn 16384 for V121
        output->data.dataInfo.dataArray[0].baseinfo.caps.cap1 = ((xnNum - 1) << MOVE_TOW_BYTES) | (gsaNum - 1);
        uint32_t ckeNum = 1024;     // Checlist Entry(CKE) 1024
        uint32_t msNum = 1536;      // MemorySlice(MS) 1536
        output->data.dataInfo.dataArray[0].baseinfo.caps.cap2 = ((msNum - 1) << MOVE_TOW_BYTES) | (ckeNum - 1);
        uint32_t channelNum = 1024;  // Channel 映射表 1024 for V121
        uint32_t jettyNum = 128;    // Jetty context 128
        output->data.dataInfo.dataArray[0].baseinfo.caps.cap3 = ((jettyNum - 1) << MOVE_TOW_BYTES) | (channelNum - 1);
        uint32_t pfeNum = 20;       // PFE配置表 20 for V121
        output->data.dataInfo.dataArray[0].baseinfo.caps.cap4 = (pfeNum - 1) & 0x000000FF;
    }
}

std::mutex hccpMutex;
int ra_custom_channel(struct ra_info info, struct custom_chan_info_in *in, struct custom_chan_info_out *out)
{
    lock_guard<mutex> lock(hccpMutex);
    auto &simRunnerMgr = SimRunnerMgr::GetInstance();
    Hccl::CustomChannelInfoIn *input = reinterpret_cast<Hccl::CustomChannelInfoIn *>(in);
    Hccl::CustomChannelInfoOut *output = reinterpret_cast<Hccl::CustomChannelInfoOut *>(out);
    uint8_t dieId = input->data.dataInfo.udieIdx;
    uint8_t devId = info.phy_id;
    //对2D场景下的enableFlag做一个适配

    /*
     * 双die场景：die0，die1均使能；
     * 单die场景：仅die0使能
     * 1D绕路场景：仅die0使能
     * 2D绕路场景：hccl业务代码暂不支持
     */
    if (input->op == Hccl::CcuOpcodeType::CCU_U_OP_GET_DIE_WORKING) {
        const char* dieNum = std::getenv("HCCL_IODIE_NUM");
        if (dieNum != nullptr) {
            if (std::string(dieNum) == "2") {
                output->data.dataInfo.dataArray[0].dieinfo.enableFlag = 1;
                return 0;
            }
            HCCL_INFO("[ra_custom_channel]环境变量值HCCL_IODIE_NUM=[%s]", dieNum);
        }
        output->data.dataInfo.dataArray[0].dieinfo.enableFlag = (dieId == 0) ? 1 : 0;  // 单die场景，只让die0可用
    } else if (input->op == Hccl::CcuOpcodeType::CCU_U_OP_GET_BASIC_INFO) {
        SetCcuResourceBasicInfo(output, dieId, devId);
    }
    
    if (simRunnerMgr.GetCcuFeatureFlag()) {
        auto chanInfo = (Hccl::CustomChannelInfoIn*)in;
        if (chanInfo->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION) {
            auto ccuDataTmp = (Hccl::CcuDataTypeUnion)(chanInfo->data.dataInfo.dataArray[0]);
            LaunchCcuInstrs(info.phy_id, chanInfo->data.dataInfo.udieIdx, *chanInfo);
        } else if (chanInfo->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_CHANNEL) {
            Hccl::ChannelDataV1 chDataTmp;
            (void)memcpy_s(&chDataTmp, sizeof(struct Hccl::ChannelDataV1), chanInfo->data.dataInfo.dataArray,
                   sizeof(struct Hccl::ChannelDataV1));
            hccp_eid eid;
            for (uint32_t i = 0; i < Hccl::URMA_EID_LEN; i++) {
                eid.raw[i] = chDataTmp.eidRaw[Hccl::URMA_EID_LEN - i - 1];
            }
            auto ipAddr = Hccl::IpAddress(eid.in4.addr);
            simRunnerMgr.GetNetDeviceMgr()->SaveEid(chanInfo->data.dataInfo.udieIdx, chanInfo->offsetStartIdx, eid.in4.addr, ipAddr.GetIpStr());
        }
    }

    return 0;
}

int ra_ctx_lmem_unregister(void *ctx_handle, void *lmem_handle)
{
    return 0;
}

int ra_ctx_lmem_register(void *ctx_handle, struct mr_reg_info_t *lmem_info, void **lmem_handle)
{
    return 0;
}

int ra_ctx_rmem_unimport(void *ctx_handle, void *rmem_handle)
{
    return 0;
}

int ra_ctx_qp_import(void *ctx_handle, struct qp_import_info_t *qp_info, void **rem_qp_handle)
{
    return 0;
}

int ra_ctx_qp_unimport_async(void *rem_qp_handle, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_ctx_lmem_unregister_async(void *ctx_handle, void *lmem_handle, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_socket_send_async(const void *fd_handle, const void *data, unsigned long long size,
    unsigned long long *sent_size, void **req_handle)
{
    ra_socket_send(fd_handle, data, size, sent_size);
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_socket_recv_async(const void *fd_handle, void *data, unsigned long long size,
    unsigned long long *received_size, void **req_handle)
{
    ra_socket_recv(fd_handle, data, size, received_size);
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_ctx_qp_create_async(void *ctx_handle, struct qp_create_attr *attr, struct qp_create_info *info, void **qp_handle,
    void **req_handle)
{
    auto &simRunnerMgr = SimRunnerMgr::GetInstance();
    *qp_handle = simRunnerMgr.GetFakeUb()->GetQpHandle();
    *req_handle = reinterpret_cast<void *>(0x12345678);
    if (!simRunnerMgr.IsAicpuSim() && !simRunnerMgr.GetCcuFeatureFlag()) {
        return 0;
    }

    if (simRunnerMgr.GetCcuFeatureFlag()) {  // ccu模式下，jetty Id需要赋值
        info->ub.id = attr->ub.jetty_id;
        return 0;
    }

    ShmPoolLock shmPoolLock;
    ShmPub *shmPub = simRunnerMgr.GetShmPoolMgr()->GetShmPub();
    if ((shmPub == nullptr) || (shmPub->socket.jettyIdGen >= MAX_STREAM_NUM)) {
        std::cout << "ra_ctx_qp_create_async invalid jettyIdGen!" << std::endl;
        return -1;
    }

    int jettyId = shmPub->socket.jettyIdGen++;
    shmPub->socket.piValJettyIdMap[jettyId] = 0;
    shmPub->socket.sqVaJettyIdMap[jettyId] = attr->ub.sq.buff_va;  // 绑定jettyId和sqVa地址
    info->ub.id = jettyId;                                // 对device侧jettyId打桩

    return 0;
}

int ra_ctx_qp_destroy_async(void *qp_handle, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_get_async_req_result(void *req_handle, int *req_result)
{
    return 0;
}

int ra_ctx_token_id_alloc(void *ctx_handle, struct hccp_token_id *info, void **token_id_handle)
{
    return 0;
}

int ra_ctx_token_id_free(void *ctx_handle, void *token_id_handle)
{
    return 0;
}

int ra_ctx_qp_import_async(void *ctx_handle, struct qp_import_info_t *info, void **rem_qp_handle, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_ctx_lmem_register_async(void *ctx_handle, struct mr_reg_info_t *lmem_info, void **lmem_handle,
    void **req_handle)
{
    *lmem_handle = reinterpret_cast<void *>(lmem_info->in.mem.addr);
    *req_handle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int ra_batch_send_wr(void *qp_handle, struct send_wr_data wr_list[], struct send_wr_resp op_resp[],
                     unsigned int num, unsigned int *complete_num)
{
    // 将send_wr_data[转化为wqe]存入本端qp_handle的SQ
    // 构造出参，避免出参校验失败
    *complete_num = num;  // 暂时认为都是1个wr
    op_resp->doorbell_info.dieId = 0;
    op_resp->doorbell_info.rsv = 0;
    op_resp->doorbell_info.dwqe_size = 64;
    int *jettfId = reinterpret_cast<int *>(qp_handle);
    op_resp->doorbell_info.jettyId = *jettfId;

    // 构造FakeWqe，存入UB队列，返回对应下标赋值给funcId
    FakeWqe wqe_data;
    if (wr_list->ub.opcode == RA_UB_OPC_WRITE) {
        // UbMemTransport::SubmitNotify->DevUbConnection::PrepareInlineWrite 写Notify
        wqe_data.type = WqeType::WRITE;
        wqe_data.notifyAddr = wr_list->remote_addr;
    } else if (wr_list->ub.opcode == RA_UB_OPC_WRITE_NOTIFY) {
        // UbMemTransport::SubmitWriteWithNotify->DevUbConnection::PrepareWriteWithNotify
        bool isReduce = wr_list->ub.reduce_info.reduce_en;
        wqe_data.type = isReduce ? WqeType::REDUCE_WITH_NOTIFY : WqeType::WRITE_WITH_NOTIFY;
        wqe_data.notifyAddr = wr_list->ub.notify_info.notify_addr;
        wqe_data.localAddr = wr_list->sges[0].addr;
        wqe_data.size = wr_list->sges[0].len;
        wqe_data.remoteAddr = wr_list->remote_addr;

        if (isReduce) {
            wqe_data.reduceOpType = wr_list->ub.reduce_info.reduce_opcode;
            wqe_data.reduceDataType = wr_list->ub.reduce_info.reduce_data_type;
        }
    } else {
        std::cout << "[ra_batch_send_wr]wr_list->ub.opcode not support: " << static_cast<int>(wr_list->ub.opcode) << std::endl;
    }
    op_resp->doorbell_info.funcId = SimRunnerMgr::GetInstance().GetFakeUb()->PushWqe(wqe_data);

    return 0;
}

int ra_get_tp_info_list_async(void *ctx_handle, struct get_tp_cfg *cfg, struct tp_info info_list[],
    unsigned int *num, void **req_handle)
{
    *req_handle = reinterpret_cast<void *>(0x12345678);
    *num = 1;
    return 0;
}

int ra_get_qp_context(void* qpHandle, void** qp, void** sendCq, void** recvCq)
{
    return 0;
}

int ra_get_tsqp_depth(void *rdev_handle, unsigned int *temp_depth, unsigned int *qp_num)
{
    *temp_depth = 1;
    *qp_num = 1;
    return 0;
}

int ra_set_tsqp_depth(void *rdev_handle, unsigned int temp_depth, unsigned int *qp_num)
{
    return 0;
}

int ra_get_notify_mr_info(void* handle, struct mr_info *mrInfo)
{
    return 0;
}

int ra_send_wrlist(void *handle, struct send_wrlist_data wr[], struct send_wr_rsp op_rsp[], unsigned int send_num, unsigned int *complete_num)
{
    HCCL_INFO("ra_send_wrlist send_num[%u]", send_num);
    if ((handle == NULL) || (wr == NULL) || (op_rsp == NULL)) {
        HCCL_ERROR("invalid parameters, handle[%p], wr[%p], op_rsp[%p]",
            handle, wr, op_rsp);
        return -1;
    }
    s32 ret = 0;
    struct send_wr wr_tmp = {0};

    for (u32 j = 0; j < send_num; j++) {
        wr_tmp.buf_list = &(wr[j].mem_list);
        wr_tmp.buf_num = 1; /* 此处list只有一个，设置为1 */
        wr_tmp.dst_addr = (u64)(uintptr_t)(wr[j].dst_addr);
        wr_tmp.op = wr[j].op; /* RDMA_WRITE: 0 */
        wr_tmp.send_flag = wr[j].send_flags;
        HCCL_INFO("[cc] ra_send_wrlist dst[%0x] local[%0x] sendNum[%u] len[%u]", wr_tmp.dst_addr, wr[j].mem_list.addr, send_num, wr[j].mem_list.len);
        ret = ra_send_wr(handle, &wr_tmp, op_rsp);
        if (ret != 0) {
            *complete_num = j;
            HCCL_ERROR("ra_send_wr failed, idx[%u] ret[%d]", j, ret);
            return ret;
        }
        op_rsp++;
    }
    *complete_num = send_num;
    return 0;
}

int ra_send_wrlist_ext(void *qp_handle, struct send_wrlist_data_ext wr[], struct send_wr_rsp op_rsp[],
    unsigned int send_num, unsigned int *complete_num)
{
    return 0;
}

int ra_register_mr(const void* handle, struct mr_info *mrInfo, void **mrHandle)
{
    *mrHandle = (void *)0xabcd;
    return ((handle == NULL) || (mrInfo == NULL)) ? -1 :0;
}

int ra_deregister_mr(const void* handle, void *mrHandle)
{
    return ((handle == NULL) || (mrHandle == NULL)) ? -1 :0;
}

int ra_is_first_used(int ins_id)
{
    return 0;
}

int ra_is_last_used(int ins_id)
{
    return 0;
}

int ra_rdev_init_v2(struct rdev_init_info init_info, struct rdev rdev_info, void **rdma_handle)
{
    return 0;
}

int ra_get_interface_version(unsigned int phy_id, unsigned int interface_opcode, unsigned int* interface_version)
{
    if (interface_opcode == SOCKET_VNIC_IP_INFOS_INTERFACE) {
        *interface_version = 0;
    } else if (interface_opcode == GET_NOTIFY_BA) {
        *interface_version = 2;
    } else {
        *interface_version = 1;
    }
    return DRV_ERROR_NONE;
}

int ra_epoll_ctl_add(const void *fd_handle, RaEpollEvent event)
{
    return 0;
}

int ra_epoll_ctl_mod(const void *fd_handle, RaEpollEvent event)
{
    return 0;
}

int ra_epoll_ctl_del(const void *fd_handle)
{
    return 0;
}

int ra_socket_get_vnic_ip_infos(unsigned int phy_id, enum id_type type, unsigned int* ids, unsigned int num, struct ip_info *infos)
{
    return 0;
}

int ra_rdev_get_support_lite(void *rdma_handle, int *support_lite)
{
    return 0;
}

int ra_cq_create(void *rdev_handle, struct cq_attr *attr)
{
    return 0;
}

int ra_cq_destroy(void *rdev_handle, struct cq_attr *attr)
{
    return 0;
}

int ra_normal_qp_destroy(void *qp_handle)
{
    if(qp_handle == nullptr)
    {
        return HCCL_E_PTR;
    }
    return 0;
}

int ra_set_qp_attr_qos(void *qpHandle, struct qos_attr *attr)
{
    return 0;
}

int ra_set_qp_attr_timeout(void *qpHandle, u32 *timeout)
{
    return 0;
}

int ra_set_qp_attr_retry_cnt(void *qpHandle, u32 *retry_cnt)
{
    return 0;
}

int ra_get_cqe_err_info(unsigned int phy_id, struct cqe_err_info *info)
{
    return 0;
}

int ra_create_srq(const void*, struct srq_attr *)
{
    return 0;
}

int ra_destroy_srq(const void*, struct srq_attr *)
{
    return 0;
}
int ra_create_event_handle(int *event_handle)
{
    return 0;
}

int ra_ctl_event_handle(int event_handle, const void *fd_handle, int opcode, enum RaEpollEvent event)
{
    return 0;
}

int ra_wait_event_handle(int event_handle, struct socket_event_info *event_infos, int timeout, unsigned int maxevents,
    unsigned int *events_num)
{
    return 0;
}

int ra_destroy_event_handle(int *event_handle)
{
    return 0;
}

int ra_normal_qp_create(void *rdev_handle, struct ibv_qp_init_attr *qp_init_attr, void **qp_handle, void** qp)
{
    return 0;
}

int ra_create_comp_channel(const void *rdma_handle, void **comp_channel)
{
    *comp_channel = (void *)0xabcd;
    return ((rdma_handle == NULL) || (comp_channel == NULL)) ? -1 :0;
}

int ra_destroy_comp_channel(const void *rdma_handle, void *comp_channel)
{
    return ((rdma_handle == NULL) || (comp_channel == NULL)) ? -1 :0;
}

const void(*g_raSetTcpRecvCallbackPtr)(const void *fdHandle);
int ra_set_tcp_recv_callback(const void *socket_Handle, const void *callback)
{
    g_raSetTcpRecvCallbackPtr = reinterpret_cast<const void(*)(const void *)>(callback);
    return 0;
}

int ra_qp_create_with_attrs(void *rdma_handle, struct qp_ext_attrs *qp_attrs, void **qpHandle)
{
    return 0;
}

int ra_ai_qp_create(void *rdma_handle, struct qp_ext_attrs *qp_attrs, struct ai_qp_info *info, void **qpHandle)
{
    return 0;
}

int ra_send_wr_v2(QpHandle qphandle, struct send_wr_v2* wr, struct send_wr_rsp* rsp)
{
    return 0;
}

int ra_poll_cq(QpHandle qphandle, bool status, unsigned int num, void* ptr)
{
    return 0;
}

int ra_recv_wrlist(QpHandle handle, struct recv_wrlist_data* wr, unsigned int recvNum, unsigned int* completeNum)
{
    return 0;
}

int ra_qp_batch_modify(RdmaHandle rdmaHandle, QpHandle qpHandle[], unsigned int num, int expectStatus)
{
    return 0;
}

int ra_rdev_get_cqe_err_info_list(void *rdev_handle, struct cqe_err_info *infolist, u32 *num)
{
    return 0;
}

int ra_rdev_init_with_backup(struct rdev_init_info *init_info, struct rdev *rdev_info,
    struct rdev *backup_rdev_info, void **rdma_handle)
{
    return 0;
}

int ra_typical_qp_create(void *rdev_handle, int flag, int qp_mode, struct typical_qp *qp_info, void **qp_handle)
{
    return 0;
}

int ra_typical_qp_modify(void *qp_handle, struct typical_qp *local_qp_info, struct typical_qp *remote_qp_info)
{
    return 0;
}

int ra_typical_send_wr(void *qp_handle, struct send_wr *wr, struct send_wr_rsp *op_rsp)
{
    return 0;
}

int ra_rdev_get_port_status(RdmaHandle rdmaHandle, enum port_status *status)
{
    return 0;
}

int ra_get_qp_attr(void *qp_handle, struct qp_attr *attr)
{
    return 0;
}

int ra_socket_batch_abort(struct socket_connect_info_t conn[], u32 num)
{
    return 0;
}
}
#endif