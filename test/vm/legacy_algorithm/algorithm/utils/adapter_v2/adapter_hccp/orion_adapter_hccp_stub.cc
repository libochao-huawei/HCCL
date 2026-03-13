#include "orion_adapter_hccp.h"
#include "rt_external.h"
#include "dev_type.h"
#include "hccp_ctx.h"
#include "ccu_res_specs.h"
#include "coll_gen_json.h"
#include "rank_info_recorder.h"
#include "rdma_handle_manager_stub.h"

namespace Hccl {
constexpr uint32_t MOVE_TOW_BYTES   = 16;
constexpr uint32_t MOVE_THREE_BYTES = 24;

void HrtRaUbDestroyJetty(JettyHandle jettyHandle)
{
    ;
}

void HrtRaCustomChannel(const HRaInfo &raInfo, void *customIn, void *customOut)
{
    CustomChannelInfoIn *input = reinterpret_cast<CustomChannelInfoIn *>(customIn);
    CustomChannelInfoOut *output = reinterpret_cast<CustomChannelInfoOut *>(customOut);
    uint8_t dieId = input->data.dataInfo.udieIdx;
    //对2D场景下的enableFlag做一个适配
    if (input->op == CcuOpcodeType::CCU_U_OP_GET_DIE_WORKING) {
        const char* env_value = std::getenv("HCCL_IODIE_NUM");
        if (env_value != nullptr) {
            std::string value(env_value);
            if (value == "2") {
                output->data.dataInfo.dataArray[0].dieinfo.enableFlag = 1;
                return;
            }
            std::cout << "环境变量值: " << value << std::endl;
        }
        output->data.dataInfo.dataArray[0].dieinfo.enableFlag = (dieId == 0) ? 1 : 0;  // 单die场景，只让die0可用
    } else if (input->op == CcuOpcodeType::CCU_U_OP_GET_BASIC_INFO) {
        output->data.dataInfo.dataArray[0].baseinfo.resourceAddr = 0x123456789;
        output->data.dataInfo.dataArray[0].baseinfo.missionKey = 0;
        output->data.dataInfo.dataArray[0].baseinfo.msId = 3;  // ???
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
    }

    return;
}

HrtRaUbJettyCreatedOutParam HrtRaUbCreateJetty(RdmaHandle handle, const HrtRaUbCreateJettyParam &in)
{
    HrtRaUbJettyCreatedOutParam out;
    return out;
}

JfcHandle HrtRaUbCreateJfc(RdmaHandle handle, HrtUbJfcMode mode)
{
    void *jfcHandle = nullptr;
    return reinterpret_cast<JfcHandle>(jfcHandle);
}

void HrtRaUbUnimportJetty(RdmaHandle handle, TargetJettyHandle targetJettyHandle)
{ }

HrtRaUbJettyImportedOutParam HrtRaUbImportJetty(RdmaHandle handle, u8 *key, u32 keyLen, u32 tokenValue)
{
    HrtRaUbJettyImportedOutParam out;
    return out;
}

std::pair<uint32_t, uint32_t> HraGetDieAndFuncId(RdmaHandle handle)
{
    if (g_rdmaHandle2DieIdAndFuncId.count(handle) == 0) {
        HCCL_ERROR("handle %p is not found", handle);
    }
    auto result = g_rdmaHandle2DieIdAndFuncId[handle];
    return std::make_pair(result.first, result.second);
}

std::vector<HrtDevEidInfo> HrtRaGetDevEidInfoList(const HRaInfo &raInfo)
{
    return g_devId2EidInfo[raInfo.phyId];
}

HrtRaUbJettyImportedOutParam ImportJetty(RdmaHandle handle, u8 *key, u32 keyLen,
    u32 tokenValue, JettyImportExpCfg cfg)
{
    HrtRaUbJettyImportedOutParam out;
    return out;
}

HrtRaUbJettyImportedOutParam RaUbCtpImportJetty(RdmaHandle handle, u8 *key, u32 keyLen,
    u32 tokenValue)
{
    struct JettyImportExpCfg cfg = {};
    return ImportJetty(handle, key, keyLen, tokenValue, cfg);
}

HrtRaUbJettyImportedOutParam RaUbImportJetty(RdmaHandle handle, u8 *key, u32 keyLen, u32 tokenValue)
{
    struct JettyImportExpCfg cfg = {};
    return ImportJetty(handle, key, keyLen, tokenValue, cfg);
}

Eid IpAddressToEid(const IpAddress &ipAddr)
{
    Eid eid;
    return eid;
}

HrtRaUbJettyImportedOutParam RaUbTpImportJetty(RdmaHandle handle, u8 *key, u32 keyLen,
    u32 tokenValue, const JettyImportCfg &jettyImportCfg)
{
    struct JettyImportExpCfg cfg = {};
    return ImportJetty(handle, key, keyLen, tokenValue, cfg);
}

std::pair<TokenIdHandle, uint32_t> RaUbAllocTokenIdHandle(RdmaHandle handle)
{
    (void)handle;
    TokenIdHandle tokenIdHandle = 0;
    uint32_t tokenId = 0;
    return std::make_pair(tokenIdHandle, tokenId);
}

void RaUbFreeTokenIdHandle(RdmaHandle handle, TokenIdHandle tokenIdHandle)
{
    (void)handle;
    (void)tokenIdHandle;
}

}

