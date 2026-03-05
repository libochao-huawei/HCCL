#include "hccl_rank_graph_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
HcclResult (*hcclGetRankIdPtr)(HcclComm, uint32_t*) = NULL;
HcclResult (*hcclGetRankSizePtr)(HcclComm, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetLayersPtr)(HcclComm, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetRanksByLayerPtr)(HcclComm, uint32_t, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetRankSizeByLayerPtr)(HcclComm, uint32_t, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetTopoTypeByLayerPtr)(HcclComm, uint32_t, CommTopo*) = NULL;
HcclResult (*hcclRankGraphGetInstSizeListByLayerPtr)(HcclComm, uint32_t, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetLinksPtr)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetTopoInstsByLayerPtr)(HcclComm, uint32_t, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetTopoTypePtr)(HcclComm, uint32_t, uint32_t, CommTopo*) = NULL;
HcclResult (*hcclRankGraphGetRanksByTopoInstPtr)(HcclComm, uint32_t, uint32_t, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclGetHeterogModePtr)(HcclComm, HcclHeterogMode*) = NULL;
HcclResult (*hcclRankGraphGetEndpointNumPtr)(HcclComm, uint32_t, uint32_t, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetEndpointDescPtr)(HcclComm, uint32_t, uint32_t, uint32_t*, EndpointDesc*) = NULL;
HcclResult (*hcclRankGraphGetEndpointInfoPtr)(HcclComm, uint32_t, const EndpointDesc*, EndpointAttr, uint32_t, void*) = NULL;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static HcclResult StubHcclGetRankId(HcclComm comm, uint32_t* rank) {
    (void)comm; (void)rank;
    fprintf(stderr, "[HcclWrapper] HcclGetRankId not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclGetRankSize(HcclComm comm, uint32_t* rankSize) {
    (void)comm; (void)rankSize;
    fprintf(stderr, "[HcclWrapper] HcclGetRankSize not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum) {
    (void)comm; (void)netLayers; (void)netLayerNum;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetLayers not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum) {
    (void)comm; (void)netLayer; (void)ranks; (void)rankNum;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetRanksByLayer not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum) {
    (void)comm; (void)netLayer; (void)rankNum;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetRankSizeByLayer not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo* topoType) {
    (void)comm; (void)netLayer; (void)topoType;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetTopoTypeByLayer not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize) {
    (void)comm; (void)netLayer; (void)instSizeList; (void)listSize;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetInstSizeListByLayer not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
                                            CommLink** links, uint32_t* linkNum) {
    (void)comm; (void)netLayer; (void)srcRank; (void)dstRank; (void)links; (void)linkNum;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetLinks not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum) {
    (void)comm; (void)netLayer; (void)topoInsts; (void)topoInstNum;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetTopoInstsByLayer not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType) {
    (void)comm; (void)netLayer; (void)topoInstId; (void)topoType;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetTopoType not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId,
                                                      uint32_t** ranks, uint32_t* rankNum) {
    (void)comm; (void)netLayer; (void)topoInstId; (void)ranks; (void)rankNum;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetRanksByTopoInst not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclGetHeterogMode(HcclComm comm, HcclHeterogMode* mode) {
    (void)comm; (void)mode;
    fprintf(stderr, "[HcclWrapper] HcclGetHeterogMode not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num) {
    (void)comm; (void)layer; (void)topoInstId; (void)num;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetEndpointNum not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId,
                                                   uint32_t* descNum, EndpointDesc* endpointDesc) {
    (void)comm; (void)layer; (void)topoInstId; (void)descNum; (void)endpointDesc;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetEndpointDesc not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc,
                                                   EndpointAttr endpointAttr, uint32_t infoLen, void* info) {
    (void)comm; (void)rankId; (void)endpointDesc; (void)endpointAttr; (void)infoLen; (void)info;
    fprintf(stderr, "[HcclWrapper] HcclRankGraphGetEndpointInfo not supported\n");
    return HCCL_E_NOT_SUPPORTED;
}

void HcclRankGraphDlInit(void* libHcommHandle) {
    // 辅助宏：解析符号，失败则指向对应桩函数
    #define SET_PTR(ptr, name, stub) \
        do { \
            ptr = (typeof(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) ptr = stub; \
        } while(0)

    SET_PTR(hcclGetRankIdPtr, "HcclGetRankId", StubHcclGetRankId);
    SET_PTR(hcclGetRankSizePtr, "HcclGetRankSize", StubHcclGetRankSize);
    SET_PTR(hcclRankGraphGetLayersPtr, "HcclRankGraphGetLayers", StubHcclRankGraphGetLayers);
    SET_PTR(hcclRankGraphGetRanksByLayerPtr, "HcclRankGraphGetRanksByLayer", StubHcclRankGraphGetRanksByLayer);
    SET_PTR(hcclRankGraphGetRankSizeByLayerPtr, "HcclRankGraphGetRankSizeByLayer", StubHcclRankGraphGetRankSizeByLayer);
    SET_PTR(hcclRankGraphGetTopoTypeByLayerPtr, "HcclRankGraphGetTopoTypeByLayer", StubHcclRankGraphGetTopoTypeByLayer);
    SET_PTR(hcclRankGraphGetInstSizeListByLayerPtr, "HcclRankGraphGetInstSizeListByLayer", StubHcclRankGraphGetInstSizeListByLayer);
    SET_PTR(hcclRankGraphGetLinksPtr, "HcclRankGraphGetLinks", StubHcclRankGraphGetLinks);
    SET_PTR(hcclRankGraphGetTopoInstsByLayerPtr, "HcclRankGraphGetTopoInstsByLayer", StubHcclRankGraphGetTopoInstsByLayer);
    SET_PTR(hcclRankGraphGetTopoTypePtr, "HcclRankGraphGetTopoType", StubHcclRankGraphGetTopoType);
    SET_PTR(hcclRankGraphGetRanksByTopoInstPtr, "HcclRankGraphGetRanksByTopoInst", StubHcclRankGraphGetRanksByTopoInst);
    SET_PTR(hcclGetHeterogModePtr, "HcclGetHeterogMode", StubHcclGetHeterogMode);
    SET_PTR(hcclRankGraphGetEndpointNumPtr, "HcclRankGraphGetEndpointNum", StubHcclRankGraphGetEndpointNum);
    SET_PTR(hcclRankGraphGetEndpointDescPtr, "HcclRankGraphGetEndpointDesc", StubHcclRankGraphGetEndpointDesc);
    SET_PTR(hcclRankGraphGetEndpointInfoPtr, "HcclRankGraphGetEndpointInfo", StubHcclRankGraphGetEndpointInfo);

    #undef SET_PTR

    if (dlerror()) {
        fprintf(stderr, "[HcclWrapper] Warning: dlerror after symbol resolution\n");
    }
}

// 销毁函数：将指针重置为桩函数（可选，与 HcclResDlFini 配合使用）
void HcclRankGraphDlFini(void) {
    hcclGetRankIdPtr = StubHcclGetRankId;
    hcclGetRankSizePtr = StubHcclGetRankSize;
    hcclRankGraphGetLayersPtr = StubHcclRankGraphGetLayers;
    hcclRankGraphGetRanksByLayerPtr = StubHcclRankGraphGetRanksByLayer;
    hcclRankGraphGetRankSizeByLayerPtr = StubHcclRankGraphGetRankSizeByLayer;
    hcclRankGraphGetTopoTypeByLayerPtr = StubHcclRankGraphGetTopoTypeByLayer;
    hcclRankGraphGetInstSizeListByLayerPtr = StubHcclRankGraphGetInstSizeListByLayer;
    hcclRankGraphGetLinksPtr = StubHcclRankGraphGetLinks;
    hcclRankGraphGetTopoInstsByLayerPtr = StubHcclRankGraphGetTopoInstsByLayer;
    hcclRankGraphGetTopoTypePtr = StubHcclRankGraphGetTopoType;
    hcclRankGraphGetRanksByTopoInstPtr = StubHcclRankGraphGetRanksByTopoInst;
    hcclGetHeterogModePtr = StubHcclGetHeterogMode;
    hcclRankGraphGetEndpointNumPtr = StubHcclRankGraphGetEndpointNum;
    hcclRankGraphGetEndpointDescPtr = StubHcclRankGraphGetEndpointDesc;
    hcclRankGraphGetEndpointInfoPtr = StubHcclRankGraphGetEndpointInfo;
}