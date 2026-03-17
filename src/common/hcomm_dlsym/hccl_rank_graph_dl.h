#ifndef HCCL_RANK_GRAPH_DL_H
#define HCCL_RANK_GRAPH_DL_H

#include "hccl_rank_graph.h"   // 原头文件，包含所有类型和 inline 函数

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum);
HcclResult HcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum);
HcclResult HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum);
HcclResult HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo* topoType);
HcclResult HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize);
HcclResult HcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
                                 CommLink** links, uint32_t* linkNum);
HcclResult HcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
HcclResult HcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType);
HcclResult HcclRankGraphGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId,
                                          uint32_t** ranks, uint32_t* rankNum);
HcclResult HcclGetHeterogMode(HcclComm comm, HcclHeterogMode* mode);
HcclResult HcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num);
HcclResult HcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId,
                                        uint32_t* descNum, EndpointDesc* endpointDesc);
HcclResult HcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc,
                                        EndpointAttr endpointAttr, uint32_t infoLen, void* info);

void HcclRankGraphDlInit(void* libHcommHandle);
void HcclRankGraphDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_RANK_GRAPH_DL_H