/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RANK_GRAPH_DL_H
#define HCCL_RANK_GRAPH_DL_H

#include "hccl_rank_graph.h"   // 原头文件，包含所有类型和 inline 函数

#ifdef __cplusplus
extern "C" {
#endif

HcclResult __attribute__((weak)) HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum);
HcclResult __attribute__((weak)) HcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum);
HcclResult __attribute__((weak)) HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum);
HcclResult __attribute__((weak)) HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo* topoType);
HcclResult __attribute__((weak)) HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize);
HcclResult __attribute__((weak)) HcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
                                 CommLink** links, uint32_t* linkNum);
HcclResult __attribute__((weak)) HcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
HcclResult __attribute__((weak)) HcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType);
HcclResult __attribute__((weak)) HcclRankGraphGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId,
                                          uint32_t** ranks, uint32_t* rankNum);
HcclResult __attribute__((weak)) HcclGetHeterogMode(HcclComm comm, HcclHeterogMode* mode);
HcclResult __attribute__((weak)) HcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num);
HcclResult __attribute__((weak)) HcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId,
                                        uint32_t* descNum, EndpointDesc* endpointDesc);
HcclResult __attribute__((weak)) HcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc,
                                        EndpointAttr endpointAttr, uint32_t infoLen, void* info);

void HcclRankGraphDlInit(void* libHcommHandle);
void HcclRankGraphDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_RANK_GRAPH_DL_H