#ifndef HCCL_RANK_GRAPH_DL_H
#define HCCL_RANK_GRAPH_DL_H

#include "hccl_rank_graph.h"   // 原头文件，包含所有类型和 inline 函数

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HCCL_E_NOT_SUPPORTED
#define HCCL_E_NOT_SUPPORTED  ((HcclResult)(-2))
#endif

// 声明全局函数指针（小驼峰命名）
extern HcclResult (*hcclRankGraphGetLayersPtr)(HcclComm, uint32_t**, uint32_t*);
extern HcclResult (*hcclRankGraphGetRanksByLayerPtr)(HcclComm, uint32_t, uint32_t**, uint32_t*);
extern HcclResult (*hcclRankGraphGetRankSizeByLayerPtr)(HcclComm, uint32_t, uint32_t*);
extern HcclResult (*hcclRankGraphGetTopoTypeByLayerPtr)(HcclComm, uint32_t, CommTopo*);
extern HcclResult (*hcclRankGraphGetInstSizeListByLayerPtr)(HcclComm, uint32_t, uint32_t**, uint32_t*);
extern HcclResult (*hcclRankGraphGetLinksPtr)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*);
extern HcclResult (*hcclRankGraphGetTopoInstsByLayerPtr)(HcclComm, uint32_t, uint32_t**, uint32_t*);
extern HcclResult (*hcclRankGraphGetTopoTypePtr)(HcclComm, uint32_t, uint32_t, CommTopo*);
extern HcclResult (*hcclRankGraphGetRanksByTopoInstPtr)(HcclComm, uint32_t, uint32_t, uint32_t**, uint32_t*);
extern HcclResult (*hcclGetHeterogModePtr)(HcclComm, HcclHeterogMode*);
extern HcclResult (*hcclRankGraphGetEndpointNumPtr)(HcclComm, uint32_t, uint32_t, uint32_t*);
extern HcclResult (*hcclRankGraphGetEndpointDescPtr)(HcclComm, uint32_t, uint32_t, uint32_t*, EndpointDesc*);
extern HcclResult (*hcclRankGraphGetEndpointInfoPtr)(HcclComm, uint32_t, const EndpointDesc*, EndpointAttr, uint32_t, void*);

// 宏：将原始API名映射为函数指针调用（保持API名大驼峰）
#define HcclRankGraphGetLayers              (*hcclRankGraphGetLayersPtr)
#define HcclRankGraphGetRanksByLayer        (*hcclRankGraphGetRanksByLayerPtr)
#define HcclRankGraphGetRankSizeByLayer     (*hcclRankGraphGetRankSizeByLayerPtr)
#define HcclRankGraphGetTopoTypeByLayer     (*hcclRankGraphGetTopoTypeByLayerPtr)
#define HcclRankGraphGetInstSizeListByLayer (*hcclRankGraphGetInstSizeListByLayerPtr)
#define HcclRankGraphGetLinks                (*hcclRankGraphGetLinksPtr)
#define HcclRankGraphGetTopoInstsByLayer     (*hcclRankGraphGetTopoInstsByLayerPtr)
#define HcclRankGraphGetTopoType              (*hcclRankGraphGetTopoTypePtr)
#define HcclRankGraphGetRanksByTopoInst       (*hcclRankGraphGetRanksByTopoInstPtr)
#define HcclGetHeterogMode                    (*hcclGetHeterogModePtr)
#define HcclRankGraphGetEndpointNum           (*hcclRankGraphGetEndpointNumPtr)
#define HcclRankGraphGetEndpointDesc          (*hcclRankGraphGetEndpointDescPtr)
#define HcclRankGraphGetEndpointInfo          (*hcclRankGraphGetEndpointInfoPtr)

void HcclRankGraphDlInit(void* libHcommHandle);
void HcclRankGraphDlFini(void);

#ifdef __cplusplus
}
#endif

#endif // HCCL_RANK_GRAPH_DL_H