/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BINRAY_DATA_OPERATOR_H
#define BINRAY_DATA_OPERATOR_H

#include "binary_data_type_pub.h"
#include "hccl_types.h"

namespace HcclSim {

HcclResult FileHeaderWrite(FILE *fp, FileHeader &header);

HcclResult FileHeaderRead(FILE *fp, FileHeader &header, uint32_t magic);

HcclResult HcclVmSynDataRead(FILE *fp, HcclVmSynData &synData, uint32_t magic);

HcclResult HcclVmSynDataWrite(FILE *fp, const HcclVmSynData &synData);

HcclResult ModelInfoCommWrite(FILE *fp, const ModelInfoCommInner &comm);

HcclResult ModelInfoCommRead(FILE *fp, ModelInfoCommInner &comm);

HcclResult VDataDesTagWrite(FILE *fp, const VDataDesTagInner &vDataDes);

HcclResult VDataDesTagRead(FILE *fp, VDataDesTagInner &vDataDes);

HcclResult All2AllDataDesTagWrite(FILE *fp, const All2AllDataDesTagInner &all2AllDataDes);

HcclResult All2AllDataDesTagRead(FILE *fp, All2AllDataDesTagInner &all2AllDataDes);

HcclResult HcclVmInstrDataRead(FILE *fp, HcclVmInstrData &isntrData, uint32_t magic);

HcclResult HcclVmInstrDataWrite(FILE *fp, const HcclVmInstrData &isntrData);

HcclResult ModelInfoWrite(FILE *fp, const ModelInfoInner &modelInfo);

HcclResult ModelInfoRead(FILE *fp, ModelInfoInner &modelInfo);

HcclResult ChannelInfoWrite(FILE *fp, const ChannelInfoInner &chInfo);

HcclResult ChannelInfoRead(FILE *fp, ChannelInfoInner &chInfo);

HcclResult MemLayoutWrite(FILE *fp, const MemLayoutInfoInner &memLayoutInfo);

HcclResult MemLayoutRead(FILE *fp, MemLayoutInfoInner &memLayoutInfo);

HcclResult MicrocodeInstrWrite(FILE *fp, const MicrocodeInstrInner &mcInstrInfo);

HcclResult MicrocodeInstrRead(FILE *fp, MicrocodeInstrInner &mcInstrInfo);

HcclResult TaskMetaWrite(FILE *fp, const HcclTaskMetaData &taskData);

HcclResult TaskMetaRead(FILE *fp, HcclTaskMetaData &taskData);

HcclResult HcclVmTaskMetaDataWrite(FILE *fp, const HcclVmTaskMetaData &taskMeta);

HcclResult HcclVmTaskMetaDataRead(FILE *fp, HcclVmTaskMetaData &taskMeta, uint32_t magic);

}
#endif