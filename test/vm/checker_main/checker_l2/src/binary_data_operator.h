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
#include "hccl_common_defs.h"

namespace HcclSim {

HcclVmResult FileHeaderWrite(FILE *fp, FileHeader &header);

HcclVmResult FileHeaderRead(FILE *fp, FileHeader &header, uint32_t magic);

HcclVmResult HcclVmSynDataRead(FILE *fp, HcclVmSynData &synData, uint32_t magic);

HcclVmResult HcclVmSynDataWrite(FILE *fp, const HcclVmSynData &synData);

HcclVmResult ModelInfoCommWrite(FILE *fp, const ModelInfoCommInner &comm);

HcclVmResult ModelInfoCommRead(FILE *fp, ModelInfoCommInner &comm);

HcclVmResult VDataDesTagWrite(FILE *fp, const VDataDesTagInner &vDataDes);

HcclVmResult VDataDesTagRead(FILE *fp, VDataDesTagInner &vDataDes);

HcclVmResult All2AllDataDesTagWrite(FILE *fp, const All2AllDataDesTagInner &all2AllDataDes);

HcclVmResult All2AllDataDesTagRead(FILE *fp, All2AllDataDesTagInner &all2AllDataDes);

HcclVmResult HcclVmInstrDataRead(FILE *fp, HcclVmInstrData &isntrData, uint32_t magic);

HcclVmResult HcclVmInstrDataWrite(FILE *fp, const HcclVmInstrData &isntrData);

HcclVmResult ModelInfoWrite(FILE *fp, const ModelInfoInner &modelInfo);

HcclVmResult ModelInfoRead(FILE *fp, ModelInfoInner &modelInfo);

HcclVmResult ChannelInfoWrite(FILE *fp, const ChannelInfoInner &chInfo);

HcclVmResult ChannelInfoRead(FILE *fp, ChannelInfoInner &chInfo);

HcclVmResult MemLayoutWrite(FILE *fp, const MemLayoutInfoInner &memLayoutInfo);

HcclVmResult MemLayoutRead(FILE *fp, MemLayoutInfoInner &memLayoutInfo);

HcclVmResult MicrocodeInstrWrite(FILE *fp, const MicrocodeInstrInner &mcInstrInfo);

HcclVmResult MicrocodeInstrRead(FILE *fp, MicrocodeInstrInner &mcInstrInfo);

HcclVmResult TaskMetaWrite(FILE *fp, const HcclTaskMetaData &taskData);

HcclVmResult TaskMetaRead(FILE *fp, HcclTaskMetaData &taskData);

HcclVmResult HcclVmTaskMetaDataWrite(FILE *fp, const HcclVmTaskMetaData &taskMeta);

HcclVmResult HcclVmTaskMetaDataRead(FILE *fp, HcclVmTaskMetaData &taskMeta, uint32_t magic);

}
#endif