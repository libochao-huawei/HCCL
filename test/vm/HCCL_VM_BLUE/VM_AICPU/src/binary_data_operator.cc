#include <cstring>
#include <cstdlib> // strtoull
#include <iostream>
#include "binary_data_operator.h"

namespace HcclSim {

static int WriteExact(FILE *fp, const void *ptr, size_t size, size_t n)
{
    return fwrite(ptr, size, n, fp) == n ? 0 : -1;
}

static int ReadExact(FILE *fp, void *ptr, size_t size, size_t n)
{
    return fread(ptr, size, n, fp) == n ? 0 : -1;
}

HcclVmResult FileHeaderWrite(FILE *fp, const FileHeader &header)
{
    if (WriteExact(fp, &header, sizeof(header), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult FileHeaderRead(FILE *fp, FileHeader &header, uint32_t magic)
{
    if (ReadExact(fp, &header, sizeof(header), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    if (header.magic != magic) {
        std::cout<<"[ERROR][FileHeaderRead] Unmatched magic number:"<<std::hex<<header.magic<<"≠"<<magic<<std::endl;
        return HcclVmResult::HCCL_SIM_E_NOT_SUPPORT;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmSynDataRead(FILE *fp, HcclVmSynData &synData, uint32_t magic)
{
    auto ret = FileHeaderRead(fp, synData.header, magic);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmSynDataRead] Read file header failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = ModelInfoRead(fp, synData.model_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmSynDataRead] Read file model info failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = ChannelInfoRead(fp, synData.channel_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmSynDataRead] Read file channel info failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = MemLayoutRead(fp, synData.memory_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmSynDataRead] Read file memory layout info failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmSynDataWrite(FILE *fp, const HcclVmSynData &synData)
{
    auto ret = FileHeaderWrite(fp, synData.header);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmSynDataRead] Read file header failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = ModelInfoWrite(fp, synData.model_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmSynDataRead] Read file model info failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = ChannelInfoWrite(fp, synData.channel_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmSynDataRead] Read file channel info failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = MemLayoutWrite(fp, synData.memory_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmSynDataRead] Read file memory layout info failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmInstrDataRead(FILE *fp, HcclVmInstrData &instrData, uint32_t magic)
{
    auto ret = FileHeaderRead(fp, instrData.header, magic);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmSynDataRead] Read file header failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (instrData.instr_data.size() != instrData.header.count) {
        std::cout << "[HcclVmInstrDataRead] Wrong ccu number. " << instrData.instr_data.size() << " ≠ "
                  << instrData.header.count << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    instrData.instr_data.resize(instrData.header.count);
    for (uint32_t i = 0; i < instrData.header.count; i++) {
        ret = MicrocodeInstrRead(fp, instrData.instr_data[i]);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            std::cout << "[HcclVmSynDataRead] Read file microcode instruction info failed. times= "<< i << std::endl;
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmInstrDataWrite(FILE *fp, const HcclVmInstrData &instrData)
{
    auto ret = FileHeaderWrite(fp, instrData.header);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmInstrDataWrite] Read file header failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    for (uint32_t i = 0; i < instrData.header.count; i++) {
        ret = MicrocodeInstrWrite(fp, instrData.instr_data[i]);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            std::cout << "[HcclVmInstrDataWrite] Write file microcode instruction info failed. times= "<< i << std::endl;
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ModelInfoCommWrite(FILE *fp, const ModelInfoCommInner &comm)
{
    if (WriteExact(fp, &comm, sizeof(comm), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ModelInfoCommRead(FILE *fp, ModelInfoCommInner &comm)
{
    if (ReadExact(fp, &comm, sizeof(comm), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult VDataDesTagWrite(FILE *fp, const VDataDesTagInner &vDataDes)
{
    if (WriteExact(fp, &vDataDes.dataType, sizeof(vDataDes.dataType), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &vDataDes.count, sizeof(vDataDes.count), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, vDataDes.displs.data(), sizeof(uint64_t), vDataDes.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, vDataDes.counts.data(), sizeof(uint64_t), vDataDes.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult VDataDesTagRead(FILE *fp, VDataDesTagInner &vDataDes)
{
    if (ReadExact(fp, &vDataDes.dataType, sizeof(vDataDes.dataType), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &vDataDes.count, sizeof(vDataDes.count), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    vDataDes.displs.resize(vDataDes.count);
    if (ReadExact(fp, vDataDes.displs.data(), sizeof(uint64_t), vDataDes.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    vDataDes.counts.resize(vDataDes.count);
    if (ReadExact(fp, vDataDes.counts.data(), sizeof(uint64_t), vDataDes.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult All2AllDataDesTagWrite(FILE *fp, const All2AllDataDesTagInner &all2AllDataDes)
{
    if (WriteExact(fp, &all2AllDataDes.sendType, sizeof(all2AllDataDes.sendType), 1)) {
        std::cout << "[All2AllDataDesTagWrite] Write all2allDataDes sendType failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &all2AllDataDes.recvType, sizeof(all2AllDataDes.recvType), 1)) {
        std::cout << "[All2AllDataDesTagWrite] Write all2allDataDes recvType failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &all2AllDataDes.sendCount, sizeof(all2AllDataDes.sendCount), 1)) {
        std::cout << "[All2AllDataDesTagWrite] Write all2allDataDes sendCount failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &all2AllDataDes.recvCount, sizeof(all2AllDataDes.recvCount), 1)) {
        std::cout << "[All2AllDataDesTagWrite] Write all2allDataDes recvCount failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &all2AllDataDes.count, sizeof(all2AllDataDes.count), 1)) {
        std::cout << "[All2AllDataDesTagWrite] Write all2allDataDes count failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, all2AllDataDes.sendCountMatrix.data(), sizeof(uint64_t), all2AllDataDes.count)) {
        std::cout << "[All2AllDataDesTagWrite] Write all2allDataDes sendCountMatrix failed. " <<all2AllDataDes.count<<", "<<all2AllDataDes.sendCountMatrix.size()<< std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult All2AllDataDesTagRead(FILE *fp, All2AllDataDesTagInner &all2AllDataDes)
{
    if (ReadExact(fp, &all2AllDataDes.sendType, sizeof(all2AllDataDes.sendType), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &all2AllDataDes.recvType, sizeof(all2AllDataDes.recvType), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &all2AllDataDes.sendCount, sizeof(all2AllDataDes.sendCount), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &all2AllDataDes.recvCount, sizeof(all2AllDataDes.recvCount), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &all2AllDataDes.count, sizeof(all2AllDataDes.count), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    all2AllDataDes.sendCountMatrix.resize(all2AllDataDes.count);
    if (ReadExact(fp, all2AllDataDes.sendCountMatrix.data(), sizeof(uint64_t), all2AllDataDes.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ModelInfoWrite(FILE *fp, const ModelInfoInner &modelInfo)
{
    auto ret = ModelInfoCommWrite(fp, modelInfo.comm);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[ModelInfoWrite] Write model info comm data failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = VDataDesTagWrite(fp, modelInfo.vDataDes);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[ModelInfoWrite] Write model info vdatades data failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = All2AllDataDesTagWrite(fp, modelInfo.all2AllDataDes);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[ModelInfoWrite] Write model info all2allDataDes data failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ModelInfoRead(FILE *fp, ModelInfoInner &modelInfo)
{
    auto ret = ModelInfoCommRead(fp, modelInfo.comm);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[ModelInfoRead] Read model info comm data failed " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = VDataDesTagRead(fp, modelInfo.vDataDes);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[ModelInfoRead] Read model info vdatades data failed.  " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = All2AllDataDesTagRead(fp, modelInfo.all2AllDataDes);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[ModelInfoRead] Read model info all2allDataDes data failed." << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ChannelInfoWrite(FILE *fp, const ChannelInfoInner &chInfo)
{
    if (WriteExact(fp, &chInfo.count, sizeof(chInfo.count), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, chInfo.data.data(), sizeof(ChannelData), chInfo.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ChannelInfoRead(FILE *fp, ChannelInfoInner &chInfo)
{
    if (ReadExact(fp, &chInfo.count, sizeof(chInfo.count), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    chInfo.data.resize(chInfo.count);
    if (ReadExact(fp, chInfo.data.data(), sizeof(ChannelData), chInfo.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MemLayoutWrite(FILE *fp, const MemLayoutInfoInner &memLayoutInfo)
{
    if (WriteExact(fp, &memLayoutInfo.count, sizeof(memLayoutInfo.count), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, memLayoutInfo.data.data(), sizeof(MemLayoutData), memLayoutInfo.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MemLayoutRead(FILE *fp, MemLayoutInfoInner &memLayoutInfo)
{
    if (ReadExact(fp, &memLayoutInfo.count, sizeof(memLayoutInfo.count), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    memLayoutInfo.data.resize(memLayoutInfo.count);
    if (ReadExact(fp, memLayoutInfo.data.data(), sizeof(MemLayoutData), memLayoutInfo.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MicrocodeInstrWrite(FILE *fp, const MicrocodeInstrInner &mcInstrInfo)
{
    if (WriteExact(fp, &mcInstrInfo.desc, sizeof(mcInstrInfo.desc), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, mcInstrInfo.data.data(), sizeof(hcomm::CcuRep::CcuInstr), mcInstrInfo.desc.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MicrocodeInstrRead(FILE *fp, MicrocodeInstrInner &mcInstrInfo)
{
    if (ReadExact(fp, &mcInstrInfo.desc, sizeof(mcInstrInfo.desc), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    mcInstrInfo.data.resize(mcInstrInfo.desc.count);
    if (ReadExact(fp, mcInstrInfo.data.data(), sizeof(hcomm::CcuRep::CcuInstr), mcInstrInfo.desc.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult TaskMetaWrite(FILE *fp, const HcclTaskMetaData &taskData)
{
    if (WriteExact(fp, &taskData, sizeof(HcclTaskMetaData), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult TaskMetaRead(FILE *fp, HcclTaskMetaData &taskData)
{
    if (ReadExact(fp, &taskData, sizeof(HcclTaskMetaData), 1)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// task读写
HcclVmResult HcclVmTaskMetaDataWrite(FILE *fp, const HcclVmTaskMetaData &taskMeta)
{
    auto ret = FileHeaderWrite(fp, taskMeta.header);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[TaskMetaDataWrite] Read file header failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // for (uint32_t i = 0; i < taskMeta.header.count; i++) {
    //     ret = TaskMetaWrite(fp, taskMeta.task_meta[i]);
    //     if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
    //         std::cout << "[TaskMetaDataWrite] Write file task meta data failed. times= "<< i << std::endl;
    //         return HcclVmResult::HCCL_SIM_E_INTERNAL;
    //     }
    // }

    if (WriteExact(fp, taskMeta.task_meta.data(), sizeof(HcclTaskMetaData), taskMeta.header.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmTaskMetaDataRead(FILE *fp, HcclVmTaskMetaData &taskMeta, uint32_t magic)
{
    auto ret = FileHeaderRead(fp, taskMeta.header, magic);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[HcclVmTaskMetaDataRead] Read file header failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // for (uint32_t i = 0; i < taskMeta.header.count; i++) {
    //     ret = TaskMetaRead(fp, taskMeta.task_meta[i]);
    //     if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
    //         std::cout << "[HcclVmTaskMetaDataRead] Read file task meta data failed. times= "<< i << std::endl;
    //         return HcclVmResult::HCCL_SIM_E_INTERNAL;
    //     }
    // }
    taskMeta.task_meta.resize(taskMeta.header.count);
    if (ReadExact(fp, taskMeta.task_meta.data(), sizeof(HcclTaskMetaData), taskMeta.header.count)) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

}