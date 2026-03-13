#include "storage_manager.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>
#include <zlib.h>
#include <vector>
#include <cstring>
#include <cstdlib> // strtoull
#include "binary_data_operator.h"
#include "hccl_vm_log.h"

extern std::map<RankId, std::map<u32, HcclSim::ChannelsPerDie>> g_allRankChannelInfo;

namespace HcclSim {

static const std::string PLUGIN_PATH = "/plugin";
static const std::string DATA_FILE_PATH = "/data";
static const std::string TASK_COLLECTION_FILE = "/%s_task.jsonl.gz";
static const std::string MEM_LAYOUT_FILE = "/%s_mem_layout.jsonl.gz";
static const std::string MODEL_FILE = "/%s_model.jsonl.gz";

static const std::string HCCLVM_TASK_DATA_FILE = "/%s_hcclvm_task_data.bin";
static const std::string HCCLVM_SYN_DATA_FILE = "/%s_hcclvm_syn_data.bin";
static const std::string HCCLVM_INSTR_DATA_FILE = "/%s_hcclvm_instr_data.bin";

std::vector<HcclTaskMetaData> StorageManager::GetTaskCollection()
{
    if (m_data_id.empty()) {
        HCCL_VM_ERROR("[StorageManager][GetTaskCollection] DataId is empty");
        return {};
    }

    char fileName[256];
    // 使用 snprintf 是安全的，因为它限制了长度
    snprintf(fileName, sizeof(fileName), TASK_COLLECTION_FILE.c_str(), m_data_id.c_str());
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("[StorageManager] Failed to find root path");
        return {};
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

    gzFile file = gzopen(fullPath.c_str(), "rb");
    if (!file) {
        HCCL_VM_ERROR("[StorageManager] Failed to open task collection file: {}", fullPath);
        return {};
    }

    std::vector<HcclTaskMetaData> taskCollection;
    char buffer[2048];
    uint32_t totalCount = 0;

    // 1. 读取 Header 预分配内存
    if (gzgets(file, buffer, sizeof(buffer))) {
        try {
            auto header = nlohmann::json::parse(buffer);
            if (header.contains("total_count")) {
                totalCount = header["total_count"].get<uint32_t>();
                // 性能核心：预分配千万级内存，避免动态扩容导致的多次拷贝
                taskCollection.reserve(totalCount);
                HCCL_VM_INFO("[StorageManager] Total tasks to load: {}", totalCount);
            }
        } catch (const nlohmann::json::parse_error& e) {
            HCCL_VM_ERROR("[StorageManager] Header parse error: {}", e.what());
            gzclose(file);
            return {};
        }
    }

    uint32_t processedCount = 0;
    // 2. 逐行解析数据
    while (gzgets(file, buffer, sizeof(buffer))) {
        try {
            // 直接解析当前行的 JSON 数组
            auto j = nlohmann::json::parse(buffer);
            if (!j.is_array() || j.size() < 15) continue;

            HcclTaskMetaData meta;
            // 公共基础字段解析
            meta.taskType = static_cast<HccLTaskMetaType>(j[0].get<int>());
            meta.commId   = j[1].get<uint16_t>();
            meta.rankId   = j[2].get<uint32_t>();
            meta.streamId = j[3].get<uint64_t>();
            // 根据任务类型还原 Union 数据
            // 字段顺序必须与写入时的定义严格一致
            if (meta.taskType == HccLTaskMetaType::MEM_CPY) {
                meta.taskData.transMem.srcRankId = j[4].get<uint32_t>();
                meta.taskData.transMem.srcOffset = j[5].get<uint64_t>();
                meta.taskData.transMem.dstRankId = j[6].get<uint32_t>();
                meta.taskData.transMem.dstOffset = j[7].get<uint64_t>();
                meta.taskData.transMem.len       = j[8].get<uint64_t>();
                meta.taskData.transMem.protocol  = j[14].get<uint8_t>();
            } 
            else if (meta.taskType == HccLTaskMetaType::REDUCE) {
                meta.taskData.reduce.srcRankId   = j[4].get<uint32_t>();
                meta.taskData.reduce.srcOffset   = j[5].get<uint64_t>();
                meta.taskData.reduce.dstRankId   = j[6].get<uint32_t>();
                meta.taskData.reduce.dstOffset   = j[7].get<uint64_t>();
                meta.taskData.reduce.dataType    = j[9].get<uint8_t>();
                meta.taskData.reduce.dataCount   = j[10].get<uint64_t>();
                meta.taskData.reduce.reduceOp    = j[11].get<uint8_t>();
                meta.taskData.reduce.protocol    = j[14].get<uint8_t>();
            } 
            else if (meta.taskType == HccLTaskMetaType::NOTIFY_RECORD
                || meta.taskType == HccLTaskMetaType::NOTIFY_WAIT) {
                meta.taskData.notify.srcRankId   = j[4].get<uint32_t>();
                meta.taskData.notify.notifyId    = j[12].get<uint64_t>();
                meta.taskData.notify.dstRankId   = j[6].get<uint32_t>();
                meta.taskData.notify.notifyCount = j[13].get<uint8_t>();
                meta.taskData.notify.protocol    = j[14].get<uint8_t>();
            }

            taskCollection.push_back(meta);
            processedCount++;

            // 5. 可选：进度打印 (每 100 万条打印一次)
            if (totalCount > 0 && processedCount % 1000000 == 0) {
                float progress = (static_cast<float>(processedCount) / totalCount) * 100.0f;
                HCCL_VM_INFO("[StorageManager] Loading progress: {.1f}%", progress);
            }

        } catch (const nlohmann::json::parse_error& e) {
            HCCL_VM_ERROR("[StorageManager] Line parse error: {}", e.what());
            // 容错处理：单行损坏不中断整体流程
            continue; 
        }
    }

    gzclose(file);

    // 6. 最终完整性校验
    if (totalCount > 0 && processedCount != totalCount) {
        HCCL_VM_INFO("[StorageManager][Warning] Data integrity mismatch!");
        HCCL_VM_INFO("Expected: {}, Actually processed: {}", totalCount, processedCount);
        // 根据业务需求，可以选择返回空或者抛出异常
        return {}; 
    } else {
        HCCL_VM_INFO("[StorageManager] Data load success. Total tasks: {}", processedCount);
    }
    return taskCollection;
}

HcclResult StorageManager::LoadCheckerParam()
{
    if (m_data_id.empty()) {
        HCCL_VM_ERROR("[StorageManager][LoadCheckerParam] DataId is empty");
        return HcclResult::HCCL_E_PARA;
    }

    char fileName[256];
    snprintf(fileName, sizeof(fileName), MODEL_FILE.c_str(), m_data_id.c_str());
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("[StorageManager] Failed to find root path");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

    gzFile file = gzopen(fullPath.c_str(), "rb");
    if (!file) {
        HCCL_VM_ERROR("[StorageManager] Failed to open model file: {}", fullPath);
        return HcclResult::HCCL_E_OPEN_FILE_FAILURE;
    }

    // 3. 将文件内容读入字符串
    std::string content;
    char buffer[4096];
    int bytesRead;
    while ((bytesRead = gzread(file, buffer, sizeof(buffer))) > 0) {
        content.append(buffer, bytesRead);
    }
    gzclose(file);

    // 4. 解析 JSON 对象
    try {
        auto j = nlohmann::json::parse(content);

        // 使用 value() 优雅地处理缺失字段
        // 如果 Key 不存在，则使用第二个参数作为默认值
        m_checker_param.cmdType    = static_cast<HcclCMDType>(j.value("cmdType", 0));
        m_checker_param.rankSize   = j.value("rankSize", 0u);
        m_checker_param.dataType   = static_cast<HcclDataType>(j.value("dataType", 0));
        m_checker_param.dataCount  = j.value("dataCount", 0ull);
        
        m_checker_param.reduceType = static_cast<HcclReduceOp>(j.value("reduceType", 0));
        m_checker_param.srcRank    = j.value("srcRank", 0u);
        m_checker_param.dstRank    = j.value("dstRank", 0u);
        m_checker_param.root       = j.value("root", 0u);

    } catch (const nlohmann::json::parse_error& e) {
        HCCL_VM_ERROR("Param parse error: {}", e.what());
        return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult StorageManager::Trans2CheckerParam()
{
    m_checker_param.cmdType = static_cast<HcclCMDType>(m_synData.model_info.comm.op_type);
    m_checker_param.rankSize = m_synData.model_info.comm.rank_size;
    m_checker_param.dataType = static_cast<HcclDataType>(m_synData.model_info.comm.data_type);
    m_checker_param.dataCount = m_synData.model_info.comm.data_count;
    m_checker_param.reduceType = static_cast<HcclReduceOp>(m_synData.model_info.comm.reduce_op);
    m_checker_param.srcRank  = m_synData.model_info.comm.src_rank;
    m_checker_param.dstRank  = m_synData.model_info.comm.dst_rank;
    m_checker_param.root     = m_synData.model_info.comm.root;
    m_checker_param.all2AllDataDes.sendType = m_synData.model_info.all2AllDataDes.sendType;
    m_checker_param.all2AllDataDes.recvType = m_synData.model_info.all2AllDataDes.recvType;
    m_checker_param.all2AllDataDes.sendCount = m_synData.model_info.all2AllDataDes.sendCount;
    m_checker_param.all2AllDataDes.recvCount = m_synData.model_info.all2AllDataDes.recvCount;
    m_checker_param.all2AllDataDes.count = m_synData.model_info.all2AllDataDes.count;
    // m_checker_param.all2AllDataDes.sendCountMatrix = m_synData.model_info.all2AllDataDes.sendCountMatrix;
    for (uint32_t i = 0; i < m_checker_param.all2AllDataDes.count; i++) {
        m_checker_param.all2AllDataDes.sendCountMatrix.push_back(m_synData.model_info.all2AllDataDes.sendCountMatrix[i]);
    }

    HCCL_VM_INFO("[Trans2CheckerParam] Success");

    return HcclResult::HCCL_SUCCESS;
}

HcclResult StorageManager::LoadMemLayout()
{
    if (m_data_id.empty()) {
        HCCL_VM_ERROR("[StorageManager][LoadMemLayout] DataId is empty");
        return HcclResult::HCCL_E_PARA;
    }

    // 1. 构造路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), MEM_LAYOUT_FILE.c_str(), m_data_id.c_str());
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("[StorageManager] Failed to find root path");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

    gzFile file = gzopen(fullPath.c_str(), "rb");
    if (!file) {
        HCCL_VM_ERROR("[StorageManager] Failed to open buffer file: {}", fullPath);
        return HcclResult::HCCL_E_OPEN_FILE_FAILURE;
    }

    m_mem_layout.clear();
    char buffer[4096];

    // 2. 跳过 Header 行
    if (!gzgets(file, buffer, sizeof(buffer))) {
        HCCL_VM_ERROR("[StorageManager] Failed to read buffer file header: {}", fullPath);
        gzclose(file);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 3. 逐行解析数据行，格式例: [0, [1024, 2048, 512]]
    while (gzgets(file, buffer, sizeof(buffer))) {
        try {
            auto j = nlohmann::json::parse(buffer);
            if (!j.is_array() || j.size() < 4) continue;

            uint32_t rid   = j[0].get<uint32_t>();
            uint64_t sAddr = j[1].get<uint64_t>();
            uint64_t size  = j[2].get<uint64_t>();
            BufferType  type  = static_cast<BufferType>(j[3].get<uint8_t>());

            // 存入时，map 会自动按 sAddr 排序
            m_mem_layout[rid][type][sAddr] = {sAddr, size, type, 0};
        } catch (...) { continue; }
    }

    for (auto& rankEntry : m_mem_layout) {
        // 遍历每个 Rank 下的所有类型 (INPUT, OUTPUT...)
        for (auto& typeEntry : rankEntry.second) {
            uint64_t currentAccumulatedSize = 0;
            // 因为 addrMap 是按 startAddr 从小到大排序的，遍历顺序即物理地址顺序
            for (auto& addrEntry : typeEntry.second) {
                MemBlock& block = addrEntry.second;
                block.globalOffset = currentAccumulatedSize; // 记录在该块之前的总长度
                currentAccumulatedSize += block.size;         // 累加
            }
        }
    }

    gzclose(file);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult StorageManager::LoadHcclVmSynthesisData()
{
    if (m_data_id.empty()) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclvmSynthesisData] DataId is empty");
        return HcclResult::HCCL_E_PARA;
    }

    // 1. 构造路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_SYN_DATA_FILE.c_str(), m_data_id.c_str());
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclvmSynthesisData] Failed to find root path");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;
    FILE *fp = fopen(fullPath.c_str(), "rb");
    if (!fp) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclvmSynthesisData] Open file failed: {}", rootPath);
        return HcclResult::HCCL_E_INTERNAL;
    }

    auto ret = HcclVmSynDataRead(fp, m_synData, HCCLVM_SYN_FILE_MAGIC);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclvmSynthesisData] Read hccl vm synthesis file failed. ");
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 转换channel映射表
    for (auto &channel : m_synData.channel_info.data) {
        RemoteDieInfo rmtDieInfo1;
        rmtDieInfo1.dstRank = channel.dstRank;
        rmtDieInfo1.remoteDieId = channel.dstDieId;
        HCCL_VM_INFO("zhf-channel info: channelId= {}, srcRank= {}, srcDie= {}, dstRank= {}, dstDie= {}",
            channel.channelId, channel.srcRank, static_cast<uint32_t>(channel.srcDieId), rmtDieInfo1.dstRank, rmtDieInfo1.remoteDieId);
        g_allRankChannelInfo[channel.srcRank][channel.srcDieId][channel.channelId] = rmtDieInfo1;
    }

    // 构造memory layout
    HCCL_VM_INFO("Read memory info: {},{}", m_synData.memory_info.count, m_synData.memory_info.data.size());
    for (auto &memInfo : m_synData.memory_info.data) {
        auto rankId  = memInfo.rank_id;
        auto bufType = static_cast<BufferType>(memInfo.buffer_type);
        auto startAddr = memInfo.start_addr;

        MemBlock memBlock;
        memBlock.bufferType = bufType;
        memBlock.startAddr  = startAddr;
        memBlock.size       = memInfo.size;
        memBlock.globalOffset = 0; // todo: 预期一个rank只有一个同类型的buffer时，globalOffset为0。若有多个，需要按照下面json方案计算

        m_mem_layout[rankId][bufType][startAddr] = memBlock;
        HCCL_VM_INFO("[Init MemLayout] rank{}, bufType= {}, startAddr={}, size={}, globalOffset= {}",
            rankId, memInfo.buffer_type, startAddr, memInfo.size, memBlock.globalOffset);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult StorageManager::LoadHcclVmInstrData()
{
    if (m_data_id.empty()) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclVmInstrData] DataId is empty");
        return HcclResult::HCCL_E_PARA;
    }

    // 1. 构造路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_INSTR_DATA_FILE.c_str(), m_data_id.c_str());
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclVmInstrData] Failed to find root path");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;
    FILE *fp = fopen(fullPath.c_str(), "rb");
    if (!fp) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclVmInstrData] Open file failed: {}", rootPath);
        return HcclResult::HCCL_E_INTERNAL;
    }

    auto ret = HcclVmInstrDataRead(fp, m_instrData, HCCLVM_INSTR_FILE_MAGIC);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclVmInstrData] Read hccl vm instruction file failed. ");
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_VM_INFO("[LoadHcclVmInstrData] Read instructions success....");
    HCCL_VM_INFO("[LoadHcclVmInstrData] ccu size= {}", m_instrData.instr_data.size());
    for (auto &ccuInstr : m_instrData.instr_data) {
        HCCL_VM_INFO("[LoadHcclVmInstrData] rankId= {}, dieId= {}, count= {}",
            ccuInstr.desc.rank_id, static_cast<uint32_t>(ccuInstr.desc.die_id), ccuInstr.desc.count);
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult StorageManager::LoadHcclVmTaskMetaData()
{
    if (m_data_id.empty()) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclVmTaskMetaData] DataId is empty");
        return HcclResult::HCCL_E_PARA;
    }

    // 1. 构造路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_TASK_DATA_FILE.c_str(), m_data_id.c_str());
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclVmTaskMetaData] Failed to find root path");
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;
    FILE *fp = fopen(fullPath.c_str(), "rb");
    if (!fp) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclVmTaskMetaData] Open file failed: ");
        return HcclResult::HCCL_E_INTERNAL;
    }

    auto ret = HcclVmTaskMetaDataRead(fp, m_taskMeataData, HCCLVM_TASK_FILE_MAGIC);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("[StorageManager][LoadHcclVmTaskMetaData] Read hccl vm task meta file failed. ");
        return HcclResult::HCCL_E_INTERNAL;
    }

    HCCL_VM_INFO("[LoadHcclVmTaskMetaData] Read task meata success....");
    HCCL_VM_INFO("[LoadHcclVmTaskMetaData] task_count= {}", m_taskMeataData.header.count);
    for (auto &taskMeta : m_taskMeataData.task_meta) {
        HCCL_VM_INFO("[LoadHcclVmTaskMetaData] rankId= {}, dieId= {}, instrCnt= {}, argSize= {}, streamId= {}",
            taskMeta.rankId, static_cast<uint32_t>(taskMeta.taskData.ccu.dieId), taskMeta.taskData.ccu.instCnt, taskMeta.taskData.ccu.argSize, taskMeta.streamId);
    }

    return HcclResult::HCCL_SUCCESS;
}

uint64_t StorageManager::GetBlockSize(uint32_t rankId, BufferType bufferType) {
    // 1. 定位 Rank
    auto rankIt = m_mem_layout.find(rankId);
    if (rankIt == m_mem_layout.end()) {
        HCCL_VM_INFO("[GetBlockSize] Cannot find rank id from memory layout");
        return 0;
    }

    // 2. 定位 BufferType
    auto typeIt = rankIt->second.find(bufferType);
    if (typeIt == rankIt->second.end()) {
        HCCL_VM_INFO("[GetBlockSize] Cannot find buffer type from memory layout");
        return 0;
    }

    // 3. 获取该类型下的最后一个块 (map 的 rbegin)
    const auto& addrMap = typeIt->second;
    if (addrMap.empty()) {
        HCCL_VM_INFO("[GetBlockSize] Cannot find addr from memory layout");
        return 0;
    }

    // map 是有序的，rbegin() 指向起始地址最大的那个块
    const MemBlock& lastBlock = addrMap.rbegin()->second;

    // HCCL_VM_INFO("[GetBlockSize] return block size: {}, {}", lastBlock.globalOffset, lastBlock.size);

    // 总大小 = 最后一个块的逻辑起始偏移 + 最后一个块的大小
    return lastBlock.globalOffset + lastBlock.size;
}

DataSlice StorageManager::GetDataSlice(uint32_t rankId, uint64_t addr, uint64_t size)
{
    DataSlice slice;
    slice.SetSize(size);

    // 1. 定位该 Rank 的内存布局
    auto rankIter = m_mem_layout.find(rankId);
    if (rankIter == m_mem_layout.end()) {
        return slice;
    }

    // 2. 遍历该 Rank 下的所有 Buffer 类型 (INPUT, OUTPUT, CCL...)
    // typeEntry.first 是 BufferType, typeEntry.second 是 addrMap
    for (auto const& typeEntry : rankIter->second) {
        const auto& addrMap = typeEntry.second;

        // 3. 使用 upper_bound 在当前类型的地址图中快速查找
        // 找到第一个起始地址大于 addr 的块，那么目标块就是它的前一个
        auto it = addrMap.upper_bound(addr);
        
        if (it != addrMap.begin()) {
            --it;
            const MemBlock& block = it->second;

            // 4. 边界检查：确认物理地址 addr 是否落在该块 [start, start + size) 内
            if (addr >= block.startAddr && addr < (block.startAddr + block.size)) {
                // 校验区间完整性（可选）：确保整个 size 都在这个块内
                // 如果允许跨块，逻辑会更复杂，这里按单块逻辑处理
                
                slice.SetBufferType(block.bufferType);
                // 核心转换公式：逻辑基址 + (物理地址 - 物理块基址)
                slice.SetOffset(block.globalOffset + (addr - block.startAddr));
                
                return slice; // 找到即返回
            }
        }
    }

    return slice; // 未找到匹配的物理区间
}

HcclResult StorageManager::GetSlice(uint64_t addr, uint64_t len, DataSlice& dataSlice, uint32_t* rank)
{
    dataSlice.SetSize(len);

    // for (auto &rankMem : m_mem_layout) {
    //     // 2. 遍历该 Rank 下的所有 Buffer 类型 (INPUT, OUTPUT, CCL...)
    //     // typeEntry.first 是 BufferType, typeEntry.second 是 addrMap
    //     for (auto const& typeEntry : rankMem.second) {
    //         for (auto &xx: typeEntry.second) {
    //             std::cout<<"zhf-memlayout: rank"<<rankMem.first<<", "<<xx.first<<", size="<<xx.second.size<<std::endl;
    //         }
    //     }
    // }

    for (auto &rankMem : m_mem_layout) {
        // 2. 遍历该 Rank 下的所有 Buffer 类型 (INPUT, OUTPUT, CCL...)
        // typeEntry.first 是 BufferType, typeEntry.second 是 addrMap
        for (auto const& typeEntry : rankMem.second) {
            const auto& addrMap = typeEntry.second;

            // 3. 使用 upper_bound 在当前类型的地址图中快速查找
            // 找到第一个起始地址大于 addr 的块，那么目标块就是它的前一个
            auto it = addrMap.upper_bound(addr);
            // std::cout<<"[FIND MEM] start "<<rankMem.first<<", "<<addr<<std::endl;
            if (it != addrMap.begin()) {
                --it;
                const MemBlock& block = it->second;

                // std::cout<<"[FIND MEM] inner 1: "<<block.startAddr<<", "<<block.size<<", "<<block.startAddr + block.size<<std::endl;
                // 4. 边界检查：确认物理地址 addr 是否落在该块 [start, start + size) 内
                if (addr >= block.startAddr && addr < (block.startAddr + block.size)) {
                    // 校验区间完整性（可选）：确保整个 size 都在这个块内
                    // 如果允许跨块，逻辑会更复杂，这里按单块逻辑处理
                    
                    dataSlice.SetBufferType(block.bufferType);
                    // 核心转换公式：逻辑基址 + (物理地址 - 物理块基址)
                    dataSlice.SetOffset(block.globalOffset + (addr - block.startAddr));
                    // std::cout<<"zhf-found ...."<<rankMem.first<<std::endl;
                    if (rank != nullptr) {
                        *rank = rankMem.first;
                    }
                    
                    return HcclResult::HCCL_SUCCESS;
                }
            }
        }
    }

    HCCL_VM_ERROR("[GetSlice] Cannot find addr...:addr= {}, len={}", addr, len);
    return HcclResult::HCCL_E_MEMORY;
}

std::string StorageManager::FindRootPath()
{
    // 使用相对路径前缀：.  ./..  ./../..
    std::string current_search_path = ".";
    for (int i = 0; i <= 3; ++i) {
        char abs_path[PATH_MAX];
        // 尝试获取当前探测点的绝对路径
        if (realpath(current_search_path.c_str(), abs_path) != nullptr) {
            std::string check_target = std::string(abs_path) + PLUGIN_PATH;

            // 检查该绝对路径下的 plugin 目录是否存在
            if (IsDirExists(check_target)) {
                return std::string(abs_path);
            }
        } else {
            // 如果 realpath 失败（例如路径被删除或权限不足）
            HCCL_VM_ERROR("[FindRootPath] Iteration {}: realpath failed for {}", i, current_search_path);
        }

        // 没找到，将探测路径向上推一级
        current_search_path += "/..";
    }

    HCCL_VM_INFO("[FindRootPath] RootPath NOT found.");
    return ""; 
}

bool StorageManager::IsDirExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false; // 不存在
    }
    return (info.st_mode & S_IFDIR); // 存在且是目录
}

uint32_t StorageManager::GetRankSize() const
{
    return m_checker_param.rankSize;
}

HcclVmInstrData StorageManager::GetHvmInstrData() const
{
    return m_instrData;
}

HcclVmTaskMetaData StorageManager::GetHvmTaskMetaData() const
{
    return m_taskMeataData;
}
}