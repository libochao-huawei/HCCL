#include "hccl_data_dump.h"
#include "cmd_base_utils.h"
#include "hccl/hccl_types.h"
#include "hccl_vm_log.h"
#include <json.hpp>
#include <zlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <random>
#include "binary_data_type_pub.h"
#include "binary_data_operator.h"
#include "sim_runner_db.h"
#include "sim_models.h"
#include <securec.h>

static const std::string DATA_FILE_PATH = "/data";
static const std::string TASK_COLLECTION_FILE = "/%s_task.jsonl.gz";
static const std::string MEM_LAYOUT_FILE = "/%s_mem_layout.jsonl.gz";
static const std::string MODEL_FILE = "/%s_model.jsonl.gz";

static const std::string HCCLVM_TASK_DATA_FILE = "/%s_hcclvm_task_data.bin";
static const std::string HCCLVM_SYN_DATA_FILE = "/%s_hcclvm_syn_data.bin";
static const std::string HCCLVM_INSTR_DATA_FILE = "/%s_hcclvm_instr_data.bin";

HcclVmResult DumpData()
{
    std::string dataId = GenDataId();
    if (dataId.empty()) {
        HCCL_VM_ERROR("[DumpData] failed to gen dataId.");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    HcclVmResult ret;
    // ret = DumpModel(dataId);
    // if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
    //     std::cout << "[DumpData] failed to Dump Model." << std::endl;
    // }
    // ret = DumpMemLayout(dataId);
    // if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
    //     std::cout << "[DumpData] failed to Dump Mem Layout." << std::endl;
    // }
    
    ret = DumpHcclVmSynthesisData(dataId);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[DumpData] failed to Dump hccl vm snythesis data." << std::endl;
    }
    ret = DumpHcclVmInstrData(dataId);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[DumpData] failed to Dump hccl vm instruction data." << std::endl;
    }
    ret = DumpHcclVmTask(dataId);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[DumpData] failed to Dump hccl vm task data." << std::endl;
    }
    // ret = DumpTask(dataId);
    // if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
    //     std::cout << "[DumpData] failed to Dump Task." << std::endl;
    // }
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    nlohmann::json j;
    j["status"] = "finish";
    j["data_id"] = dataId;
    pluginManager.BroadcastToAllPlugin("status", j);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::string GenDataId()
{
    std::string dataId;
    
    // 1. 初始化随机数生成器 (static 保证只初始化一次，提高性能和随机性)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999); // 4位随机数
    int32_t retryCount = 10;
    while (true) {
        if (retryCount == 0) {
            HCCL_VM_ERROR("[GenDataId] failed to gen dataId.");
            break;
        }
        retryCount--;
        // 2. 获取当前时间戳
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        struct tm buf;
        localtime_r(&in_time_t, &buf);

        // 3. 拼接时间 + 随机数
        std::stringstream ss;
        ss << std::put_time(&buf, "%Y%m%d_%H%M%S") << "_" << dis(gen);
        dataId = ss.str();

        // 4. 冲突检查：检查核心文件是否已存在
        char fileName[256];
        snprintf(fileName, sizeof(fileName), "/%s_model.jsonl.gz", dataId.c_str());
        std::string rootPath = GetBinLocation();
        std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

        if (access(fullPath.c_str(), F_OK) == -1) {
            break; // 文件不存在，DataId 可用，跳出循环
        }
        // 如果冲突，循环会立即重新生成（时间或随机数会变）
    }

    return dataId;
}

HcclVmResult GetChannelDataByEPKey(uint64_t epKey, ChannelData &chData)
{
    // 根据endpointpair id获取根据endpointpair
    auto ep = RunnerDB::GetById<sim::EndPointPair>(epKey);
    if (!ep.has_value()) {
        printf("[ERROR][DumpSimSynData] can not find end point pair by key:%lu\n", epKey);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    // 根据port key获取src/dst port
    auto srcPort = RunnerDB::GetById<sim::Port>(ep->src_port);
    if (!srcPort.has_value()) {
        printf("[ERROR][DumpSimSynData] can not find source port by key:%lu\n", ep->src_port);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    auto dstPort = RunnerDB::GetById<sim::Port>(ep->dst_port);
    if (!dstPort.has_value()) {
        printf("[ERROR][DumpSimSynData] can not find destination port by key:%lu\n", ep->dst_port);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    auto srcDevice = RunnerDB::GetById<sim::Device>(srcPort->device_id);
    if (!srcDevice.has_value()) {
        printf("[ERROR][DumpSimSynData] can not find source device by key:%lu\n", srcPort->device_id);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    auto dstDevice = RunnerDB::GetById<sim::Device>(dstPort->device_id);
    if (!dstDevice.has_value()) {
        printf("[ERROR][DumpSimSynData] can not find destination device by key:%lu\n", dstPort->device_id);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    auto srcCcu = RunnerDB::GetById<sim::Ccu>(srcPort->ccu_id);
    if (!srcCcu.has_value()) {
        printf("[ERROR][DumpSimSynData] can not find source ccu by key:%lu\n", srcPort->ccu_id);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    auto dstCcu = RunnerDB::GetById<sim::Ccu>(dstPort->ccu_id);
    if (!dstCcu.has_value()) {
        printf("[ERROR][DumpSimSynData] can not find destination ccu by key:%lu\n", dstPort->ccu_id);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    
    chData.srcRank  = srcDevice->logic_id;
    chData.dstRank  = dstDevice->logic_id;
    chData.srcDieId = srcCcu->die_id;
    chData.dstDieId = dstCcu->die_id;

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateSimSynData(HcclVmSynData &hvmSynData)
{
    std::cout<<"[CreateSimSynData] Start get simulator synthesis data..."<<std::endl;
    // header
    hvmSynData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    hvmSynData.header.version = 1;
    hvmSynData.header.header_size = 20;
    hvmSynData.header.count = 1;

    // simModel
    std::vector<sim::SimModelData> allSimModels; 
    allSimModels = RunnerDB::GetByPred<sim::SimModelData>([](const sim::SimModelData &d) {
        return d.rank_id == 0;
    });
    if (allSimModels.size() != 1) {
        std::cout<<"[ERROR][CreateSimSynData] Get all simulator models size = "<<allSimModels.size()<<std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    hvmSynData.model_info.comm.root       = allSimModels[0].root;
    hvmSynData.model_info.comm.rank_size  = allSimModels[0].rank_size;
    hvmSynData.model_info.comm.chip_type  = allSimModels[0].chip_type;
    hvmSynData.model_info.comm.op_type    = allSimModels[0].op_type;
    hvmSynData.model_info.comm.reduce_op  = allSimModels[0].reduce_op;
    hvmSynData.model_info.comm.data_type  = allSimModels[0].data_type;
    hvmSynData.model_info.comm.data_count = allSimModels[0].data_count;
    hvmSynData.model_info.all2AllDataDes.sendType = allSimModels[0].all2AllDataDes.sendType;
    hvmSynData.model_info.all2AllDataDes.recvType = allSimModels[0].all2AllDataDes.recvType;
    hvmSynData.model_info.all2AllDataDes.sendCount = allSimModels[0].all2AllDataDes.sendCount;
    hvmSynData.model_info.all2AllDataDes.recvCount = allSimModels[0].all2AllDataDes.recvCount;
    hvmSynData.model_info.all2AllDataDes.count = allSimModels[0].all2AllDataDes.count;
    std::cout<<"[INFO][CreateSimSynData]: "<<hvmSynData.model_info.all2AllDataDes.count<<std::endl;
    if (allSimModels[0].op_type == static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV)) {
        std::vector<sim::SimModelData> allSimModelsAll2All = RunnerDB::GetByPred<sim::SimModelData>([](const sim::SimModelData &d) {
            return true;
        });
        sort(allSimModelsAll2All.begin(), allSimModelsAll2All.end(), [](const sim::SimModelData &a, const sim::SimModelData &b) {
            return a.rank_id < b.rank_id;
        });
        for (auto iter: allSimModelsAll2All) {
            for (uint32_t i = 0; i < allSimModels[0].rank_size; i++) {
                std::cout << "zhf-DEBUG-PER-ALL2ALLV: " << iter.rank_id << " -> " << i << " : " << iter.all2AllDataDes.sendCountMatrix[iter.rank_id * allSimModels[0].rank_size + i] << std::endl;
                hvmSynData.model_info.all2AllDataDes.sendCountMatrix.push_back(iter.all2AllDataDes.sendCountMatrix[iter.rank_id * allSimModels[0].rank_size + i]);
            }
        }
    } else {
        std::cout<<"zhf-DEBUG: "<<hvmSynData.model_info.all2AllDataDes.count<<std::endl;
        for (uint32_t i = 0; i < hvmSynData.model_info.all2AllDataDes.count; i++) {
            std::cout<<"zhf-DEBUG-PER: "<<allSimModels[0].all2AllDataDes.sendCountMatrix[i]<<std::endl;
            hvmSynData.model_info.all2AllDataDes.sendCountMatrix.push_back(allSimModels[0].all2AllDataDes.sendCountMatrix[i]);
        }
    }

    // ChannelMap
    std::vector<sim::CcuChannel> allChannels; 
    allChannels = RunnerDB::GetByPred<sim::CcuChannel>([](const sim::CcuChannel &d) {
        return true;
    });
    hvmSynData.channel_info.count = allChannels.size();
    for (auto &channel : allChannels) {
        ChannelData chData;
        chData.channelId = channel.channel_id;
        chData.srcRank = channel.src_rank;
        chData.dstRank = channel.dst_rank;
        chData.srcDieId = channel.src_die;
        chData.dstDieId = channel.dst_die;
        // auto ret = GetChannelDataByEPKey(channel.end_point_pair_id, chData);
        // if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        //     return ret;
        // }
        hvmSynData.channel_info.data.push_back(chData);
    }

    // MemoryLayout
    std::vector<sim::MemoryLayout> allMemLayouts; 
    allMemLayouts = RunnerDB::GetByPred<sim::MemoryLayout>([](const sim::MemoryLayout &d) {
        return true;
    });
    hvmSynData.memory_info.count = allMemLayouts.size();
    for (auto &memLayout : allMemLayouts) {
        MemLayoutData memLayoutData;
        memLayoutData.rank_id       = memLayout.rank_id;
        memLayoutData.buffer_type   = memLayout.buf_type;
        memLayoutData.start_addr    = memLayout.base_addr;
        memLayoutData.size          = memLayout.size;
        memLayoutData.global_offset = memLayout.global_offset;
        hvmSynData.memory_info.data.push_back(memLayoutData);
    }

    std::cout<<"[INFO][Create memory info] "<<hvmSynData.memory_info.count<<", "<<hvmSynData.memory_info.data.size()<<std::endl;

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpHcclVmSynthesisData(const std::string &dataId)
{
    std::cout<<"[INFO][DumpHcclVmSynthesisData] Start dumping hccl vm synthesis data..."<<std::endl;
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_SYN_DATA_FILE.c_str(), dataId.c_str());

    // 假设 FindRootPath() 已经实现并返回插件根目录
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

    FILE *fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        std::cout << "[ERROR][DumpHcclVmSynthesisData] Open file failed: "<<fullPath <<", error="<<strerror(errno)<< std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 2. 构造hccl vm synthesis数据
    HcclVmSynData hvmSynData;
    auto ret = CreateSimSynData(hvmSynData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        std::cout << "[ERROR][DumpHcclVmSynthesisData] Get hccl vm synthesis data failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 3. dump为二进制文件
    ret = HcclVmSynDataWrite(fp, hvmSynData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        std::cout << "[ERROR][DumpHcclVmSynthesisData] Write hccl vm synthesis file failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fclose(fp);
    std::cout << "[INFO][DumpHcclVmSynthesisData] Write hccl vm synthesis file success. " << std::endl;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateSimInstrData(HcclVmInstrData &hvmInstrData)
{
    std::cout<<"[CreateSimInstrData] Start create simulator instruction data..."<<std::endl;

    // Instructions
    std::vector<sim::CcuResource> allCcuRes; 
    allCcuRes = RunnerDB::GetByPred<sim::CcuResource>([](const sim::CcuResource &d) {
        return d.state == 1;
    });

    // header
    hvmInstrData.header.magic = HCCLVM_INSTR_FILE_MAGIC;
    hvmInstrData.header.version = 1;
    hvmInstrData.header.header_size = 20;
    hvmInstrData.header.count = allCcuRes.size(); // CCU有微码指令的个数

    for (auto &ccuRes : allCcuRes) {
        MicrocodeInstrInner mcInstr;
        auto ccu = RunnerDB::GetById<sim::Ccu>(ccuRes.ccu_id);
        if (!ccu.has_value()) {
            printf("[ERROR][CreateSimInstrData] can not find ccu by key:%lu\n", ccuRes.ccu_id);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }

        auto devKey = ccu->device_id;
        auto rank = RunnerDB::GetOneByPred<sim::Rank>([devKey](const sim::Rank& r) {
            std::cout<<"zhf-debug: "<<r.device_id<<", "<<devKey<<std::endl;
            return r.device_id == devKey;
        });
        if (!rank.second) {
            HCCL_VM_ERROR("[CreateSimInstrData] can not find any rank by device key: {}", devKey);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        std::cout<<"zhf-debgu: find rank success...."<<std::endl;

        mcInstr.desc.rank_id = rank.first.rank_id;
        mcInstr.desc.die_id  = ccu->die_id;
        mcInstr.desc.count   = ccuRes.instr_cnt;

        mcInstr.data.resize(mcInstr.desc.count);
        auto size = sizeof(hcomm::CcuRep::CcuInstr) * mcInstr.desc.count;
        memcpy(mcInstr.data.data(), ccuRes.instr_space, size);

        hvmInstrData.instr_data.push_back(mcInstr);
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpHcclVmInstrData(const std::string &dataId)
{
    std::cout<<"[INFO][DumpHcclVmInstrData] Start dumping hccl vm instruction data..."<<std::endl;
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_INSTR_DATA_FILE.c_str(), dataId.c_str());
    
    // 假设 FindRootPath() 已经实现并返回插件根目录
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

    FILE *fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        std::cout << "[ERROR][DumpHcclVmInstrData] Open file failed: "<<rootPath << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 2. 构造hccl vm instruction数据
    HcclVmInstrData hvmInstrData;
    auto ret = CreateSimInstrData(hvmInstrData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        std::cout << "[ERROR][DumpHcclVmInstrData] Get hccl vm instruction data failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 3. dump为二进制文件
    ret = HcclVmInstrDataWrite(fp, hvmInstrData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        std::cout << "[ERROR][DumpHcclVmInstrData] Write hccl vm instruction file failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fclose(fp);
    std::cout << "[INFO][DumpHcclVmInstrData] Write hccl vm instruction file success. " << std::endl;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateSimTaskMetaData(HcclVmTaskMetaData &hvmTaskMetaData)
{
    std::cout<<"[CreateSimTaskMetaData] Start create simulator instruction data..."<<std::endl;

    // 1. 获取原始数据
    uint32_t taskCount = 0;
    // 使用 std::unique_ptr 自动管理内存，防止 new 导致的内存泄漏
    std::unique_ptr<HcclTaskMetaData[]> tasks(new HcclTaskMetaData[SHM_TASK_COLLECTION_LENGTH]);
    
    HcclVmResult ret = GetTaskCollection(tasks.get(), &taskCount);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        std::cout << "[CreateSimTaskMetaData] failed to get Task Collection from SHM." << std::endl;
        return ret;
    }

    // header
    hvmTaskMetaData.header.magic = HCCLVM_TASK_FILE_MAGIC;
    hvmTaskMetaData.header.version = 1;
    hvmTaskMetaData.header.header_size = 20;
    hvmTaskMetaData.header.count = taskCount; // task个数

    for (uint32_t idx = 0; idx < taskCount; idx++) {
        hvmTaskMetaData.task_meta.push_back(tasks[idx]);
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpHcclVmTask(const std::string &dataId)
{
    std::cout<<"[INFO][DumpHcclVmTask] Start dumping hccl vm task data..."<<std::endl;
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_TASK_DATA_FILE.c_str(), dataId.c_str());
    
    // 假设 FindRootPath() 已经实现并返回插件根目录
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

    FILE *fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        std::cout << "[ERROR][DumpHcclVmTask] Open file failed: "<<rootPath << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 2. 构造hccl vm task meta数据
    HcclVmTaskMetaData hvmTaskMeta;
    auto ret = CreateSimTaskMetaData(hvmTaskMeta);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        std::cout << "[ERROR][DumpHcclVmTask] Get hccl vm task data failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 3. dump为二进制文件
    ret = HcclVmTaskMetaDataWrite(fp, hvmTaskMeta);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        std::cout << "[ERROR][DumpHcclVmTask] Write hccl vm task file failed. " << std::endl;
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fclose(fp);
    std::cout << "[INFO][DumpHcclVmTask] Write hccl vm task file success. " << std::endl;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpModel(std::string dataId)
{
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), MODEL_FILE.c_str(), dataId.c_str());
    
    // 假设 FindRootPath() 已经实现并返回插件根目录
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

    // 2. 打开 Gzip 文件进行写入 ("wb" 表示二进制写入模式)
    gzFile file = gzopen(fullPath.c_str(), "wb");
    if (!file) {
        HCCL_VM_ERROR("[StorageManager] Failed to create model file: {}", fullPath);
        return HcclVmResult::HCCL_SIM_E_OPEN_FILE_FAILURE;
    }

    // 3. 将结构体转换为 JSON 对象
    // 使用 value 映射，确保即使某些字段是默认值也会被记录

    uint32_t rankSize = 0;
    HcclVmResult ret = GetNpuNum(&rankSize);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[StorageManager] failed to get rankSize.");
    }

    nlohmann::json j;
    j["cmdType"]    = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_SCATTER);
    j["rankSize"]   = rankSize;
    j["dataType"]   = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_FP32);
    j["dataCount"]  = 64 * 1024 * 1024 / 4 / rankSize;
    j["reduceType"] = static_cast<uint32_t>(HcclReduceOp::HCCL_REDUCE_SUM);
    j["srcRank"]    = 0;
    j["dstRank"]    = 0;
    j["root"]       = 0;

    // 4. 序列化为字符串并添加换行符 (JSONL 格式)
    std::string content = j.dump() + "\n";

    // 5. 写入并关闭
    int res = gzwrite(file, content.c_str(), static_cast<unsigned int>(content.length()));
    if (res <= 0) {
        int errnum;
        HCCL_VM_ERROR("[StorageManager] gzwrite error: {}", gzerror(file, &errnum));
        gzclose(file);
        return HcclVmResult::HCCL_SIM_E_OPEN_FILE_FAILURE;
    }

    gzclose(file);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpMemLayout(std::string dataId)
{
    std::vector<DumpMemBlock> memLayout;
    HcclVmResult ret = GetMemLayout(memLayout);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[DumpMemLayout] failed to get Mem Layout from SHM.");
        return ret;
    }

    char fileName[256];
    snprintf(fileName, sizeof(fileName), MEM_LAYOUT_FILE.c_str(), dataId.c_str());
    
    // 路径处理逻辑
    std::string rootPath = GetBinLocation(); 
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

    // 3. 打开文件
    gzFile file = gzopen(fullPath.c_str(), "wb");
    if (!file) {
        HCCL_VM_ERROR("[DumpMemLayout] Failed to open file for writing: {}", fullPath);
        return HcclVmResult::HCCL_SIM_E_OPEN_FILE_FAILURE;
    }

    // 4. 写入 Header
    // 注意：这里应该是 memLayout 的大小
    nlohmann::json header;
    header["total_count"] = static_cast<uint32_t>(memLayout.size());
    header["cols"] = {"rankId", "startAddr", "size", "type"}; // 增加可读性
    
    std::string headerStr = header.dump() + "\n";
    gzwrite(file, headerStr.c_str(), headerStr.length());

    // 5. 遍历并写入数据行
    // 格式必须对应 LoadMemLayout 中的: [rid, sAddr, size, type]
    for (const auto& block : memLayout) {
        nlohmann::json row = nlohmann::json::array();
        row.push_back(block.rankId);
        row.push_back(block.startAddr);
        row.push_back(block.size);
        row.push_back(block.bufferType);

        std::string line = row.dump() + "\n";
        if (gzwrite(file, line.c_str(), line.length()) <= 0) {
            HCCL_VM_ERROR("[DumpMemLayout] gzwrite error during layout dump.");
            gzclose(file);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    gzclose(file);
    HCCL_VM_INFO("[DumpMemLayout] Success. Saved {} blocks.", memLayout.size());
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpTask(std::string dataId)
{
    // 1. 获取原始数据
    uint32_t taskCount = 0;
    // 使用 std::unique_ptr 自动管理内存，防止 new 导致的内存泄漏
    std::unique_ptr<HcclTaskMetaData[]> tasks(new HcclTaskMetaData[SHM_TASK_COLLECTION_LENGTH]);
    
    HcclVmResult ret = GetTaskCollection(tasks.get(), &taskCount);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("[DumpTask] failed to get Task Collection from SHM.");
        return ret;
    }

    // 2. 构造文件路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), TASK_COLLECTION_FILE.c_str(), dataId.c_str());
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

    // 3. 打开文件
    gzFile file = gzopen(fullPath.c_str(), "wb"); // 写二进制模式
    if (!file) {
        HCCL_VM_ERROR("[DumpTask] Failed to open file for writing: {}", fullPath);
        return HcclVmResult::HCCL_SIM_E_OPEN_FILE_FAILURE;
    }

    // 4. 写入 Header (对应解析逻辑中的 header.contains("total_count"))
    nlohmann::json header;
    header["total_count"] = taskCount;
    header["cols"] = {
        "taskType",    // 0
        "commId",      // 1
        "rankId",      // 2
        "streamId",    // 3
        "srcRankId",   // 4 (MEM_CPY / REDUCE / NOTIFY)
        "srcOffset",   // 5 (MEM_CPY / REDUCE)
        "dstRankId",   // 6 (MEM_CPY / REDUCE / NOTIFY)
        "dstOffset",   // 7 (MEM_CPY / REDUCE)
        "len",         // 8 (MEM_CPY)
        "dataType",    // 9 (REDUCE)
        "dataCount",   // 10 (REDUCE)
        "reduceOp",    // 11 (REDUCE)
        "notifyId",    // 12 (NOTIFY)
        "notifyCount", // 13 (NOTIFY)
        "protocol"     // 14 (Common)
    };
    std::string headerStr = header.dump() + "\n";
    gzwrite(file, headerStr.c_str(), headerStr.length());

    // 5. 遍历并写入 Task 数据
    for (uint32_t i = 0; i < taskCount; ++i) {
        const auto& meta = tasks[i];
        
        // 初始化一个有15个元素的空数组 [0...14]
        // 必须初始化为0，确保即便某些字段没赋值，JSON数组长度也固定为15
        nlohmann::json row = nlohmann::json::array({0,0,0,0,0,0,0,0,0,0,0,0,0,0,0});

        // 公共基础字段 (0-3)
        row[0] = static_cast<int>(meta.taskType);
        row[1] = meta.commId;
        row[2] = meta.rankId;
        row[3] = meta.streamId;

        // 根据类型填充特定索引 (需与解析逻辑 j[4]~j[14] 严格对应)
        if (meta.taskType == HccLTaskMetaType::MEM_CPY) {
            row[4] = meta.taskData.transMem.srcRankId;
            row[5] = meta.taskData.transMem.srcOffset;
            row[6] = meta.taskData.transMem.dstRankId;
            row[7] = meta.taskData.transMem.dstOffset;
            row[8] = meta.taskData.transMem.len;
            row[14] = meta.taskData.transMem.protocol;
        } 
        else if (meta.taskType == HccLTaskMetaType::REDUCE) {
            row[4] = meta.taskData.reduce.srcRankId;
            row[5] = meta.taskData.reduce.srcOffset;
            row[6] = meta.taskData.reduce.dstRankId;
            row[7] = meta.taskData.reduce.dstOffset;
            row[9] = meta.taskData.reduce.dataType;
            row[10] = meta.taskData.reduce.dataCount;
            row[11] = meta.taskData.reduce.reduceOp;
            row[14] = meta.taskData.reduce.protocol;
        } 
        else if (meta.taskType == HccLTaskMetaType::NOTIFY_RECORD || 
                 meta.taskType == HccLTaskMetaType::NOTIFY_WAIT) {
            row[4] = meta.taskData.notify.srcRankId;
            row[6] = meta.taskData.notify.dstRankId;
            row[12] = meta.taskData.notify.notifyId;
            row[13] = meta.taskData.notify.notifyCount;
            row[14] = meta.taskData.notify.protocol;
        }

        // 写入一行
        std::string line = row.dump() + "\n";
        if (gzwrite(file, line.c_str(), line.length()) <= 0) {
            HCCL_VM_ERROR("[DumpTask] gzwrite error at index {}", i);
            gzclose(file);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    gzclose(file);
    HCCL_VM_INFO("[DumpTask] Success. Saved {} tasks to {}", taskCount, fullPath);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}