/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 本程序是自由软件，您可以根据CANN开源软件许可证2.0版（"许可证"）的条款和条件重新分发和/或修改它。
 * 请参阅许可证了解详情。未经许可证授权，您不得使用本文件。
 * 本软件按"原样"提供，不附带任何明示或暗示的担保，包括但不限于非侵权、适销性或特定用途适用性。
 * 请参阅软件存储库根目录中的LICENSE以获取许可证全文。
 */

// 包含本类的头文件声明
#include "ins_v2_reduce_scatter_order_preserved_executor.h"
// 包含保序ReduceScatter模板类的头文件
#include "ins_temp_reduce_scatter_order_preserved_level1.h"
// 包含环境配置
#include "alg_env_config.h"
// 包含数学函数库，用于ceil、floor等计算
#include <cmath>
// 包含算法库，用于std::min、std::max等函数
#include <algorithm>

// 使用ops_hccl命名空间
namespace ops_hccl {

// 构造函数：初始化保序ReduceScatter执行器
// AlgTopoMatch: 拓扑匹配类型
// InsAlgTemplate: 算法模板类型
template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2ReduceScatterOrderPreservedExecutor()
{
    // 初始化严格确定性模式标志为true（保序模式默认启用）
    deterministicStrict_ = true;
}

// 计算算法层级信息：确定算法在通信层级中的位置和子通信域
// comm: HCCL通信句柄
// topoInfo: 拓扑信息，包含网络层详细信息
// algHierarchyInfo: 输出参数，填充算法层级信息
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 以下注释掉的代码是备用的rank和设备类型初始化
    // myRank_ = topoInfo->userRank;
    // rankSize_ = topoInfo->userRankSize;
    // devType_ = topoInfo->deviceType;
    // 创建拓扑匹配对象
    AlgTopoMatch topoMatch;
    // 调用拓扑匹配函数，根据拓扑信息计算算法层级
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    // 打印调试日志，显示当前rank和rank总数 zdy, 打印为赋值
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][CalcAlgHierarchyInfo] myRank[%u], rankSize[%u]",
        myRank_, rankSize_);
    // 返回成功
    return HCCL_SUCCESS;
}

// 计算资源请求：确定执行所需的线程、通知和通道资源
// comm: HCCL通信句柄
// param: 操作参数
// topoInfo: 拓扑信息
// algHierarchyInfo: 算法层级信息
// resourceRequest: 输出参数，填充资源请求信息
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    // 创建算法模板实例，传入操作参数、用户rank和算法层级信息
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    // 调用模板的CalcRes函数计算所需资源
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    // 打印调试日志，显示从线程数和主线程通知数
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][CalcRes] slaveThreadNum[%u], notifyNumOnMainThread[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    // 返回成功
    return HCCL_SUCCESS;
}

// 编排执行：协调ReduceScatter操作的执行流程
// param: 操作参数，包含输入输出指针、数据类型等信息
// resCtx: 资源上下文，包含线程、通道、临时缓冲区等资源
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    // 打印执行开始日志
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][Orchestrate] Start");

    // 从资源上下文获取当前rank编号
    myRank_ = resCtx.topoInfo.userRank;
    // 从资源上下文获取rank总数
    rankSize_ = resCtx.topoInfo.userRankSize;
    // 从操作参数获取数据元素个数
    dataCount_ = param.DataDes.count;
    // 根据数据类型获取数据类型大小（字节）
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    // 计算总数据大小（字节）= 元素个数 * 数据类型大小
    dataSize_ = dataCount_ * dataTypeSize_;
    // 保存数据类型
    dataType_ = param.DataDes.dataType;
    // 保存归约操作类型
    reduceOp_ = param.reduceType;
    // 保存线程句柄列表
    threads_ = resCtx.threads;
    // 注释掉的代码：获取AICPU展开模式配置
    //aicpuUnfoldMode_ = GetExternalInputHcclAicpuUnfold();

    // 获取临时缓冲区大小
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 如果不是AIV或CCU引擎，需要恢复通道映射
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        // 从资源上下文恢复通道映射到remoteRankToChannelInfo_
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    // 初始化执行器信息（检查是否启用严格模式）
    InitExecutorInfo(param);
    // // 计算每个数据块的大小
    // CalcSizePerBlock(param);
    // // 计算每个rank的数据切片大小
    // CalcGroupSlices(param);

    // 调用OrchestrateLoop执行核心循环逻辑
    HcclResult ret = OrchestrateLoop(param, resCtx);
    // 检查返回结果，如果失败则打印错误日志
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2ReduceScatterOrderPreservedExecutor][Orchestrate] kernel run failed, err[0x%016llx]",
            HCCL_ERROR_CODE(ret)), ret);
    // 返回成功
    return HCCL_SUCCESS;
}

// 编排循环执行：执行ReduceScatter操作的核心循环逻辑
// param: 操作参数
// resCtx: 资源上下文
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    // 打印循环执行开始日志，显示严格确定性模式标志
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] Start, deterministicStrict[%d]",
        deterministicStrict_);

    // 准备模板资源结构体
    TemplateResource templateAlgRes;
    // 如果存在通道映射，获取第一层通道信息
    if (remoteRankToChannelInfo_.size() > 0) {
        // 将第一层通道映射赋值给模板资源
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    // 获取线程句柄列表
    templateAlgRes.threads = resCtx.threads;
    // 获取AIV通信信息指针
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    // 准备模板数据参数结构体
    TemplateDataParams tempAlgParams;
    // 设置输入缓冲区指针
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    // 设置输出缓冲区指针
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    // 设置输入缓冲区大小
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    // 设置输出缓冲区大小
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    // 设置临时缓冲区信息
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    // 设置输入缓冲区类型为INPUT
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    // 设置输出缓冲区类型为OUTPUT
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    // 设置临时缓冲区类型为HCCL_BUFFER
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // 创建算法模板实例
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);

    // 计算临时缓冲区倍数（用于确定单次循环最大数据量）
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType,
                                                                     tempAlgParams.buffInfo.outBuffType);
    // 初始化单次循环最大数据大小
    u64 maxDataSizePerLoop = 0;
    // 获取临时缓冲区大小
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    // 设置传输边界数据大小（UB最大数据大小）
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    // 打印临时缓冲区大小日志
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor]maxTmpMemSize_ [%llu]", maxTmpMemSize_);
    // 如果临时缓冲区倍数不为0，根据缓冲区大小计算单次循环最大数据量
    if (templateScratchMultiplier != 0) {
        // 计算基于缓冲区的数据量边界：缓冲区大小 / 倍数，并按对齐要求向下取整
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED
            * HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED;
        // 取传输边界和缓冲区边界的较小值作为单次循环最大数据量
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        // 如果倍数为0，单次循环最大数据量等于传输边界
        maxDataSizePerLoop = transportBoundDataSize;
    }
    // 计算单次循环最大数据元素个数
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_;
    // 打印单次循环数据量计算结果日志
    HCCL_INFO(
        "[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    // 检查单次循环最大数据元素个数是否为0，如果为0则返回错误
    CHK_PRT_RET(maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);

    // 计算循环次数：总数据量 / 单次最大数据量，向上取整
    // u64 loopTimes = (dataCount_ * rankSize_) / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    // 设置是否启用远端内存访问（OFFLOAD模式下启用）
    tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
    // 初始化已处理的数据元素个数
    u64 processedDataCount = 0;
    // zdy 增加dataCount_、loopTimes打印
    HCCL_INFO("zdy [InsV2ReduceScatterOrderPreservedExecutor] dataCount_[%llu][%u]loopTimes[%u]\n", dataCount_, maxDataCountPerLoop, loopTimes);

    // 循环执行，每次处理一部分数据
    for (u64 loop = 0; loop < loopTimes; loop++) {
        // 计算当前循环处理的数据元素个数
        // 如果是最后一次循环，处理剩余数据；否则处理单次最大数据量
        u64 currDataCount = (loop == loopTimes - 1) ? dataCount_ - processedDataCount : maxDataCountPerLoop;
        HCCL_INFO("zdy [InsV2ReduceScatterOrderPreservedExecutor] currDataCount[%llu]\n", currDataCount);
        // 设置当前循环的数据元素个数
        tempAlgParams.count = currDataCount;
        // 计算输入缓冲区基址偏移（已处理数据的偏移）
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        // 计算输出缓冲区基址偏移（已处理数据的偏移）
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        // 设置临时缓冲区基址偏移为0（每次循环重新使用临时缓冲区）
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;

        // 计算当前循环的总数据大小（字节）
        u64 currDataSize = currDataCount * dataTypeSize_;

        // 计算当前循环每个rank的数据块大小（向上取整到rankSize份）
        // u64 currSizePerBlock = (currDataCount + rankSize_ - 1) / rankSize_ * dataTypeSize_;
        u64 currSizePerBlock = currDataSize;
        HCCL_INFO("zdy [InsV2ReduceScatterOrderPreservedExecutor] currSizePerBlock[%llu][%u]\n", currSizePerBlock, rankSize_);
        // 将数据块大小向上对齐到最小切片对齐要求
        //currSizePerBlock = RoundUpWithDivisor(currSizePerBlock, HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED);
        HCCL_INFO("zdy [InsV2ReduceScatterOrderPreservedExecutor] currSizePerBlock[%llu]\n", currSizePerBlock);

        // 创建当前循环每个rank的数据大小向量
        std::vector<u64> currGroupSize;
        // 初始化剩余数据大小
        u64 sizeRemain = currDataSize;
        // 遍历每个rank，计算其数据切片大小
        for (u32 rankId = 0; rankId < rankSize_; rankId++) {
            // 计算当前rank的数据大小：取剩余数据和数据块大小的较小值
            // u64 size = (sizeRemain > currSizePerBlock) ? currSizePerBlock : sizeRemain;
            // // 将数据大小添加到向量
            // currGroupSize.push_back(size);
            // // 减少剩余数据大小
            // sizeRemain -= size;
            // currGroupSize.push_back(currSizePerBlock);
        }

        // 创建内存块信息结构体，用于存储每个rank的数据大小和偏移
        MemBlockInfo memBlockInfo;
        // 清空数据大小向量
        memBlockInfo.size.clear();
        // 清空用户输入偏移向量
        memBlockInfo.userInputOffsets.clear();
        // 清空输入偏移向量
        memBlockInfo.inputOffsets.clear();
        // 清空输出偏移向量
        memBlockInfo.outputOffsets.clear();
        // 设置数据类型大小作为单位大小
        u64 unitSize = dataTypeSize_;
        // 遍历每个数据块，计算其偏移量
        for (u32 dataId = 0; dataId < rankSize_; dataId++) {
            // 计算当前数据块在临时缓冲区中的偏移
            u64 offset = dataId * currSizePerBlock;
            // 计算当前数据块在用户输入缓冲区中的偏移
            // 基址偏移 + 数据块ID * 当前循环数据量 * 单位大小
            // u64 userMemInOffset = processedDataCount * unitSize + dataId * unitSize * currDataCount;
            // u64 userMemInOffset = processedDataCount * unitSize + offset;
            u64 userMemInOffset = processedDataCount * unitSize + dataId * dataSize_;
            // 添加当前数据块的大小
            memBlockInfo.size.push_back(currSizePerBlock);
            // 添加用户输入偏移
            memBlockInfo.userInputOffsets.push_back(userMemInOffset);
            // 添加输入偏移（临时缓冲区）
            memBlockInfo.inputOffsets.push_back(offset);
            // 添加输出偏移（临时缓冲区）
            memBlockInfo.outputOffsets.push_back(offset);
        }

        // 设置所有rank的数据切片大小向量
        tempAlgParams.allRankSliceSize = currGroupSize;
        // 设置本rank的数据切片大小
        // tempAlgParams.sliceSize = currGroupSize[myRank_ % rankSize_];
        tempAlgParams.sliceSize = currDataSize;
        // 设置尾部大小等于切片大小
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        // 设置输入切片步幅（总数据大小，用于多份输入数据的间隔）
        tempAlgParams.inputSliceStride = dataSize_;
        // 设置输出切片步幅为0（输出只有一份）
        tempAlgParams.outputSliceStride = 0;
        // 设置重复次数为1（不重复）
        tempAlgParams.repeatNum = 1;
        // 设置输入重复步幅为0
        tempAlgParams.inputRepeatStride = 0;
        // 设置输出重复步幅为0
        tempAlgParams.outputRepeatStride = 0;

        // 打印当前循环的调试日志
        HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor] loop[%llu] sliceSize[%llu], "
            "inBuffBaseOff[%llu], outBuffBaseOff[%llu]",
            loop, tempAlgParams.sliceSize, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);

        // 将内存块信息设置到算法模板
        algTemplate->SetMemBlockInfo(memBlockInfo);
        // 调用算法模板的KernelRun函数执行核心算法
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        // 更新已处理的数据元素个数
        processedDataCount += currDataCount;
    }

    // 打印循环执行结束日志
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][OrchestrateLoop] End");
    // 返回成功
    return HCCL_SUCCESS;
}

// 初始化执行器信息：检查和设置严格确定性模式
// param: 操作参数
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::InitExecutorInfo(const OpParam &param)
{
    // 检查是否需要启用严格确定性模式
    deterministicStrict_ = IsNeedStrictMode(param);
    // 打印严格确定性模式标志日志
    HCCL_INFO("[InsV2ReduceScatterOrderPreservedExecutor][InitExecutorInfo] deterministicStrict[%d]",
        deterministicStrict_);
    // 返回成功
    return HCCL_SUCCESS;
}

// 向上对齐到除数的倍数
// value: 需要对齐的值
// divisor: 除数（对齐边界）
// 返回值: 对齐后的值
template <typename AlgTopoMatch, typename InsAlgTemplate>
u64 InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::RoundUpWithDivisor(
    u64 value, u64 divisor) const
{
    // 如果值为0或除数为0，返回除数本身
    if (value == 0 || divisor == 0) {
        return divisor;
    }
    // 向上对齐计算：(value + divisor - 1) / divisor * divisor
    return ((value + (divisor - 1)) / divisor) * divisor;
}

// // 计算每个数据块的大小：确定每个rank需要处理的数据量
// // param: 操作参数
// template <typename AlgTopoMatch, typename InsAlgTemplate>
// HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcSizePerBlock(const OpParam &param)
// {
//     // 计算单卡数据量：总数据量 / rank数，向上取整
//     u64 sizePerBlock = (dataCount_ + rankSize_ - 1) / rankSize_ * dataTypeSize_;
//     // 将数据块大小向上对齐到最小切片对齐要求
//     // memInfo_.sizePerBlock = RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED);
//     // memInfo_.sizePerBlock = sizePerBlock;
//     // 初始化all2all偏移量为0（用于AllToAll操作）
//     memInfo_.all2allOffset = 0;
//     // 初始化scratch内存标志为false（不使用scratch内存）
//     memInfo_.scratchMemFlag = false;
//     // 初始化总数据大小为0
//     memInfo_.totalSize = 0;

//     // 打印数据块大小计算结果日志
//     HCCL_INFO("[CalcSizePerBlock] sizePerBlock[%llu], dataCount[%llu], rankSize[%u]",
//         memInfo_.sizePerBlock, dataCount_, rankSize_);
//     // 返回成功
//     return HCCL_SUCCESS;
// }

// // 计算数据切片分组：确定每个rank实际分配的数据切片大小
// // param: 操作参数
// template <typename AlgTopoMatch, typename InsAlgTemplate>
// HcclResult InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::CalcGroupSlices(const OpParam &param)
// {
//     // 清空数据切片大小向量
//     memInfo_.groupSize.clear();
//     // 初始化剩余数据大小为总数据大小
//     u64 sizeRemain = dataSize_;
//     // 遍历每个rank，分配数据切片大小
//     for (u32 rankId = 0; rankId < rankSize_; rankId++) {
//         // 计算当前rank的数据切片大小：取剩余数据和数据块大小的较小值
//         u64 size = (sizeRemain > memInfo_.sizePerBlock) ? memInfo_.sizePerBlock : sizeRemain;
//         // 将数据切片大小添加到向量
//         memInfo_.groupSize.push_back(size);
//         // 减少剩余数据大小
//         sizeRemain -= size;
//     }
//     // 计算总数据大小：取数据块大小 * rank数和实际数据大小的较大值
//     memInfo_.totalSize = std::max(memInfo_.sizePerBlock * rankSize_, dataSize_);
//     // 打印数据切片分组计算结果日志
//     HCCL_INFO("[CalcGroupSlices] groupSize.size[%u], totalSize[%llu]",
//         memInfo_.groupSize.size(), memInfo_.totalSize);
//     // 返回成功
//     return HCCL_SUCCESS;
// }

// 检查是否需要严格确定性模式
// param: 操作参数
// 返回值: 如果需要严格模式返回true，否则返回false
template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2ReduceScatterOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplate>::IsNeedStrictMode(const OpParam &param) const
{
    // 严格模式条件：
    // 1. 环境变量HCCL_DETERMINISTIC设置为STRICT
    // 2. 数据类型为FP16、FP32、BFP16或FP64
    // 3. 归约操作为SUM或PROD
    // 4. rank数大于等于最小严格rank数
    bool isStrictMode = (GetExternalInputHcclDeterministic() == static_cast<u8>(DeterministicEnableLevel::DETERMINISTIC_STRICT))
        && (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32 ||
            param.DataDes.dataType == HCCL_DATA_TYPE_BFP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP64)
        && (param.reduceType == HCCL_REDUCE_SUM || param.reduceType == HCCL_REDUCE_PROD)
        && rankSize_ >= MIN_STRICT_RANK_NUM_ORDER_PRESERVED;
    // 返回严格模式标志
    return isStrictMode;
}

// 注册保序ReduceScatter执行器
// HCCL_CMD_REDUCE_SCATTER: 命令类型
// ReduceScatterOrderPreserved: 算法名称
// InsV2ReduceScatterOrderPreservedExecutor: 执行器类型
// TopoMatch1D: 拓扑匹配类型
// InsTempReduceScatterOrderPreservedLevel1: 模板类型
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, ReduceScatterOrderPreserved,
    InsV2ReduceScatterOrderPreservedExecutor, TopoMatch1D, InsTempReduceScatterOrderPreservedLevel1);

// 命名空间结束
}