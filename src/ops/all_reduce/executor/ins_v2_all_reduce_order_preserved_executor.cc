/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 本程序是自由软件，您可以根据CANN开源软件许可证2.0版（"许可证"）的条款和条件重新分发和/或修改它。
 * 请参阅许可证了解详情。未经许可证授权，您不得使用本文件。
 * 本软件按"原样"提供，不附带任何明示或暗示的担保，包括但不限于非侵权、适销性或特定用途适用性。
 * 请参阅软件存储库根目录中的LICENSE以获取许可证全文。
 */

// 包含本类的头文件声明
#include "ins_v2_all_reduce_order_preserved_executor.h"
// 包含保序ReduceScatter模板类的头文件
#include "ins_temp_reduce_scatter_order_preserved_level1.h"
// 包含AllGather NHR模板类的头文件
#include "ins_temp_all_gather_mesh_1D.h"
// 包含数学函数库，用于ceil、floor等计算
#include <cmath>
// 包含算法库，用于std::min、std::max等函数
#include <algorithm>

// 使用ops_hccl命名空间
namespace ops_hccl {

// 构造函数：初始化保序AllReduce执行器
// AlgTopoMatch: 拓扑匹配类型
// InsAlgTemplateRS: ReduceScatter算法模板类型
// InsAlgTemplateAG: AllGather算法模板类型
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InsV2AllReduceOrderPreservedExecutor()
{
    // 初始化严格确定性模式标志为true（保序模式默认启用）
    deterministicStrict_ = true;
}

// 计算算法层级信息：确定算法在通信层级中的位置和子通信域
// comm: HCCL通信句柄
// topoInfo: 拓扑信息，包含网络层详细信息
// algHierarchyInfo: 输出参数，填充算法层级信息
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcAlgHierarchyInfo(
    HcclComm comm, TopoInfoWithNetLayerDetails *topoInfo, AlgHierarchyInfoForAllLevel &algHierarchyInfo)
{
    // 从拓扑信息获取当前rank编号
    myRank_ = topoInfo->userRank;
    // 从拓扑信息获取rank总数
    rankSize_ = topoInfo->userRankSize;
    // 从拓扑信息获取设备类型
    devType_ = topoInfo->deviceType;
    // 创建拓扑匹配对象
    AlgTopoMatch topoMatch;
    // 调用拓扑匹配函数，根据拓扑信息计算算法层级
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    // 打印调试日志，显示当前rank和rank总数（flat level1模式）
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcAlgHierarchyInfo] myRank[%u], rankSize[%u] (flat level1 only)",
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
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    const AlgHierarchyInfoForAllLevel &algHierarchyInfo, AlgResourceRequest &resourceRequest)
{
    // 从拓扑信息获取当前rank编号
    myRank_ = topoInfo->userRank;
    // 从拓扑信息获取rank总数
    rankSize_ = topoInfo->userRankSize;
    // 从拓扑信息获取设备类型
    devType_ = topoInfo->deviceType;
    // 保存归约操作类型
    reduceOp_ = param.reduceType;
    // 保存数据类型
    dataType_ = param.DataDes.dataType;
    // 保存数据元素个数
    dataCount_ = param.DataDes.count;
    // 根据数据类型获取数据类型大小（字节）
    dataTypeSize_ = SIZE_TABLE[param.DataDes.dataType];
    // 获取AICPU展开模式配置
    aicpuUnfoldMode_ = GetExternalInputHcclAicpuUnfold();

    // 保存算法层级信息
    algHierarchyInfo_ = algHierarchyInfo;

    // 初始化执行器信息（检查是否启用严格模式）
    InitExecutorInfo(param);
    // 计算每个数据块的大小
    CalcSizePerBlock(param);
    // 计算每个rank的数据切片大小
    CalcGroupSlices(param);

    // 创建ReduceScatter算法模板实例
    std::shared_ptr<InsAlgTemplateRS> rsTempAlg =
        std::make_shared<InsAlgTemplateRS>(param, myRank_, algHierarchyInfo.infos[0]);

    // 创建AllGather算法模板实例
    std::shared_ptr<InsAlgTemplateAG> agTempAlg =
        std::make_shared<InsAlgTemplateAG>(param, myRank_, algHierarchyInfo.infos[0]);

    // 创建ReduceScatter资源请求结构体
    AlgResourceRequest resReqRS;
    // 创建AllGather资源请求结构体
    AlgResourceRequest resReqAG;

    // 调用ReduceScatter模板的CalcRes函数计算所需资源
    CHK_RET(rsTempAlg->CalcRes(comm, param, topoInfo, resReqRS));
    // 调用AllGather模板的CalcRes函数计算所需资源
    CHK_RET(agTempAlg->CalcRes(comm, param, topoInfo, resReqAG));

    // 设置从线程数为两个模板的最大值
    resourceRequest.slaveThreadNum = std::max(resReqRS.slaveThreadNum, resReqAG.slaveThreadNum);

    // 清空每个线程的通知数向量
    resourceRequest.notifyNumPerThread.clear();
    // 调整向量大小为从线程数
    resourceRequest.notifyNumPerThread.resize(resourceRequest.slaveThreadNum);
    // 遍历每个从线程，设置通知数为两个模板的最大值
    for (u32 i = 0; i < resourceRequest.slaveThreadNum; ++i) {
        // 如果ReduceScatter模板有该线程的通知数配置
        if (i < resReqRS.notifyNumPerThread.size()) {
            // 取当前值和ReduceScatter值的较大者
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqRS.notifyNumPerThread[i]);
        }
        // 如果AllGather模板有该线程的通知数配置
        if (i < resReqAG.notifyNumPerThread.size()) {
            // 取当前值和AllGather值的较大者
            resourceRequest.notifyNumPerThread[i] = std::max(resourceRequest.notifyNumPerThread[i],
                resReqAG.notifyNumPerThread[i]);
        }
    }

    // 设置主线程通知数为两个模板的最大值
    resourceRequest.notifyNumOnMainThread = std::max(resReqRS.notifyNumOnMainThread,
        resReqAG.notifyNumOnMainThread);

    // 清空通道向量
    resourceRequest.channels.clear();
    // 如果ReduceScatter模板有通道配置，添加第一层通道
    if (resReqRS.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqRS.channels[0]);
    }
    // 如果AllGather模板有通道配置，添加第一层通道
    if (resReqAG.channels.size() > 0) {
        resourceRequest.channels.push_back(resReqAG.channels[0]);
    }

    // 打印调试日志，显示从线程数和主线程通知数
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][CalcRes] slaveThreadNum[%u], notifyNumOnMainThread[%u]",
        resourceRequest.slaveThreadNum, resourceRequest.notifyNumOnMainThread);
    // 返回成功
    return HCCL_SUCCESS;
}

// 编排执行：协调AllReduce操作的执行流程
// param: 操作参数，包含输入输出指针、数据类型等信息
// resCtx: 资源上下文，包含线程、通道、临时缓冲区等资源
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable& resCtx)
{
    // 打印执行开始日志
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] Start");

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
    // 保存算法层级信息
    algHierarchyInfo_ = resCtx.algHierarchyInfo;
    // 保存线程句柄列表
    threads_ = resCtx.threads;
    // 获取AICPU展开模式配置
    aicpuUnfoldMode_ = GetExternalInputHcclAicpuUnfold();

    // 获取临时缓冲区大小
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 如果不是AIV或CCU引擎，需要恢复通道映射
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        // 从资源上下文恢复通道映射到remoteRankToChannelInfo_
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    // 初始化执行器信息（检查是否启用严格模式）
    InitExecutorInfo(param);
    // 计算每个数据块的大小
    CalcSizePerBlock(param);
    // 计算每个rank的数据切片大小
    CalcGroupSlices(param);

    // 调用OrchestrateLoop执行核心循环逻辑
    HcclResult ret = OrchestrateLoop(param, resCtx);
    // 检查返回结果，如果失败则打印错误日志
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AllReduceOrderPreservedExecutor][Orchestrate] kernel run failed, err[0x%016llx]",
            HCCL_ERROR_CODE(ret)), ret);
    // 返回成功
    return HCCL_SUCCESS;
}

// 初始化模板资源结构体
// resCtx: 资源上下文
// templateAlgRes: 输出的模板资源结构体
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
void InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InitTemplateResource(
    const AlgResourceCtxSerializable &resCtx, TemplateResource &templateAlgRes)
{
    // 如果存在通道映射，获取第一层通道信息
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    // 获取线程句柄列表
    templateAlgRes.threads = resCtx.threads;
    // 获取AIV通信信息指针
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
}

// 初始化模板数据参数结构体
// param: 操作参数
// resCtx: 资源上下文
// tempAlgParams: 输出的模板数据参数结构体
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
void InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InitTemplateDataParams(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx, TemplateDataParams &tempAlgParams)
{
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
    // 设置是否启用远端内存访问（OFFLOAD模式下启用）
    tempAlgParams.enableRemoteMemAccess = param.opMode == OpMode::OFFLOAD;
}

// 打印模板资源结构体信息
// templateAlgRes: 模板资源结构体
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
void InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::PrintTemplateResource(
    const TemplateResource &templateAlgRes)
{
    HCCL_INFO("[PrintTemplateResource] channels.size[%u], threads.size[%u], aivCommInfoPtr[%p]",
        templateAlgRes.channels.size(), templateAlgRes.threads.size(), templateAlgRes.aivCommInfoPtr);
}

// 打印模板数据参数结构体信息
// tempAlgParams: 模板数据参数结构体
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
void InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::PrintTemplateDataParams(
    const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[PrintTemplateDataParams] inputPtr[%p], outputPtr[%p], inputSize[%llu], outputSize[%llu], "
        "hcclBuff.size[%llu], hcclBuff.addr[%p], inBuffType[%d], outBuffType[%d], hcclBuffType[%d], "
        "enableRemoteMemAccess[%d]",
        tempAlgParams.buffInfo.inputPtr, tempAlgParams.buffInfo.outputPtr,
        tempAlgParams.buffInfo.inputSize, tempAlgParams.buffInfo.outputSize,
        tempAlgParams.buffInfo.hcclBuff.size, tempAlgParams.buffInfo.hcclBuff.addr,
        static_cast<int>(tempAlgParams.buffInfo.inBuffType), static_cast<int>(tempAlgParams.buffInfo.outBuffType),
        static_cast<int>(tempAlgParams.buffInfo.hcclBuffType), tempAlgParams.enableRemoteMemAccess);
}

// 编排循环执行：执行AllReduce操作的核心循环逻辑
// param: 操作参数
// resCtx: 资源上下文
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    // 打印循环执行开始日志，显示严格确定性模式标志（flat level1模式）
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] Start, deterministicStrict[%d] (flat level1)",
        deterministicStrict_);

    // 准备模板资源结构体并初始化
    TemplateResource templateAlgRes;
    InitTemplateResource(resCtx, templateAlgRes);
    PrintTemplateResource(templateAlgRes);

    // 准备模板数据参数结构体并初始化
    TemplateDataParams tempAlgParams;
    InitTemplateDataParams(param, resCtx, tempAlgParams);
    PrintTemplateDataParams(tempAlgParams);

    // 创建ReduceScatter算法模板实例
    std::shared_ptr<InsAlgTemplateRS> rsTempAlg =
        std::make_shared<InsAlgTemplateRS>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);

    // 创建AllGather算法模板实例
    std::shared_ptr<InsAlgTemplateAG> agTempAlg =
        std::make_shared<InsAlgTemplateAG>(param, myRank_, resCtx.algHierarchyInfo.infos[0]);

    // 计算ReduceScatter临时缓冲区倍数（用于确定单次循环最大数据量）
    u32 rsScratchMultiplier = rsTempAlg->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType,
                                                              tempAlgParams.buffInfo.outBuffType);
    // 计算AllGather临时缓冲区倍数
    u32 agScratchMultiplier = agTempAlg->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType,
                                                              tempAlgParams.buffInfo.outBuffType);

    // 初始化单次循环最大数据大小
    u64 maxDataSizePerLoop = 0;
    // 获取临时缓冲区大小
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    // 设置传输边界数据大小（UB最大数据大小）
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    // 打印临时缓冲区大小日志
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor]maxTmpMemSize_ [%llu]", maxTmpMemSize_);

    // 取两个模板的缓冲区倍数的较大值，确保两个阶段都能正常运行
    u32 templateScratchMultiplier = std::max(rsScratchMultiplier, agScratchMultiplier);
    templateScratchMultiplier = 0x1;
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
        "[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    // 检查单次循环最大数据元素个数是否为0，如果为0则返回错误
    CHK_PRT_RET(maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);

    // 计算循环次数：总数据量 / 单次最大数据量，向上取整
    u64 loopTimes = dataCount_ / maxDataCountPerLoop + static_cast<u64>(dataCount_ % maxDataCountPerLoop != 0);
    // 初始化已处理的数据元素个数
    u64 processedDataCount = 0;

    // 打印循环次数日志
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor] dataCount_[%llu], maxDataCountPerLoop[%llu], loopTimes[%llu]",
        dataCount_, maxDataCountPerLoop, loopTimes);

    // 循环执行，每次处理一部分数据
    for (u64 loop = 0; loop < loopTimes; loop++) {
        uint32_t count = dataCount_ / rankSize_;
        // 计算当前循环处理的数据元素个数
        // 如果是最后一次循环，处理剩余数据；否则处理单次最大数据量
        u64 currDataCount = (loop == loopTimes - 1) ? count - processedDataCount : maxDataCountPerLoop;

        // 计算当前循环的总数据大小（字节）
        u64 currDataSize = currDataCount * dataTypeSize_;
        // 计算当前循环每个rank的数据块大小（向上取整到rankSize份）
        // u64 currSizePerBlock = (currDataCount + rankSize_ - 1) / rankSize_ * dataTypeSize_;
        u64 currSizePerBlock = currDataSize;

        // 创建当前循环每个rank的数据大小向量
        std::vector<u64> currGroupSize;
        // 初始化剩余数据大小
        u64 sizeRemain = currDataSize;
        // 遍历每个rank，计算其数据切片大小
        for (u32 rankId = 0; rankId < rankSize_; rankId++) {
            // 计算当前rank的数据大小：取剩余数据和数据块大小的较小值
            u64 size = (sizeRemain > currSizePerBlock) ? currSizePerBlock : sizeRemain;
            // 将数据大小添加到向量
            currGroupSize.push_back(size);
            // 减少剩余数据大小
            sizeRemain -= size;
        }

        // 打印当前循环的调试日志
        HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor] loop[%llu] sliceSize[%llu], processedDataCount[%llu]",
            loop, currGroupSize[myRank_ % rankSize_], processedDataCount);

        // 调用ReduceScatter阶段执行函数
        CHK_RET(RunReduceScatter(param, resCtx, currGroupSize, currSizePerBlock, 
            processedDataCount, count, rsTempAlg));
        // 调用AllGather阶段执行函数
        CHK_RET(RunAllGather(param, resCtx, currGroupSize, currSizePerBlock,
            processedDataCount, count, agTempAlg));

        // 更新已处理的数据元素个数
        processedDataCount += currDataCount;
    }

    // 打印循环执行结束日志
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][OrchestrateLoop] Success");
    // 返回成功
    return HCCL_SUCCESS;
}

// 初始化执行器信息：检查和设置严格确定性模式
// param: 操作参数
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::InitExecutorInfo(const OpParam &param)
{
    // 检查是否需要启用严格确定性模式
    deterministicStrict_ = IsNeedStrictMode(param);
    // 如果启用严格模式，检查是否满足严格模式条件
    if (deterministicStrict_) {
        // 检查严格模式条件，如果不满足则返回不支持错误
        CHK_PRT_RET(!CheckStrictCondition(param),
            HCCL_ERROR("[InsV2AllReduceOrderPreservedExecutor] not support DETERMINISTIC_STRICT mode."),
            HCCL_E_NOT_SUPPORT);
    }
    // 打印严格确定性模式标志日志
    HCCL_INFO("[InsV2AllReduceOrderPreservedExecutor][InitExecutorInfo] deterministicStrict[%d]",
        deterministicStrict_);
    // 返回成功
    return HCCL_SUCCESS;
}

// 向上对齐到除数的倍数
// value: 需要对齐的值
// divisor: 除数（对齐边界）
// 返回值: 对齐后的值
// template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
// u64 InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RoundUpWithDivisor(
//     u64 value, u64 divisor) const
// {
//     // 如果值为0或除数为0，返回除数本身
//     if (value == 0 || divisor == 0) {
//         return divisor;
//     }
//     // 向上对齐计算：(value + divisor - 1) / divisor * divisor
//     return ((value + (divisor - 1)) / divisor) * divisor;
// }

// 计算每个数据块的大小：确定每个rank需要处理的数据量
// param: 操作参数
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcSizePerBlock(const OpParam &param)
{
    // 计算单卡数据量：总数据量 / rank数，向上取整
    u64 sizePerBlock = (dataCount_ + rankSize_ - 1) / rankSize_ * dataTypeSize_;
    // 将数据块大小向上对齐到最小切片对齐要求
    memInfo_.sizePerBlock = RoundUpWithDivisor(sizePerBlock, HCCL_MIN_SLICE_ALIGN_ORDER_PRESERVED);
    // 初始化all2all偏移量为0（用于AllToAll操作）
    memInfo_.all2allOffset = 0;
    // 初始化scratch内存标志为false（不使用scratch内存）
    memInfo_.scratchMemFlag = false;
    // 初始化总数据大小为0
    memInfo_.totalSize = 0;
    // 打印数据块大小计算结果日志
    HCCL_INFO("[CalcSizePerBlock] sizePerBlock[%llu], dataCount[%llu], rankSize[%u]",
        memInfo_.sizePerBlock, dataCount_, rankSize_);
    // 返回成功
    return HCCL_SUCCESS;
}

// 计算数据切片分组：确定每个rank实际分配的数据切片大小
// param: 操作参数
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CalcGroupSlices(const OpParam &param)
{
    // 清空数据切片大小向量
    memInfo_.groupSize.clear();
    // 初始化剩余数据大小为总数据大小
    u64 sizeRemain = dataSize_;
    // 遍历每个rank，分配数据切片大小
    for (u32 rankId = 0; rankId < rankSize_; rankId++) {
        // 计算当前rank的数据切片大小：取剩余数据和数据块大小的较小值
        u64 size = (sizeRemain > memInfo_.sizePerBlock) ? memInfo_.sizePerBlock : sizeRemain;
        // 将数据切片大小添加到向量
        memInfo_.groupSize.push_back(size);
        // 减少剩余数据大小
        sizeRemain -= size;
    }
    // 计算总数据大小：取数据块大小 * rank数和实际数据大小的较大值
    memInfo_.totalSize = std::max(memInfo_.sizePerBlock * rankSize_, dataSize_);
    // 打印数据切片分组计算结果日志
    HCCL_INFO("[CalcGroupSlices] groupSize.size[%u], totalSize[%llu]",
        memInfo_.groupSize.size(), memInfo_.totalSize);
    // 返回成功
    return HCCL_SUCCESS;
}

// 检查是否需要严格确定性模式
// param: 操作参数
// 返回值: 如果需要严格模式返回true，否则返回false
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
bool InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::IsNeedStrictMode(const OpParam &param) const
{
    // 严格模式条件：
    // 1. 环境变量HCCL_DETERMINISTIC设置为STRICT
    // 2. 数据类型为FP16、FP32或BFP16
    // 3. 归约操作为SUM
    // 4. rank数大于等于最小严格rank数
    bool isStrictMode = (GetLocalDeterministicConfig() == static_cast<u8>(DeterministicEnableLevel::DETERMINISTIC_STRICT))
        && (param.DataDes.dataType == HCCL_DATA_TYPE_FP16 || param.DataDes.dataType == HCCL_DATA_TYPE_FP32 ||
            param.DataDes.dataType == HCCL_DATA_TYPE_BFP16)
        && (param.reduceType == HCCL_REDUCE_SUM)
        && rankSize_ >= MIN_STRICT_RANK_NUM_ORDER_PRESERVED;
    // 返回严格模式标志
    return isStrictMode;
}

// 检查严格模式条件：验证是否支持严格模式
// param: 操作参数
// 返回值: 如果支持返回true，否则返回false
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
bool InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::CheckStrictCondition(const OpParam &param) const
{
    // 检查归约操作是否为PROD，严格模式不支持PROD
    CHK_PRT_RET(param.reduceType == HCCL_REDUCE_PROD,
        HCCL_ERROR("[CheckStrictCondition] DETERMINISTIC_STRICT mode not support PROD."), false);
    // 检查数据类型是否为FP64，严格模式不支持FP64
    CHK_PRT_RET(param.DataDes.dataType == HCCL_DATA_TYPE_FP64,
        HCCL_ERROR("[CheckStrictCondition] DETERMINISTIC_STRICT mode not support FP64."), false);
    // 返回支持
    return true;
}

// 执行ReduceScatter阶段
// param: 操作参数
// resCtx: 资源上下文
// currGroupSize: 当前循环每个rank的数据大小向量
// currSizePerBlock: 当前循环每个rank的数据块大小
// processedDataCount: 已处理的数据元素个数
// currDataCount: 当前循环处理的数据元素个数
// rsTempAlg: ReduceScatter算法模板实例
// 返回值: 成功返回HCCL_SUCCESS，失败返回错误码
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunReduceScatter(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    const std::vector<u64> &currGroupSize, u64 currSizePerBlock, u64 processedDataCount, u64 currDataCount,
    std::shared_ptr<InsAlgTemplateRS> rsTempAlg)
{
    // 准备ReduceScatter模板资源结构体并初始化
    TemplateResource rsTemplateAlgRes;
    InitTemplateResource(resCtx, rsTemplateAlgRes);
    PrintTemplateResource(rsTemplateAlgRes);

    // 准备ReduceScatter模板数据参数结构体并初始化
    TemplateDataParams rsTempAlgParams;
    InitTemplateDataParams(param, resCtx, rsTempAlgParams);
    PrintTemplateDataParams(rsTempAlgParams);

    // 计算输入缓冲区基址偏移（已处理数据的偏移）
    rsTempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    // 计算输出缓冲区基址偏移（已处理数据的偏移）
    rsTempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    // 设置临时缓冲区基址偏移为0（每次循环重新使用临时缓冲区）
    rsTempAlgParams.buffInfo.hcclBuffBaseOff = 0;

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
        u64 userMemInOffset = processedDataCount * unitSize + dataId * unitSize * currDataCount;
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
    rsTempAlgParams.allRankSliceSize = currGroupSize;
    // 设置本rank的数据切片大小
    rsTempAlgParams.sliceSize = currGroupSize[myRank_ % rankSize_];
    // 设置尾部大小等于切片大小
    rsTempAlgParams.tailSize = rsTempAlgParams.sliceSize;
    // 设置输入切片步幅（总数据大小，用于多份输入数据的间隔）
    rsTempAlgParams.inputSliceStride = dataSize_;
    // 设置输出切片步幅为0（输出只有一份）
    rsTempAlgParams.outputSliceStride = 0;
    // 设置重复次数为1（不重复）
    rsTempAlgParams.repeatNum = 1;
    // 设置输入重复步幅为0
    rsTempAlgParams.inputRepeatStride = 0;
    // 设置输出重复步幅为0
    rsTempAlgParams.outputRepeatStride = 0;
    // 对于ReduceScatter模板，count是输出大小（每个rank的数据元素个数）
    rsTempAlgParams.count = currGroupSize[myRank_ % rankSize_] / dataTypeSize_;
    
    // 打印ReduceScatter模板参数
    HCCL_INFO("[RunReduceScatter] count[%llu], sliceSize[%llu], inBuffBaseOff[%llu], outBuffBaseOff[%llu]",
        rsTempAlgParams.count, rsTempAlgParams.sliceSize, 
        rsTempAlgParams.buffInfo.inBuffBaseOff, rsTempAlgParams.buffInfo.outBuffBaseOff);

    // 将内存块信息设置到ReduceScatter模板
    rsTempAlg->SetMemBlockInfo(memBlockInfo);
    // 调用ReduceScatter模板的KernelRun函数执行ReduceScatter阶段
    CHK_RET(rsTempAlg->KernelRun(param, rsTempAlgParams, rsTemplateAlgRes));
    
    // 返回成功
    return HCCL_SUCCESS;
}

// 执行AllGather阶段
// param: 操作参数
// resCtx: 资源上下文
// currGroupSize: 当前循环每个rank的数据大小向量
// currSizePerBlock: 当前循环每个rank的数据块大小
// processedDataCount: 已处理的数据元素个数
// currDataCount: 当前循环处理的数据元素个数
// agTempAlg: AllGather算法模板实例
// 返回值: 成功返回HCCL_SUCCESS，失败返回错误码
template <typename AlgTopoMatch, typename InsAlgTemplateRS, typename InsAlgTemplateAG>
HcclResult InsV2AllReduceOrderPreservedExecutor<AlgTopoMatch, InsAlgTemplateRS, InsAlgTemplateAG>::RunAllGather(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx,
    const std::vector<u64> &currGroupSize, u64 currSizePerBlock, u64 processedDataCount, u64 currDataCount,
    std::shared_ptr<InsAlgTemplateAG> agTempAlg)
{
    // 准备AllGather模板资源结构体并初始化
    TemplateResource agTemplateAlgRes;
    InitTemplateResource(resCtx, agTemplateAlgRes);
    PrintTemplateResource(agTemplateAlgRes);

    // 准备AllGather模板数据参数结构体并初始化
    TemplateDataParams agTempAlgParams;
    InitTemplateDataParams(param, resCtx, agTempAlgParams);
    PrintTemplateDataParams(agTempAlgParams);

    // 计算输入缓冲区基址偏移（已处理数据的偏移）
    agTempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
    // 计算输出缓冲区基址偏移（已处理数据的偏移）
    agTempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
    // 设置临时缓冲区基址偏移为0（每次循环重新使用临时缓冲区）
    agTempAlgParams.buffInfo.hcclBuffBaseOff = 0;

    // 设置所有rank的数据切片大小向量
    agTempAlgParams.allRankSliceSize = currGroupSize;
    // 设置本rank的数据切片大小
    agTempAlgParams.sliceSize = currGroupSize[myRank_ % rankSize_];
    // 设置尾部大小等于切片大小
    agTempAlgParams.tailSize = agTempAlgParams.sliceSize;
    // 设置输入切片步幅（总数据大小，用于多份输入数据的间隔）
    agTempAlgParams.inputSliceStride = dataSize_;
    // 设置输出切片步幅为0（输出只有一份）
    agTempAlgParams.outputSliceStride = 0;
    // 设置重复次数为1（不重复）
    agTempAlgParams.repeatNum = 1;
    // 设置输入重复步幅为0
    agTempAlgParams.inputRepeatStride = 0;
    // 设置输出重复步幅为0
    agTempAlgParams.outputRepeatStride = 0;
    // 对于AllGather模板，count同样是每个rank的数据元素个数
    agTempAlgParams.count = currGroupSize[myRank_ % rankSize_] / dataTypeSize_;
    
    // 打印AllGather模板参数
    HCCL_INFO("[RunAllGather] count[%llu], sliceSize[%llu], inBuffBaseOff[%llu], outBuffBaseOff[%llu]",
        agTempAlgParams.count, agTempAlgParams.sliceSize, 
        agTempAlgParams.buffInfo.inBuffBaseOff, agTempAlgParams.buffInfo.outBuffBaseOff);

    // 调用AllGather模板的KernelRun函数执行AllGather阶段
    CHK_RET(agTempAlg->KernelRun(param, agTempAlgParams, agTemplateAlgRes));
    
    // 返回成功
    return HCCL_SUCCESS;
}

// 注册保序AllReduce执行器
// HCCL_CMD_ALLREDUCE: 命令类型
// AllReduceOrderPreserved: 算法名称
// InsV2AllReduceOrderPreservedExecutor: 执行器类型
// TopoMatch1D: 拓扑匹配类型
// InsTempReduceScatterOrderPreservedLevel1: ReduceScatter模板类型
// InsTempAllGatherMesh1D: AllGather模板类型
REGISTER_EXECUTOR_BY_TWO_TEMPS(HcclCMDType::HCCL_CMD_ALLREDUCE, AllReduceOrderPreserved,
    InsV2AllReduceOrderPreservedExecutor, TopoMatch1D,
    InsTempReduceScatterOrderPreservedLevel1, InsTempAllGatherMesh1D);

// 命名空间结束
}