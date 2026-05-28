/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 本程序是自由软件，您可以根据CANN开源软件许可证2.0版（"许可证"）的条款和条件重新分发和/或修改它。
 * 请参阅许可证了解详情。未经许可证授权，您不得使用本文件。
 * 本软件按"原样"提供，不附带任何明示或暗示的担保，包括但不限于非侵权、适销性或特定用途适用性。
 * 请参阅软件存储库根目录中的LICENSE以获取许可证全文。
 */

// 包含本类的头文件声明
#include "ins_temp_reduce_scatter_order_preserved_level1.h"
// 包含环境配置
#include "alg_env_config.h"

// 使用ops_hccl命名空间
namespace ops_hccl {

// 构造函数：初始化ReduceScatter保序模板实例
// param: 操作参数，包含输入输出指针、数据类型等信息
// rankId: 当前rank的ID
// subCommRanks: 子通信域的rank列表
InsTempReduceScatterOrderPreservedLevel1::InsTempReduceScatterOrderPreservedLevel1(const OpParam &param,
    const u32 rankId, const std::vector<std::vector<u32>> &subCommRanks)
    : InsAlgTemplateBase(param, rankId, subCommRanks)  // 调用基类构造函数初始化基础成员
{
    // 根据环境变量HCCL_DETERMINISTIC判断是否启用严格确定性模式
    // 严格模式下确保浮点数归约结果的可重现性
    deterministicStrict_ = (GetExternalInputHcclDeterministic() == static_cast<u8>(DeterministicEnableLevel::DETERMINISTIC_STRICT));
    // 初始化all2all偏移量为0，用于计算AllToAll操作中的输出索引
    all2allOffset_ = 0;
}

// 析构函数：释放资源（当前无动态分配资源需要释放）
InsTempReduceScatterOrderPreservedLevel1::~InsTempReduceScatterOrderPreservedLevel1()
{}

// 计算资源请求：确定执行所需的线程数、通知数和通道资源
// comm: HCCL通信句柄
// param: 操作参数
// topoInfo: 拓扑信息，包含网络层详细信息
// resourceRequest: 输出参数，填充资源请求信息
HcclResult InsTempReduceScatterOrderPreservedLevel1::CalcRes(
    HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
    AlgResourceRequest &resourceRequest)
{
    // 计算线程数：至少为1，最大为rank数量
    u32 threadNum = templateRankSize_ > 1 ? templateRankSize_ : 1;
    // 限制线程数不超过保序ReduceScatter的最大流数量
    threadNum = std::min(threadNum, REDUCE_SCATTER_MAX_STREAM_NUM_ORDER_PRESERVED);
    // 从线程数从线程数量（主线程之外的辅助线程）
    resourceRequest.slaveThreadNum = threadNum - 1;

    // 为每个从线程分配2个通知对象（用于线程间同步）
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(2);
    }
    // 主线程需要的通知对象数量等于从线程数量（用于接收来自从线程的通知）
    resourceRequest.notifyNumOnMainThread = threadNum - 1;

    // 计算level0（第一层）通信通道请求
    std::vector<HcclChannelDesc> level0Channels;
    CHK_RET(CalcChannelRequestMesh1D(comm, param, topoInfo, subCommRanks_, level0Channels));
    // 将通道请求添加到资源请求中
    resourceRequest.channels.push_back(level0Channels);

    // 打印资源计算结果日志
    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][CalcRes] myRank[%u], threadNum[%u], "
        "notifyNumOnMainThread[%u], slaveThreadNum[%u]",
        myRank_, threadNum, resourceRequest.notifyNumOnMainThread, resourceRequest.slaveThreadNum);
    return HCCL_SUCCESS;
}

// 计算临时缓冲区的倍数
// inBuffType: 输入缓冲区类型
// outBuffType: 输出缓冲区类型
// 返回值: 临时缓冲区大小倍数，保序模式下需要rankSize倍的临时空间
u64 InsTempReduceScatterOrderPreservedLevel1::CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType)
{
    // 临时缓冲区倍数等于rank数量（每个rank需要存储一份中间数据）
    u64 scratchMultiple = templateRankSize_;
    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][CalcScratchMultiple] scratchMultiple[%u]", scratchMultiple);
    return scratchMultiple;
}

// 核心执行函数：执行ReduceScatter保序操作的主流程
// param: 操作参数
// tempAlgParams: 模板数据参数，包含缓冲区信息和切片大小
// templateResource: 模板资源，包含线程和通道资源
HcclResult InsTempReduceScatterOrderPreservedLevel1::KernelRun(
    const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource)
{
    // 如果切片大小和尾部大小都为0，说明没有数据需要处理，直接返回成功
    if (tempAlgParams.sliceSize == 0 && tempAlgParams.tailSize == 0) {
        HCCL_DEBUG("[InsTempReduceScatterOrderPreservedLevel1] myRank[%u] sliceSize and tailSize are 0, skip.", myRank_);
        return HCCL_SUCCESS;
    }

    // 初始化成员变量
    threadNum_ = templateResource.threads.size();  // 线程数量
    dataType_ = param.DataDes.dataType;  // 数据类型
    reduceOp_ = param.reduceType;  // 归约操作类型
    processSize_ = tempAlgParams.sliceSize;  // 处理的数据大小（字节）
    count_ = tempAlgParams.sliceSize / DATATYPE_SIZE_TABLE[dataType_];  // 数据元素个数

    // 打印执行开始日志
    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][KernelRun] Start, threadNum[%u], count[%llu], "
        "dataType[%u], deterministicStrict[%d]", threadNum_, count_, dataType_, deterministicStrict_);

    // 步骤1: 执行预处理本地拷贝（将本rank对应的数据从用户输入拷贝到临时缓冲区）
    CHK_RET(PreLocalCopy(tempAlgParams, templateResource.threads));

    // 多线程同步：如果线程数大于1，需要先同步主线程和从线程
    if (threadNum_ > 1) {
        // 获取从线程列表（跳过主线程，主线程是threads[0]）
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        // 获取主线程通知从线程的通知索引
        GetNotifyIdxMainToSub(notifyIdxMainToSub_);
        // 执行主线程到从线程的预同步（等待所有从线程就绪）
        CHK_RET(PreSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxMainToSub_));
    }

    // 步骤2: 执行AllToAll操作（每个rank将自己的数据发送给其他rank，并接收其他rank的数据）
    CHK_RET(RunAllToAll(templateResource.channels, templateResource.threads, tempAlgParams));
    // 多线程同步：如果线程数大于1，需要在操作完成后同步
    if (threadNum_ > 1) {
        // 获取从线程列表
        std::vector<ThreadHandle> subThreads(templateResource.threads.begin() + 1, templateResource.threads.end());
        // 获取从线程通知主线程的通知索引
        GetNotifyIdxSubToMain(notifyIdxSubToMain_);
        // 执行从线程到主线程的后同步（等待所有从线程完成）
        CHK_RET(PostSyncInterThreads(templateResource.threads[0], subThreads, notifyIdxSubToMain_));
    }
    if (dataType_ == HcclDataType::HCCL_DATA_TYPE_FP64) {
        // 必须确保所有通信任务完成，因为接下来的 AICPU Reduce 运行在 CPU 上，不感知任务队列同步
        // 参考 ins_temp_reduce_scatter_aicpu_reduce_nhr.cc:74-79 的标准同步模式
        CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param.algTag)));
        CHK_RET(static_cast<HcclResult>(HcommBatchModeStart(param.algTag)));
        for (const auto &thread : templateResource.threads) {
            CHK_RET(static_cast<HcclResult>(HcommThreadJoin(thread, CUSTOM_TIMEOUT)));
        }
    }

    // 步骤3: 执行本地归约操作（将收到的所有数据在本地进行归约）
    CHK_RET(RunLocalReduce(templateResource.threads, tempAlgParams));

    // 步骤4: 执行后处理拷贝（将归约结果从临时缓冲区拷贝到用户输出缓冲区）
    CHK_RET(PostCopy(tempAlgParams, templateResource.threads));

    // 打印执行结束日志
    HCCL_INFO("[InsTempReduceScatterOrderPreservedLevel1][KernelRun] End");
    return HCCL_SUCCESS;
}

// 获取资源请求（另一个版本，用于查询资源需求）
// resourceRequest: 输出参数，填充资源请求信息
HcclResult InsTempReduceScatterOrderPreservedLevel1::GetRes(AlgResourceRequest &resourceRequest) const
{
    // 获取线程数量
    u32 threadNum = GetThreadNum();
    // 设置从线程数量
    resourceRequest.slaveThreadNum = threadNum - 1;
    // 为每个从线程分配2个通知对象
    for (u32 index = 0; index < threadNum - 1; index++) {
        resourceRequest.notifyNumPerThread.push_back(2);
    }
    // 主线程的通知对象数量等于从线程数量
    resourceRequest.notifyNumOnMainThread = threadNum - 1;
    return HCCL_SUCCESS;
}

// 获取线程数量
// 返回值: 至少为1，最大为rank数量
u64 InsTempReduceScatterOrderPreservedLevel1::GetThreadNum() const
{
    // 如果rank数量大于1则返回rank数量，否则返回1（单rank情况）
    return templateRankSize_ > 1 ? templateRankSize_ : 1;
}

// 获取主线程通知从线程的通知索引列表
// notifyIdxMainToSub: 输出参数，通知索引列表
void InsTempReduceScatterOrderPreservedLevel1::GetNotifyIdxMainToSub(std::vector<u32> &notifyIdxMainToSub)
{
    // 清空输出向量
    notifyIdxMainToSub.clear();
    // 获取线程数量
    u32 threadNum = GetThreadNum();
    // 计算从线程数量（线程总数减去主线程）
    u32 slaveThreadNum = threadNum - 1;
    // 为每个从线程设置通知索引为0（所有从线程使用同一个通知索引）
    for (u32 slaveThreadIdx = 0; slaveThreadIdx < slaveThreadNum; slaveThreadIdx++) {
        notifyIdxMainToSub.push_back(0);
    }
}

// 获取从线程通知主线程的通知索引列表
// notifyIdxSubToMain: 输出参数，通知索引列表
void InsTempReduceScatterOrderPreservedLevel1::GetNotifyIdxSubToMain(std::vector<u32> &notifyIdxSubToMain)
{
    // 清空输出向量
    notifyIdxSubToMain.clear();
    // 获取线程数量
    u32 threadNum = GetThreadNum();
    // 计算通知数量（等于从线程数量）
    u32 notifyNum = threadNum - 1;
    // 每个从线程有自己独立的通知索引（0, 1, 2, ...）
    for (u32 notifyIdx = 0; notifyIdx < notifyNum; notifyIdx++) {
        notifyIdxSubToMain.push_back(notifyIdx);
    }
}

// 计算输出索引：确定数据块在临时缓冲区中的位置
// round: 当前轮次（在AllToAll中代表发送/接收的目标rank）
// localRank: 本地rank在算法中的编号
// 返回值: 输出缓冲区中的索引位置
u32 InsTempReduceScatterOrderPreservedLevel1::CalcOutputIndex(const u32 round, const u32 localRank)
{
    // 使用取模运算确保索引在rank范围内
    // all2allOffset_为偏移量（当前为0），round为轮次，localRank为本地rank
    return (all2allOffset_ + round + localRank) % templateRankSize_;
}

// 判断是否为最后一个数据块
// outputIndex: 输出索引
// 返回值: 如果是最后一个数据块返回true
bool InsTempReduceScatterOrderPreservedLevel1::IsLastBlockData(const u32 outputIndex)
{
    // 最后一个数据块的索引等于rankSize - 1
    return outputIndex == templateRankSize_ - 1;
}

// 判断是否为最后一个rank
// rankId: rank编号
// 返回值: 如果是最后一个rank返回true
bool InsTempReduceScatterOrderPreservedLevel1::IsLastRank(const u32 rankId)
{
    // 最后一个rank的编号等于rankSize - 1
    return rankId == templateRankSize_ - 1;
}

// 预处理本地拷贝：将本rank需要的数据从用户输入缓冲区拷贝到临时缓冲区
// tempAlgParams: 模板数据参数
// threads: 线程句柄列表
HcclResult InsTempReduceScatterOrderPreservedLevel1::PreLocalCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    HCCL_INFO("zdy PreLocalCopy start\n");
    // 获取内存块信息（包含每个rank的数据大小和偏移量）
    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    // 获取本rank在算法中的编号
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    // 获取本rank对应的数据块大小
    u64 sliceSize = memBlockInfo.size[myAlgRank];
    // 如果数据块大小为0，跳过拷贝
    if (sliceSize == 0) {
        HCCL_DEBUG("[PreLocalCopy] myAlgRank[%u] sliceSize is 0, skip.", myAlgRank);
        return HCCL_SUCCESS;
    }

    // 计算源地址偏移量（用户输入缓冲区中的偏移）
    u64 srcOffset = memBlockInfo.userInputOffsets[myAlgRank];
    // 计算输出索引（确定在临时缓冲区中的位置）
    u32 outputIndex = CalcOutputIndex(myAlgRank, myAlgRank);
    // 计算目标地址偏移量（临时缓冲区中的偏移）
    u64 dstOffset = memBlockInfo.outputOffsets[outputIndex];

    // 打印调试日志
    HCCL_INFO("[PreLocalCopy] myAlgRank[%u], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu], outputIndex[%u]",
        myAlgRank, sliceSize, srcOffset, dstOffset, outputIndex);

    // 创建源数据切片（用户输入缓冲区）
    DataSlice srcSlice(tempAlgParams.buffInfo.inputPtr, srcOffset, sliceSize);
    // 创建目标数据切片（临时缓冲区）
    DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, sliceSize);
    HCCL_INFO("zdy [PreLocalCopy] srcSlice: %s", srcSlice.Describe().c_str());
    HCCL_INFO("zdy [PreLocalCopy] dstSlice: %s", dstSlice.Describe().c_str());
    // 执行本地内存拷贝（使用主线程threads[0]）
    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));
    HCCL_INFO("zdy PreLocalCopy end\n");

    return HCCL_SUCCESS;
}

// 执行AllToAll操作：每个rank将数据发送给其他rank并接收来自其他rank的数据
// channels: 通道映射表，key为远端rank，value为通道信息列表
// threads: 线程句柄列表
// tempAlgParams: 模板数据参数
HcclResult InsTempReduceScatterOrderPreservedLevel1::RunAllToAll(
    const std::map<u32, std::vector<ChannelInfo>> &channels,
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[OrderPreserved RunAllToAll] Start");

    // 获取内存块信息
    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    // 获取本rank在算法中的编号
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));
    HCCL_INFO("zdy alltoall myRank_ %u myAlgRank %u\n", myRank_, myAlgRank);

    // queIdx用于选择线程（从线程1开始，线程0是主线程）
    u32 queIdx = 1;
    // 遍历除自己外的所有rank（round 1 到 rankSize-1）
    for (u32 rankIdx = 1; rankIdx < templateRankSize_; rankIdx++) {
        // 计算下一个要通信的rank（环形方式）
        u32 nextRank = (myAlgRank + rankIdx) % templateRankSize_;
        // 获取远端rank的实际编号
        u32 remoteRank = subCommRanks_[0][nextRank];
        HCCL_INFO("zdy alltoall remoteRank %u nextRank %u\n", remoteRank, nextRank);

        // 在通道映射表中查找远端rank对应的通道
        auto channelIter = channels.find(remoteRank);
        // 检查通道是否存在
        CHK_PRT_RET(channelIter == channels.end(),
            HCCL_ERROR("[RunAllToAll] channel not found for nextRank[%u], remoteRank[%u]", nextRank, remoteRank), HCCL_E_INTERNAL);

        // 获取通道列表
        const std::vector<ChannelInfo> &curChannels = channelIter->second;
        // 检查通道列表是否为空
        CHK_PRT_RET(curChannels.empty(),
            HCCL_ERROR("[RunAllToAll] curChannels empty for nextRank[%u], channels size[%zu]", nextRank, curChannels.size()), HCCL_E_INTERNAL);
        HCCL_INFO("zdy alltoall channel %u rankIdx %u\n", curChannels.size(), rankIdx);

        // 获取要发送的数据块大小
        u64 sliceSize = memBlockInfo.size[nextRank];
        // 注意：即使sliceSize为0，仍然需要发送通知以保持Post/Wait同步机制
        HCCL_INFO("[RunAllToAll] nextRank[%u], sliceSize[%llu]", nextRank, sliceSize);

        u32 channelIdx = 0;
        //for (u32 channelIdx = 0; channelIdx < curChannels.size(); channelIdx++) {
        // 获取发送通道信息
        const ChannelInfo &linkSend = curChannels[channelIdx];
        // 获取接收通道信息（发送和接收使用相同的通道）
        const ChannelInfo &linkRecv = curChannels[channelIdx];
        // 获取用于执行此通信的线程
        ThreadHandle thread = threads[queIdx];

        // 获取远端rank的临时缓冲区地址
        void* remoteCclBuffAddr = linkSend.remoteCclMem.addr;

        // 计算发送时的输出索引（确定发送数据在本地临时缓冲区的位置）
        u32 txOutputIndex = CalcOutputIndex(nextRank, myAlgRank);
        // 计算发送源偏移量（用户输入缓冲区）
        u64 txSrcOffset = memBlockInfo.userInputOffsets[nextRank];
        // 计算发送目标偏移量（远端临时缓冲区）
        u64 txDstOffset = memBlockInfo.outputOffsets[txOutputIndex];

        // 创建发送源数据切片（本地用户输入缓冲区）
        DataSlice txSrcSlice(tempAlgParams.buffInfo.inputPtr, txSrcOffset, sliceSize);
        // 创建发送目标数据切片（远端临时缓冲区）
        DataSlice txDstSlice(remoteCclBuffAddr, txDstOffset, sliceSize);

        // 将数据切片组织成向量
        std::vector<DataSlice> txSrcSlices = {txSrcSlice};
        std::vector<DataSlice> txDstSlices = {txDstSlice};
        std::vector<DataSlice> rxSrcSlices = {};
        std::vector<DataSlice> rxDstSlices = {};
        HCCL_DEBUG("zdy alltoall txSrcSlice addr %llu txSrcOffset %u sliceSize %u\n", tempAlgParams.buffInfo.inputPtr, txSrcOffset, sliceSize);
        HCCL_DEBUG("zdy alltoall txDstSlice addr %llu txDstOffset %u sliceSize %u\n", remoteCclBuffAddr, txDstOffset, sliceSize);

        HCCL_INFO("zdy [RunAllToAll] srcSlice: %s", txSrcSlice.Describe().c_str());
        HCCL_INFO("zdy [RunAllToAll] dstSlice: %s", txDstSlice.Describe().c_str());

        // HCCL_DEBUG("zdy alltoall rxSrcSlice addr %llu rxSrcOffset %u sliceSize %u\n", remoteCclBuffAddr, rxSrcOffset, sliceSize);
        // HCCL_DEBUG("zdy alltoall rxDstSlice addr %llu rxDstOffset %u sliceSize %u\n", tempAlgParams.buffInfo.hcclBuff.addr, rxDstOffset, sliceSize);

        // 构造发送接收信息结构体
        SendRecvInfo sendRecvInfo{
            TxRxChannels{linkSend, linkRecv},  // 发送和接收通道
            TxRxSlicesList{SlicesList{txSrcSlices, txDstSlices}, SlicesList{rxSrcSlices, rxDstSlices}}  // 发送和接收数据切片
        };
        // 执行发送接收操作（同时发送和接收）
        CHK_RET(SendRecvWrite(sendRecvInfo, thread));
        //}
        // 处理完一个rank后，线程索引加1
        queIdx++;
        if (queIdx >= threadNum_) {
            queIdx = 1;
        }
        HCCL_INFO("[RunAllToAll] queIdx[%u], threadNum_[%u]", queIdx, threadNum_);
    }

    HCCL_INFO("[RunAllToAll] End");
    return HCCL_SUCCESS;
}

/**
 * 计算小于给定值的最大2次幂
 * 用于树形Reduce算法中确定每一步的数据分割点
 *
 * @param value 给定值（数据块数量）
 * @return 小于value的最大2次幂
 *
 * 示例：
 *   value=6 -> 返回4 (2^2)
 *   value=4 -> 返回2 (2^1)
 *   value=2 -> 返回1 (2^0)
 */
static u32 GetLargestPowerOf2LessThan(u32 value)
{
    if (value <= 1) {
        return 0;
    }
    u32 power = 1;
    while (power * 2 < value) {
        power *= 2;
    }
    return power;
}

/**
 * 执行本地归约操作：使用树形Reduce算法将临时缓冲区中所有数据块归约到本rank对应的位置
 *
 * 算法原理：
 * ----------
 * 传统Reduce：顺序累加，需要N-1次串行操作
 *   A = A + B + C + D + E + F (串行执行5次reduce)
 *
 * 树形Reduce：利用数据块不重叠的特性，分步reduce
 *   Step1: M=4, E->A, F->B（后面的块reduce到前面的对应位置）
 *   Step2: M=2, C->(AE), D->(BF)
 *   Step3: M=1, (BFD)->(AEC)
 *   最终结果在虚拟索引0位置（即myAlgRank位置）
 *
 * 虚拟索引映射：
 * ----------
 * 为了支持任意myAlgRank，使用虚拟索引：
 *   虚拟索引i -> peerRank = (myAlgRank + i) % templateRankSize_
 *   虚拟索引0 对应 myAlgRank（最终reduce目标）
 *
 * 每步操作：
 * ----------
 * 1. 计算M = 小于remainingBlocks的最大2次幂
 * 2. 将索引[M..remainingBlocks-1]的数据块reduce到索引[0..M-1]的对应位置
 * 3. 目标索引 = 源索引 % M（取模映射）
 * 4. 更新remainingBlocks = M，进入下一轮
 *
 * @param threads 线程句柄列表，使用threads[0]（主线程）串行执行
 * @param tempAlgParams 模板数据参数，包含缓冲区地址信息
 * @return HcclResult 执行结果
 */
HcclResult InsTempReduceScatterOrderPreservedLevel1::RunLocalReduce(
    const std::vector<ThreadHandle> &threads, const TemplateDataParams &tempAlgParams)
{
    HCCL_INFO("[RunLocalReduce]lgx Start, deterministicStrict[%d], templateRankSize[%u]",
        deterministicStrict_, templateRankSize_);

    // 单rank场景无需reduce，直接返回
    if (templateRankSize_ <= 1) {
        HCCL_INFO("[RunLocalReduce] Skip for single rank");
        return HCCL_SUCCESS;
    }

    // 获取内存块信息，包含各数据块的偏移量和大小
    const MemBlockInfo &memBlockInfo = memBlockInfo_;

    // 获取本rank在算法中的编号（区分通信域rank和算法内部rank）
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    HCCL_INFO("[RunLocalReduce] myAlgRank[%u]", myAlgRank);

    // 获取本rank对应的数据块大小
    u64 sliceSize = memBlockInfo.size[myAlgRank];
    // 计算元素个数（字节数除以数据类型大小）
    u64 count = sliceSize / DATATYPE_SIZE_TABLE[dataType_];

    // 计算本rank的输出索引
    u32 myOutputIndex = CalcOutputIndex(myAlgRank, myAlgRank);
    // 获取目标偏移量（归约结果存放位置）
    u64 dstOffset = memBlockInfo.outputOffsets[myOutputIndex];

    HCCL_DEBUG("[RunLocalReduce] myAlgRank[%u], myOutputIndex[%u], dstOffset[%llu], sliceSize[%llu]",
        myAlgRank, myOutputIndex, dstOffset, sliceSize);

    // ========== 树形Reduce主循环 ==========
    // remainingBlocks: 当前待处理的数据块数量
    u32 remainingBlocks = templateRankSize_;
    u32 step = 0;

    while (remainingBlocks > 1) {
        // 获取小于remainingBlocks的最大2次幂，作为分割点M
        // 例如: remainingBlocks=6 -> M=4
        u32 M = GetLargestPowerOf2LessThan(remainingBlocks);
        if (M == 0) {
            break;
        }

        HCCL_INFO("[RunLocalReduce] Step[%u]: remainingBlocks[%u], M[%u]", step, remainingBlocks, M);

        // ========== 执行本步骤的reduce操作 ==========
        // 遍历索引[M..remainingBlocks-1]，将每个数据块reduce到对应的目标位置
        // 目标索引 = 源索引 % M（取模映射到前M个位置）
        for (u32 srcVirtualIdx = M; srcVirtualIdx < remainingBlocks; srcVirtualIdx++) {
            // 计算目标虚拟索引：取模映射
            // 例如: srcVirtualIdx=4, M=4 -> dstVirtualIdx=0
            //       srcVirtualIdx=5, M=4 -> dstVirtualIdx=1
            u32 dstVirtualIdx = srcVirtualIdx % M;

            // 虚拟索引映射到实际peerRank（数据来源rank）
            // peerRank表示该数据块来自哪个rank
            // 例如: myAlgRank=0, srcVirtualIdx=4 -> srcPeerRank=4
            u32 srcPeerRank = (myAlgRank + srcVirtualIdx) % templateRankSize_;
            u32 dstPeerRank = (myAlgRank + dstVirtualIdx) % templateRankSize_;

            // 计算数据块在临时缓冲区中的实际位置
            // CalcOutputIndex(peerRank, myAlgRank): peerRank的数据在本rank视角下的位置
            u32 srcOutputIndex = CalcOutputIndex(srcPeerRank, myAlgRank);
            u32 dstOutputIndex = CalcOutputIndex(dstPeerRank, myAlgRank);

            // 获取偏移量
            u64 srcOffset = memBlockInfo.outputOffsets[srcOutputIndex];
            u64 dstOffset = memBlockInfo.outputOffsets[dstOutputIndex];

            // 跳过零大小数据块
            if (sliceSize == 0) {
                continue;
            }

            // 创建数据切片描述符（4参数版本，包含count）
            // 注意：srcSlice和dstSlice都使用sliceSize和count（基于本rank的数据大小）
            // 因为对端发送的是属于本rank的那一部分数据，大小等于本rank的数据大小
            DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, sliceSize, count);
            DataSlice dstSlice(tempAlgParams.buffInfo.hcclBuff.addr, dstOffset, sliceSize, count);

            HCCL_INFO("[RunLocalReduce] Step[%u]: virtualIdx[%u]->[%u], peerRank[%u]->[%u], "
                "offset[%llu]->[%llu]", step, srcVirtualIdx, dstVirtualIdx, srcPeerRank, dstPeerRank,
                srcOffset, dstOffset);
            HCCL_INFO("zdy [RunLocalReduce] srcSlice: %s", srcSlice.Describe().c_str());
            HCCL_INFO("zdy [RunLocalReduce] dstSlice: %s", dstSlice.Describe().c_str());

            // 执行本地reduce操作（使用主线程threads[0]串行执行）
            CHK_RET(LocalReduce(threads[0], srcSlice, dstSlice, dataType_, reduceOp_));
        }

        // 更新剩余数据块数量，进入下一轮
        remainingBlocks = M;
        step++;
    }

    HCCL_INFO("[RunLocalReduce] End, total steps[%u], final data at virtualIdx[0] (myAlgRank[%u])",
        step, myAlgRank);
    return HCCL_SUCCESS;
}

// 后处理拷贝：将归约结果从临时缓冲区拷贝到用户输出缓冲区
// tempAlgParams: 模板数据参数
// threads: 线程句柄列表
HcclResult InsTempReduceScatterOrderPreservedLevel1::PostCopy(
    const TemplateDataParams &tempAlgParams, const std::vector<ThreadHandle> &threads)
{
    // 获取内存块信息
    const MemBlockInfo &memBlockInfo = memBlockInfo_;
    // 获取本rank在算法中的编号
    u32 myAlgRank = 0;
    CHK_RET(GetAlgRank(myRank_, subCommRanks_[0], myAlgRank));

    // 获取本rank对应的数据块大小
    u64 sliceSize = memBlockInfo.size[myAlgRank];
    // 如果数据块大小为0，跳过拷贝
    if (sliceSize == 0) {
        HCCL_DEBUG("[PostCopy] myAlgRank[%u] sliceSize is 0, skip.", myAlgRank);
        return HCCL_SUCCESS;
    }

    // 计算本rank的输出索引
    u32 outputIndex = CalcOutputIndex(myAlgRank, myAlgRank);
    //u32 outputIndex = myAlgRank;
    // 计算源地址偏移量（临时缓冲区中的归约结果位置）
    u64 srcOffset = memBlockInfo.outputOffsets[outputIndex];
    // 目标偏移量为0（用户输出缓冲区起始位置）
    u64 dstOffset = tempAlgParams.buffInfo.outBuffBaseOff;

    // 打印调试日志
    HCCL_INFO("[PostCopy] myAlgRank[%u], sliceSize[%llu], srcOffset[%llu], dstOffset[%llu]",
        myAlgRank, sliceSize, srcOffset, dstOffset);

    // 创建源数据切片（临时缓冲区）
    DataSlice srcSlice(tempAlgParams.buffInfo.hcclBuff.addr, srcOffset, sliceSize);
    // 创建目标数据切片（用户输出缓冲区）
    DataSlice dstSlice(tempAlgParams.buffInfo.outputPtr, dstOffset, sliceSize);

    HCCL_INFO("zdy [PostCopy] srcSlice: %s", srcSlice.Describe().c_str());
    HCCL_INFO("zdy [PostCopy] dstSlice: %s", dstSlice.Describe().c_str());
    // 执行本地内存拷贝（使用主线程threads[0]）
    CHK_RET(LocalCopy(threads[0], srcSlice, dstSlice));

    return HCCL_SUCCESS;
}
// 命名空间结束
}