#include <vector>

#define private public
// #include "connected_link_mgr.h"
#undef private

#include "ins_coll_alg_base.h"
#include "ins_coll_alg_registry.h"
#include "coll_service_stub.h"
#include "host_task_transform.h"
#include "rank_gph.h"
#include "rank_info_recorder.h"
#include "check_utils.h"
#include "mem_layout.h"
#include "ccu_ins_preprocessor.h"
#include "communicator_impl_stub.h"

#include "rank_graph_builder.h"
#include "coll_gen_json.h"
#include "coll_alg_component_builder.h"
#include "env_config.h"

namespace Hccl {

void CollServiceStub::Init()
{
    CollAlgComponentInit();
}

void CollServiceStub::CollAlgComponentInit()
{
    HcclMainboardId hcclMainboardId;
    HrtGetMainboardId(comm_->GetDevPhyId(), hcclMainboardId);

    CollAlgComponentBuilder collAlgComponentBuilder;

    collAlgComponent_ = collAlgComponentBuilder.SetRankGraph(comm_->GetRankGraph())
                            .SetDevType(comm_->GetDevType())
                            .SetMyRank(comm_->GetMyRank())
                            .SetRankSize(comm_->GetRankSize())
                            .SetMainboardId(hcclMainboardId)
                            .EnableDetour(EnvConfig::GetInstance().GetDetourConfig().GetDetourType()
                                            == HcclDetourType::HCCL_DETOUR_ENABLE_2P) // 当前只支持4P场景的绕路
                            .Build();

    if (collAlgComponent_ == nullptr) {
        HCCL_ERROR("collAlgComponent_ is a null pointer!");
        throw NullPtrException("collAlgComponent_ is a null pointer!");
    }
}

shared_ptr<InsQueue> CollServiceStub::Orchestrate(CollAlgOperator &op, std::string& algName)
{
    u64 tmpMemSize = static_cast<u32>(EnvConfig::GetInstance().GetAlgoConfig().GetBuffSize());
    CollAlgParams params;
    auto          insQueue = make_shared<InsQueue>();

    params.opMode        = op.opMode;
    params.maxTmpMemSize = tmpMemSize;
    HCCL_DEBUG("[CollServiceDeviceMode::%s] orchestrate with Ins start", __func__);
    HcclResult errCode = collAlgComponent_->Orchestrate(op, params, algName, insQueue);
    HCCL_DEBUG("[CollServiceDeviceMode::%s] orchestrate with Ins end", __func__);

    if (errCode != HcclResult::HCCL_SUCCESS) {
        auto msg = StringFormat("Error occurs when call collAlgComponent_.orchestrate(), error code: %d", errCode);
        THROW<InternalException>(msg);
    }

    return insQueue;
}

void CollServiceStub::LoadWithOpBasedMode(CollOperator &op, std::string& algName)
{
    insQue_ = Orchestrate(op, algName);
    // ccuResPack资源扩充
    u32 insQueueSize = insQue_->SizeOfSlaves() + 1; // 从流个数 + 主流
    ccuInsPreprocessor_.GetCcuComm()->GetCcuResPackMgr()->PrepareAlloc(insQueueSize);

    // 对insQ中每个ins,创建CcuContext实例且分配资源
    ccuInsPreprocessor_.PrepareCcuCtx(insQue_, false);
}

void CollServiceStub::TransformTask()
{
    u32 myRank = comm_->GetMyRank();
    for (auto slaveIter = insQue_->IterSlaves(); slaveIter.HasNext(); ++slaveIter) {
        for (auto insIter = slaveIter->Iter(); insIter.HasNext(); ++insIter) {
            HcclResult errCode = TransformIns2Task(*insIter, myRank, slaveIter->GetId());
            if (errCode != HcclResult::HCCL_SUCCESS) {
                auto msg = StringFormat("Error occurs when transform ins to tasks, error code: %d", errCode);
                THROW<InternalException>(msg);
            }
        }
    }
    for (auto insIter = insQue_->Iter(); insIter.HasNext(); ++insIter) {
        HcclResult errCode = TransformIns2Task(*insIter, myRank, 0);
        if (errCode != HcclResult::HCCL_SUCCESS) {
            auto msg = StringFormat("Error occurs when transform ins to tasks, error code: %d", errCode);
            THROW<InternalException>(msg);
        }
    }
}

// 这边需要再改造一下
void CollServiceStub::LoadWithOffloadMode(CollOperator &op)
{
    ;
}

CcuInsPreprocessor *CollServiceStub::GetCcuInsPreprocessor()
{
    return &ccuInsPreprocessor_;
}

}// namespace Hccl