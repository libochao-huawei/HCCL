
#include "type_conversion.h"

namespace Hccl {

std::map<CheckerOpType, OpType> g_CheckerOpType2OpType_aicpu = {
    {CheckerOpType::BROADCAST, OpType::BROADCAST},
    {CheckerOpType::ALLREDUCE, OpType::ALLREDUCE},
    {CheckerOpType::REDUCE, OpType::REDUCE},
    {CheckerOpType::SEND, OpType::SEND},
    {CheckerOpType::RECEIVE, OpType::RECV},
    {CheckerOpType::ALLGATHER, OpType::ALLGATHER},
    {CheckerOpType::REDUCE_SCATTER, OpType::REDUCESCATTER},
    {CheckerOpType::ALLTOALLV, OpType::ALLTOALLV},
    {CheckerOpType::ALLTOALLVC, OpType::ALLTOALLVC},
    {CheckerOpType::ALLTOALL, OpType::ALLTOALL},
    {CheckerOpType::GATHER, OpType::GATHER},
    {CheckerOpType::SCATTER, OpType::SCATTER},
    {CheckerOpType::ALLGATHER_V, OpType::ALLGATHERV},
    {CheckerOpType::REDUCE_SCATTER_V, OpType::REDUCESCATTERV},
};

std::map<OpType, CheckerOpType> g_OpType2CheckerOpType_aicpu = {
    {OpType::BROADCAST, CheckerOpType::BROADCAST},
    {OpType::ALLREDUCE, CheckerOpType::ALLREDUCE},
    {OpType::REDUCE, CheckerOpType::REDUCE},
    {OpType::SEND, CheckerOpType::SEND},
    {OpType::RECV, CheckerOpType::RECEIVE},
    {OpType::ALLGATHER, CheckerOpType::ALLGATHER},
    {OpType::REDUCESCATTER, CheckerOpType::REDUCE_SCATTER},
    {OpType::ALLTOALLV, CheckerOpType::ALLTOALLV},
    {OpType::ALLTOALLVC, CheckerOpType::ALLTOALLVC},
    {OpType::ALLTOALL, CheckerOpType::ALLTOALL},
    {OpType::GATHER, CheckerOpType::GATHER},
    {OpType::SCATTER, CheckerOpType::SCATTER},
    {OpType::ALLGATHERV, CheckerOpType::ALLGATHER_V},
    {OpType::REDUCESCATTERV, CheckerOpType::REDUCE_SCATTER_V},
};

std::map<CheckerReduceOp, ReduceOp> g_CheckerReduceOp2ReduceOp_aicpu = {
    {CheckerReduceOp::REDUCE_SUM, ReduceOp::SUM},
    {CheckerReduceOp::REDUCE_PROD, ReduceOp::PROD},
    {CheckerReduceOp::REDUCE_MAX, ReduceOp::MAX},
    {CheckerReduceOp::REDUCE_MIN, ReduceOp::MIN}
};

std::map<ReduceOp, CheckerReduceOp> g_ReduceOp2CheckerReduceOp_aicpu = {
    {ReduceOp::SUM, CheckerReduceOp::REDUCE_SUM},
    {ReduceOp::PROD, CheckerReduceOp::REDUCE_PROD},
    {ReduceOp::MAX, CheckerReduceOp::REDUCE_MAX},
    {ReduceOp::MIN, CheckerReduceOp::REDUCE_MIN}
};

std::map<CheckerDataType, DataType> g_CheckerDataType2DataType_aicpu = {
    {CheckerDataType::DATA_TYPE_INT8, DataType::INT8},
    {CheckerDataType::DATA_TYPE_INT16, DataType::INT16},
    {CheckerDataType::DATA_TYPE_INT32, DataType::INT32},
    {CheckerDataType::DATA_TYPE_FP16, DataType::FP16},
    {CheckerDataType::DATA_TYPE_FP32, DataType::FP32},
    {CheckerDataType::DATA_TYPE_UINT64, DataType::UINT64},
    {CheckerDataType::DATA_TYPE_UINT8, DataType::UINT8},
    {CheckerDataType::DATA_TYPE_UINT16, DataType::UINT16},
    {CheckerDataType::DATA_TYPE_UINT32, DataType::UINT32},
    {CheckerDataType::DATA_TYPE_FP64, DataType::FP64},
    {CheckerDataType::DATA_TYPE_BFP16, DataType::BFP16},
    {CheckerDataType::DATA_TYPE_INT128, DataType::INT128},
    {CheckerDataType::DATA_TYPE_INT64, DataType::INT64},
    {CheckerDataType::DATA_TYPE_HIF8, DataType::HIF8},
    {CheckerDataType::DATA_TYPE_FP8E4M3, DataType::FP8E4M3},
    {CheckerDataType::DATA_TYPE_FP8E5M2, DataType::FP8E5M2},
};

std::map<DataType, CheckerDataType> g_DataType2CheckerDataType_aicpu = {
    {DataType::INT8, CheckerDataType::DATA_TYPE_INT8},
    {DataType::INT16, CheckerDataType::DATA_TYPE_INT16},
    {DataType::INT32, CheckerDataType::DATA_TYPE_INT32},
    {DataType::FP16, CheckerDataType::DATA_TYPE_FP16},
    {DataType::FP32, CheckerDataType::DATA_TYPE_FP32},
    {DataType::UINT64, CheckerDataType::DATA_TYPE_UINT64},
    {DataType::UINT8, CheckerDataType::DATA_TYPE_UINT8},
    {DataType::UINT16, CheckerDataType::DATA_TYPE_UINT16},
    {DataType::UINT32, CheckerDataType::DATA_TYPE_UINT32},
    {DataType::FP64, CheckerDataType::DATA_TYPE_FP64},
    {DataType::BFP16, CheckerDataType::DATA_TYPE_BFP16},
    {DataType::INT128, CheckerDataType::DATA_TYPE_INT128},
    {DataType::INT64, CheckerDataType::DATA_TYPE_INT64},
    {DataType::HIF8, CheckerDataType::DATA_TYPE_HIF8},
    {DataType::FP8E4M3, CheckerDataType::DATA_TYPE_FP8E4M3},
    {DataType::FP8E5M2, CheckerDataType::DATA_TYPE_FP8E5M2},
};

std::map<LinkProtoType, LinkProtoStub> g_LinkProtoType2LinkProtoStub_aicpu = {
    {LinkProtoType::RDMA, LinkProtoStub::RDMA},
    {LinkProtoType::HCCS_PCIE, LinkProtoStub::SDMA},
    {LinkProtoType::UB, LinkProtoStub::SDMA}
};

std::map<CheckerOpMode, OpMode> g_CheckerOpMode2OpMode_aicpu = {
    {CheckerOpMode::OPBASE, OpMode::OPBASE},
    {CheckerOpMode::OFFLOAD, OpMode::OFFLOAD}
};

std::map<Hccl::BufferType, checker::BufferType> g_HcclBufferType2CheckerBufferType_aicpu = {
    {Hccl::BufferType::INPUT, checker::BufferType::INPUT},
    {Hccl::BufferType::OUTPUT, checker::BufferType::OUTPUT},
    {Hccl::BufferType::SCRATCH, checker::BufferType::SCRATCH}
};

std::map<CheckerDevType, Hccl::DevType> g_CheckerDevType2HcclDevType_aicpu = {
    {CheckerDevType::DEV_TYPE_910, Hccl::DevType::DEV_TYPE_910A},
    {CheckerDevType::DEV_TYPE_310P3, Hccl::DevType::DEV_TYPE_V51_310_P3},
    {CheckerDevType::DEV_TYPE_910B, Hccl::DevType::DEV_TYPE_910A2},
    {CheckerDevType::DEV_TYPE_310P1, Hccl::DevType::DEV_TYPE_V51_310_P1},
    {CheckerDevType::DEV_TYPE_910_93, Hccl::DevType::DEV_TYPE_910A3},
    {CheckerDevType::DEV_TYPE_NOSOC, Hccl::DevType::DEV_TYPE_NOSOC},
    {CheckerDevType::DEV_TYPE_910_95, Hccl::DevType::DEV_TYPE_910_95},
    {CheckerDevType::DEV_TYPE_HF, Hccl::DevType::DEV_TYPE_910_95}
};

std::map<uint16_t, CheckerReduceOp> g_ReduceOp2CheckerReduceOp_ccu = {
    {10, CheckerReduceOp::REDUCE_SUM},
    { 9, CheckerReduceOp::REDUCE_MIN},
    { 8, CheckerReduceOp::REDUCE_MAX}
};

std::map<uint16_t, CheckerDataType> g_DataType2CheckerDataType_ccu = {
    {0, CheckerDataType::DATA_TYPE_INT8},
    {1, CheckerDataType::DATA_TYPE_INT16},
    {2, CheckerDataType::DATA_TYPE_INT32},
    {3, CheckerDataType::DATA_TYPE_FP16},
    {4, CheckerDataType::DATA_TYPE_FP32},
    {5, CheckerDataType::DATA_TYPE_UINT64},
    {6, CheckerDataType::DATA_TYPE_UINT8},
    {7, CheckerDataType::DATA_TYPE_UINT16},
    {8, CheckerDataType::DATA_TYPE_UINT32},
    {9, CheckerDataType::DATA_TYPE_FP64},
    {10, CheckerDataType::DATA_TYPE_BFP16},
    {11, CheckerDataType::DATA_TYPE_INT128},
    {12, CheckerDataType::DATA_TYPE_INT64},
    {13, CheckerDataType::DATA_TYPE_HIF8},
    {14, CheckerDataType::DATA_TYPE_FP8E4M3},
    {15, CheckerDataType::DATA_TYPE_FP8E5M2},
};

} // namespace Hccl