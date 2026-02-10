
#include "type_conversion.h"

namespace HcclSim {

// std::map<CheckerOpType, OpType> g_CheckerOpType2OpType_aicpu = {
//     {CheckerOpType::BROADCAST, OpType::BROADCAST},
//     {CheckerOpType::ALLREDUCE, OpType::ALLREDUCE},
//     {CheckerOpType::REDUCE, OpType::REDUCE},
//     {CheckerOpType::SEND, OpType::SEND},
//     {CheckerOpType::RECEIVE, OpType::RECV},
//     {CheckerOpType::ALLGATHER, OpType::ALLGATHER},
//     {CheckerOpType::REDUCE_SCATTER, OpType::REDUCESCATTER},
//     {CheckerOpType::ALLTOALLV, OpType::ALLTOALLV},
//     {CheckerOpType::ALLTOALLVC, OpType::ALLTOALLVC},
//     {CheckerOpType::ALLTOALL, OpType::ALLTOALL},
//     {CheckerOpType::GATHER, OpType::GATHER},
//     {CheckerOpType::SCATTER, OpType::SCATTER},
//     {CheckerOpType::ALLGATHER_V, OpType::ALLGATHERV},
//     {CheckerOpType::REDUCE_SCATTER_V, OpType::REDUCESCATTERV},
// };

// std::map<OpType, CheckerOpType> g_OpType2CheckerOpType_aicpu = {
//     {OpType::BROADCAST, CheckerOpType::BROADCAST},
//     {OpType::ALLREDUCE, CheckerOpType::ALLREDUCE},
//     {OpType::REDUCE, CheckerOpType::REDUCE},
//     {OpType::SEND, CheckerOpType::SEND},
//     {OpType::RECV, CheckerOpType::RECEIVE},
//     {OpType::ALLGATHER, CheckerOpType::ALLGATHER},
//     {OpType::REDUCESCATTER, CheckerOpType::REDUCE_SCATTER},
//     {OpType::ALLTOALLV, CheckerOpType::ALLTOALLV},
//     {OpType::ALLTOALLVC, CheckerOpType::ALLTOALLVC},
//     {OpType::ALLTOALL, CheckerOpType::ALLTOALL},
//     {OpType::GATHER, CheckerOpType::GATHER},
//     {OpType::SCATTER, CheckerOpType::SCATTER},
//     {OpType::ALLGATHERV, CheckerOpType::ALLGATHER_V},
//     {OpType::REDUCESCATTERV, CheckerOpType::REDUCE_SCATTER_V},
// };

// std::map<HcclReduceOp, ReduceOp> g_CheckerReduceOp2ReduceOp_aicpu = {
//     {HcclReduceOp::REDUCE_SUM, ReduceOp::SUM},
//     {HcclReduceOp::REDUCE_PROD, ReduceOp::PROD},
//     {HcclReduceOp::REDUCE_MAX, ReduceOp::MAX},
//     {HcclReduceOp::REDUCE_MIN, ReduceOp::MIN}
// };

// std::map<ReduceOp, HcclReduceOp> g_ReduceOp2CheckerReduceOp_aicpu = {
//     {ReduceOp::SUM, HcclReduceOp::REDUCE_SUM},
//     {ReduceOp::PROD, HcclReduceOp::REDUCE_PROD},
//     {ReduceOp::MAX, HcclReduceOp::REDUCE_MAX},
//     {ReduceOp::MIN, HcclReduceOp::REDUCE_MIN}
// };

// std::map<HcclDataType, DataType> g_CheckerDataType2DataType_aicpu = {
//     {HcclDataType::DATA_TYPE_INT8, DataType::INT8},
//     {HcclDataType::DATA_TYPE_INT16, DataType::INT16},
//     {HcclDataType::DATA_TYPE_INT32, DataType::INT32},
//     {HcclDataType::DATA_TYPE_FP16, DataType::FP16},
//     {HcclDataType::DATA_TYPE_FP32, DataType::FP32},
//     {HcclDataType::DATA_TYPE_UINT64, DataType::UINT64},
//     {HcclDataType::DATA_TYPE_UINT8, DataType::UINT8},
//     {HcclDataType::DATA_TYPE_UINT16, DataType::UINT16},
//     {HcclDataType::DATA_TYPE_UINT32, DataType::UINT32},
//     {HcclDataType::DATA_TYPE_FP64, DataType::FP64},
//     {HcclDataType::DATA_TYPE_BFP16, DataType::BFP16},
//     {HcclDataType::DATA_TYPE_INT128, DataType::INT128},
//     {HcclDataType::DATA_TYPE_INT64, DataType::INT64},
//     {HcclDataType::DATA_TYPE_HIF8, DataType::HIF8},
//     {HcclDataType::DATA_TYPE_FP8E4M3, DataType::FP8E4M3},
//     {HcclDataType::DATA_TYPE_FP8E5M2, DataType::FP8E5M2},
// };

std::map<DataType, HcclDataType> g_DataType2CheckerDataType_aicpu = {
    {DataType::INT8,   HcclDataType::HCCL_DATA_TYPE_INT8},
    {DataType::INT16,  HcclDataType::HCCL_DATA_TYPE_INT16},
    {DataType::INT32,  HcclDataType::HCCL_DATA_TYPE_INT32},
    {DataType::FP16,   HcclDataType::HCCL_DATA_TYPE_FP16},
    {DataType::FP32,   HcclDataType::HCCL_DATA_TYPE_FP32},
    {DataType::UINT64, HcclDataType::HCCL_DATA_TYPE_UINT64},
    {DataType::UINT8,  HcclDataType::HCCL_DATA_TYPE_UINT8},
    {DataType::UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
    {DataType::UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
    {DataType::FP64,   HcclDataType::HCCL_DATA_TYPE_FP64},
    {DataType::BFP16,  HcclDataType::HCCL_DATA_TYPE_BFP16},
    {DataType::INT128, HcclDataType::HCCL_DATA_TYPE_INT128},
    {DataType::INT64,  HcclDataType::HCCL_DATA_TYPE_INT64},
    {DataType::HIF8,   HcclDataType::HCCL_DATA_TYPE_HIF8},
    {DataType::FP8E4M3, HcclDataType::HCCL_DATA_TYPE_FP8E4M3},
    {DataType::FP8E5M2, HcclDataType::HCCL_DATA_TYPE_FP8E5M2},
};

// std::map<LinkProtoType, LinkProtoStub> g_LinkProtoType2LinkProtoStub_aicpu = {
//     {LinkProtoType::RDMA, LinkProtoStub::RDMA},
//     {LinkProtoType::HCCS_PCIE, LinkProtoStub::SDMA},
//     {LinkProtoType::UB, LinkProtoStub::SDMA}
// };

// std::map<CheckerOpMode, OpMode> g_CheckerOpMode2OpMode_aicpu = {
//     {CheckerOpMode::OPBASE, OpMode::OPBASE},
//     {CheckerOpMode::OFFLOAD, OpMode::OFFLOAD}
// };

// std::map<Hccl::BufferType, checker::BufferType> g_HcclBufferType2CheckerBufferType_aicpu = {
//     {Hccl::BufferType::INPUT, checker::BufferType::INPUT},
//     {Hccl::BufferType::OUTPUT, checker::BufferType::OUTPUT},
//     {Hccl::BufferType::SCRATCH, checker::BufferType::SCRATCH}
// };

// std::map<CheckerDevType, Hccl::DevType> g_CheckerDevType2HcclDevType_aicpu = {
//     {CheckerDevType::DEV_TYPE_910, Hccl::DevType::DEV_TYPE_910A},
//     {CheckerDevType::DEV_TYPE_310P3, Hccl::DevType::DEV_TYPE_V51_310_P3},
//     {CheckerDevType::DEV_TYPE_910B, Hccl::DevType::DEV_TYPE_910A2},
//     {CheckerDevType::DEV_TYPE_310P1, Hccl::DevType::DEV_TYPE_V51_310_P1},
//     {CheckerDevType::DEV_TYPE_910_93, Hccl::DevType::DEV_TYPE_910A3},
//     {CheckerDevType::DEV_TYPE_NOSOC, Hccl::DevType::DEV_TYPE_NOSOC},
//     {CheckerDevType::DEV_TYPE_910_95, Hccl::DevType::DEV_TYPE_910_95}
// };

std::map<uint16_t, HcclReduceOp> g_ReduceOp2CheckerReduceOp_ccu = {
    {10, HcclReduceOp::HCCL_REDUCE_SUM},
    { 9, HcclReduceOp::HCCL_REDUCE_MIN},
    { 8, HcclReduceOp::HCCL_REDUCE_MAX}
};

// std::map<uint16_t, HcclDataType> g_DataType2CheckerDataType_ccu = {
//     {0, HcclDataType::DATA_TYPE_INT8},
//     {1, HcclDataType::DATA_TYPE_INT16},
//     {2, HcclDataType::DATA_TYPE_INT32},
//     {3, HcclDataType::DATA_TYPE_FP16},
//     {4, HcclDataType::DATA_TYPE_FP32},
//     {5, HcclDataType::DATA_TYPE_UINT64},
//     {6, HcclDataType::DATA_TYPE_UINT8},
//     {7, HcclDataType::DATA_TYPE_UINT16},
//     {8, HcclDataType::DATA_TYPE_UINT32},
//     {9, HcclDataType::DATA_TYPE_FP64},
//     {10, HcclDataType::DATA_TYPE_BFP16},
//     {11, HcclDataType::DATA_TYPE_INT128},
//     {12, HcclDataType::DATA_TYPE_INT64},
//     {13, HcclDataType::DATA_TYPE_HIF8},
//     {14, HcclDataType::DATA_TYPE_FP8E4M3},
//     {15, HcclDataType::DATA_TYPE_FP8E5M2},
// };

} // namespace Hccl