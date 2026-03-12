#ifndef _CHECKER_CMD_H_
#define _CHECKER_CMD_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <unistd.h>
#include <vector>
#include <limits.h>
#include <ctype.h>
#include <map>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fstream>
#include <fcntl.h>
#include "checker.h"
#include "topo_meta.h"
#include "simple_param.h"
#include "hccl_common.h"
#include "semantics_utils.h"
#include "cmd.pb.h"

#include "checker_def.h"

#ifdef HCCL_ALG_ANALYZER_DAVID
#include "david_alg_config.h"
#endif

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;


struct DataSize {
    u64 minBytes;
    u64 maxBytes;
    u64 dataStepBytes = 0;
    double dataStepFactor;
};

enum OpMode {
    OPBASE = 0,
    OFFLOAD = 1
};

namespace checker {

const std::map<std::string, CheckerDataType> testDtypeMap = {
    {"int8", CheckerDataType::DATA_TYPE_INT8},
    {"int16", CheckerDataType::DATA_TYPE_INT16},
    {"int32", CheckerDataType::DATA_TYPE_INT32},
    {"fp16", CheckerDataType::DATA_TYPE_FP16},
    {"fp32", CheckerDataType::DATA_TYPE_FP32},
    {"int64", CheckerDataType::DATA_TYPE_INT64},
    {"uint64", CheckerDataType::DATA_TYPE_UINT64},
    {"uint8", CheckerDataType::DATA_TYPE_UINT8},
    {"uint16" , CheckerDataType::DATA_TYPE_UINT16},
    {"uint32" , CheckerDataType::DATA_TYPE_UINT32},
    {"fp64" , CheckerDataType::DATA_TYPE_FP64},
    {"bfp16" , CheckerDataType::DATA_TYPE_BFP16},
    {"int128", CheckerDataType::DATA_TYPE_INT128},
    {"hif8", CheckerDataType::DATA_TYPE_HIF8},
    {"fp8e4m3", CheckerDataType::DATA_TYPE_FP8E4M3},
    {"fp8e5m2", CheckerDataType::DATA_TYPE_FP8E5M2},
    {"reserve" , CheckerDataType::DATA_TYPE_RESERVED}
};

const std::map<std::string, CheckerDevType> testDevTypeMap = {
    {"910", CheckerDevType::DEV_TYPE_910},
    {"310p3", CheckerDevType::DEV_TYPE_310P3},
    {"910b", CheckerDevType::DEV_TYPE_910B},
    {"310p1", CheckerDevType::DEV_TYPE_310P1},
    {"910_93", CheckerDevType::DEV_TYPE_910_93},
    {"910_95", CheckerDevType::DEV_TYPE_910_95}
};

const std::map<std::string, CheckerOpType> testCmdTypeMap = {
    {"invalid", CheckerOpType::INVALID},
    {"broadcast" , CheckerOpType::BROADCAST},
    {"allreduce", CheckerOpType::ALLREDUCE},
    {"reduce", CheckerOpType::REDUCE},
    {"send", CheckerOpType::SEND},
    {"receive", CheckerOpType::RECEIVE},
    {"allgather", CheckerOpType::ALLGATHER},
    {"reducescatter", CheckerOpType::REDUCE_SCATTER},
    {"alltoallv", CheckerOpType::ALLTOALLV},
    {"alltoallvc", CheckerOpType::ALLTOALLVC},
    {"alltoall", CheckerOpType::ALLTOALL},
    {"scatter", CheckerOpType::SCATTER},
    {"batchsendrecv", CheckerOpType::BATCH_SEND_RECV},
    {"reducescatterv", CheckerOpType::REDUCE_SCATTER_V},
    {"allgatherv", CheckerOpType::ALLGATHER_V}
};

const std::map<std::string, CheckerOpMode> testOpModeMap = {
    {"opbase", CheckerOpMode::OPBASE},
    {"offload", CheckerOpMode::OFFLOAD}
};

long StrTolAlDigit(const char *optarg);
CheckerDevType GetDevType(char *str);
u64 ParseSize(const char *value);
CheckerDataType GetHcclDtype(char *str);
CheckerOpMode GetOpMode(char *str);
int PrintHelp();
int AsyParamTopo(TopoMeta &topoMeta);

class CheckerCmd
{
public:
    CheckerCmd();
    virtual ~CheckerCmd();

    static struct option longOpts[];

    int CheckCmdLine();
    int ParseCmdLine(int argc, char *argv[]);
    int CheckDataCount();
    int ParseOpt(int opt);
    CheckerReduceOp GetReduceOp(char *str);
    CheckerOpType GetCmdType(char *str);
    void PrintArgs(u64 count);

public:
    DataSize *data;
    struct SimpleParam uiParam;
    int superPodNum = 1;
    int serverNum = 1;
    int rankNum = 2;
    u64 dataParsedBegin = 64 * 1024 * 1024;
    u64 dataParsedEnd = 64 * 1024 * 1024;
    u64 stepBytes = 0;
    double stepFactor = 1.0;
    bool stepBytesFlag = false;
    bool stepfactorFlag = false;
    bool reduceOpFlag = false;
    bool asyParamFlag = false;
    bool rankMemCheckFlag = true;
    bool taskPrintFlag = false;
    bool zeroCopyFlag = false;
};

}
#endif // _CHECKER_CMD_H_