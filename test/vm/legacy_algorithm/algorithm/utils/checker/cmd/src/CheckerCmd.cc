#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <algorithm>
#include "CheckerCmd.h"

#ifdef HCCL_ALG_ANALYZER_DAVID
namespace Hccl {
    extern std::string g_algNamev2;
}
#else
namespace hccl {
    extern std::string g_algName;
}
#endif

namespace checker {

const std::map<std::string, CheckerReduceOp> testReduceOpMap = {
    {"sum", CheckerReduceOp::REDUCE_SUM},
    {"prod", CheckerReduceOp::REDUCE_PROD},
    {"max", CheckerReduceOp::REDUCE_MAX},
    {"min", CheckerReduceOp::REDUCE_MIN},
    {"invalid", CheckerReduceOp::REDUCE_RESERVED}
};

CheckerCmd::CheckerCmd()
{
    data = new DataSize;
    data->dataStepFactor = 1;
}

CheckerCmd::~CheckerCmd()
{
    delete data;
    data = nullptr;
}

typedef signed int s32;

int CheckerCmd::ParseOpt(int opt)
{
    switch (opt) {
        case 'p':
            superPodNum = StrTolAlDigit(optarg);
            break;
        case 's':
            serverNum = StrTolAlDigit(optarg);
            break;
        case 'r':
            rankNum = StrTolAlDigit(optarg);
            break;
        case 'd':
            uiParam.devtype = GetDevType(optarg);
            break;
        case 't':
            uiParam.opType = GetCmdType(optarg);
            break;
        case 'm':
            uiParam.opMode = GetOpMode(optarg);
            break;
        case 'a':
            uiParam.root = StrTolAlDigit(optarg);
            break;
        case 'o':
            reduceOpFlag = false;
            uiParam.reduceType = GetReduceOp(optarg);
            break;
        case 'b':
            dataParsedBegin = ParseSize(optarg);
            break;
        case 'e':
            dataParsedEnd = ParseSize(optarg);
            break;
        case 'j':
            stepBytesFlag = true;
            stepBytes = ParseSize(optarg);
            break;
        case 'x':
            stepfactorFlag = true;
            char *temp;
            stepFactor = strtof(optarg, &temp);
            break;
        case 'n':
            uiParam.dataType =  GetHcclDtype(optarg);
            break;
        case 'c':
            rankMemCheckFlag = StrTolAlDigit(optarg);
            break;
        case 'q':
            asyParamFlag = StrTolAlDigit(optarg);
            break;
        case 'u':
            taskPrintFlag = StrTolAlDigit(optarg);
            break;
        case 'g':
            uiParam.algName = string(optarg);
            break;
        case 'z':
            zeroCopyFlag = StrTolAlDigit(optarg);
            break;
        case 'h':
            PrintHelp();
            std::exit(0);
        default:
            printf("invalid option \n");
            printf("Try [-h --help] for more information.\n");
            return EXIT_FAILURE;
    }
    return 0;
}

struct option CheckerCmd::longOpts[] = {
    {"superPodNum"  , no_argument, NULL, 'p'},
    {"serverNum"    , no_argument, NULL, 's'},
    {"rankNum"      , no_argument, NULL, 'r'},
    {"devType"      , required_argument, NULL, 'd'},
    {"opType"       , required_argument, NULL, 't'},
    {"opMode"       , required_argument, NULL, 'm'},
    {"root"         , no_argument, NULL, 'a'},
    {"op"           , required_argument, NULL, 'o'},
    {"minbytes"     , required_argument, NULL, 'b'},
    {"maxbytes"     , required_argument, NULL, 'e'},
    {"stepbytes"    , no_argument, NULL, 'j'},
    {"stepfactor"   , no_argument, NULL, 'x'},
    {"dataType"     , required_argument, NULL, 'n'},
    {"checkMem"     , no_argument, NULL, 'c'},
    {"asyParam"     , no_argument, NULL, 'q'},
    {"taskPrintFlag", no_argument, NULL, 'u'},
    {"algname"      , no_argument, NULL, 'g'},
    {"zeroCopy", no_argument, NULL, 'z'},
    {"help"         , no_argument,       NULL, 'h'},
    {NULL, 0, NULL, 0},
};

int CheckerCmd::ParseCmdLine(int argc, char *argv[])
{
    int opt = -1;
    int longindex = 0;
    int ret = 0;
    long parsed;

    while (-1 != (opt = getopt_long(argc, argv, "p:s:r:d:t:m:a:o:b:e:j:x:n:c:q:u:g:z:h", longOpts, &longindex))) {
        ret = ParseOpt(opt);
        if (ret != 0) {
            return ret;
        }
    }

    if (optind < argc) {
        printf("non-option ARGV-elements: ");
        while (optind < argc) {
            printf("%s ", argv[optind++]);
        }
        printf("\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int CheckerCmd::CheckCmdLine()
{
    int ret = 0;
    ret = CheckDataCount();
    if (ret != 0) {
        return ret;
    }

    if (superPodNum < 0) {
        printf("Error: [-p,--superPodNum] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return EXIT_FAILURE;
    }

    if (serverNum < 0) {
        printf("Error: [-s,--serverNum] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return EXIT_FAILURE;
    }

    if (rankNum < 0) {
        printf("Error: [-r,--rankNum] is invalid,  Use [-h,--help] to check the correct input parameter.\n");
        return EXIT_FAILURE;
    }

    if (uiParam.root < 0) {
        printf("Error: [-a,--root] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return EXIT_FAILURE;
    }

    if (uiParam.opMode < 0 || uiParam.opMode > 1) {
        printf("Error: [-m,--opMode] is invalid, Use [-h,--help] to check the correct input parameter.\n");
        return EXIT_FAILURE;
    }

    if (rankMemCheckFlag < 0 || rankMemCheckFlag > 1) {
        printf("Error: [-c,--rankMemCheckFlag] is invalid, Use [-h,--help] to check the correct input parameter.\n");
    }

    if (zeroCopyFlag != 0 && zeroCopyFlag != 1) {
        printf("Error: [-z,--zeroCopyFlag] is invalid, Use [-h,--help] to check the correct input parameter.\n");
    }

    return 0;
}

int CheckerCmd::CheckDataCount()
{
    if (dataParsedBegin < 0 || dataParsedEnd < 0) {
        printf("invalid size specified for [-b,--minbytes] or [-e,--maxbytes]\n");
        return EXIT_FAILURE;
    }
    data->minBytes = (u64)dataParsedBegin;
    data->maxBytes = (u64)dataParsedEnd;
    data->dataStepFactor = stepFactor;
    printf("[CheckerCmd][CheckDataCount] minBytes:%llu, maxBytes:%llu\n", data->minBytes, data->maxBytes);

    if (stepBytesFlag && stepBytes <= 0) {
        printf("Error: [-j,--stepbytes] must be greater than 0.\n");
        return EXIT_FAILURE;
    }

    if (stepfactorFlag && data->dataStepFactor <= 1.0) {
        printf("Error: [-x,--stepfactor] Must be greater than 1.0f, Start step mod.\n");
        return EXIT_FAILURE;
    }

    if (stepfactorFlag && stepBytesFlag) {
        printf("Warning: [-x,--stepfactor] and [-j,--stepbytes] are set, [-x,--stepfactor] is enabled by default.\n");
        return EXIT_FAILURE;
    }

    if (data->maxBytes < data->minBytes) {
        printf("invalid option: maxbytes < minbytes, Check the [-b,--minbytes] and [-e,--maxbytes] options.\n");
        return EXIT_FAILURE;
    }

    if (stepBytesFlag) { // 用户配置了增量步长
        data->dataStepBytes = stepBytes;
    } else if (!stepfactorFlag) { // 用户未配置增量步长
        if (data->maxBytes == data->minBytes) {
            data->dataStepBytes = 1; // 用户配置数据量的起始值和结束值相同，但未配置增量步长，为防止进入死循环，设置增量步长为1
            CHECKER_WARNING_LOG("[-j,--stepbytes] is not set, set to 1 as default.\n");
        } else if (data->maxBytes > data->minBytes) {
            if((data->maxBytes - data->minBytes) % 10 == 0) {
                data->dataStepBytes = (data->maxBytes - data->minBytes) / 10;
            } else {
                data->dataStepBytes = (data->maxBytes - data->minBytes) / 10 + 1;
            }
            CHECKER_WARNING_LOG("[-j,--stepbytes] is not set, set to (maxbytes - minbytes)/10 as default, stepBytes:%llu.\n", data->dataStepBytes);
        }
    }

    return 0;
}

u32 SalStrLen(const char *s, u32 maxLen = INT_MAX)
{
    return strnlen(s, maxLen);
}

int IsAllDigit(const char *strNum)
{
    // 参数有效性检查
    if (strNum == NULL) {
        printf("Error: strNum ptr is NULL\n");
        return EXIT_FAILURE;
    }
    u32 nLength = SalStrLen(strNum);
    for (u32 index = 0; index < nLength; index++) {
        if (!isdigit(strNum[index])) {
            printf("Error. The parameter %s is invalid. please check [-p -s -r -a]\n", strNum);
            return EXIT_FAILURE;
        }
    }
    return 0;
}

long StrTolAlDigit(const char *optarg)
{
    long ret = IsAllDigit(optarg);
    if (ret != 0) {
        return ret;
    }

    return strtol(optarg, NULL, 0);
}

CheckerDevType GetDevType(char *str)
{
    if (testDevTypeMap.find(str) != testDevTypeMap.end()) {
        return testDevTypeMap.at(str);
    }
    std::cout << "the devType is invalid, please check" << std::endl;
    return CheckerDevType::DEV_TYPE_NOSOC;
}

CheckerOpType CheckerCmd::GetCmdType(char *str)
{
    if (testCmdTypeMap.find(str) != testCmdTypeMap.end()) {
        if (testCmdTypeMap.at(str) == CheckerOpType::REDUCE_SCATTER_V
            || testCmdTypeMap.at(str) == CheckerOpType::REDUCE_SCATTER
            || testCmdTypeMap.at(str) == CheckerOpType::REDUCE
            || testCmdTypeMap.at(str) == CheckerOpType::ALLREDUCE) {
            reduceOpFlag = true;
        }
        return testCmdTypeMap.at(str);
    }
    std::cout << "the opType is invalid, please check" << std::endl;
    return CheckerOpType::INVALID;
}

CheckerReduceOp CheckerCmd::GetReduceOp(char *str)
{
    if(testReduceOpMap.find(str) != testReduceOpMap.end()) {
        return testReduceOpMap.at(str);
    }
    if(reduceOpFlag = true) {
        std::cout << "the reduceOp is invalid, set detault sum" << std::endl;
    }
    return CheckerReduceOp::REDUCE_SUM;
}

CheckerDataType GetHcclDtype(char *str) {
    if (testDtypeMap.find(str) != testDtypeMap.end()) {
        return testDtypeMap.at(str);
    }
    std::cout << "the dataType is invalid, please check" << std::endl;
    return CheckerDataType::DATA_TYPE_RESERVED;
}

CheckerOpMode GetOpMode(char *str) {
    if (testOpModeMap.find(str) != testOpModeMap.end()) {
        return testOpModeMap.at(str);
    }
    std::cout << "the opMode is invalid, set default opbase" << std::endl;
    return CheckerOpMode::OPBASE;
}

u64 ParseSize(const char *value)
{
    u64 units;
    u64 size;
    char *size_lit;

    size = strtol(value, &size_lit, 0);
    if (strlen(size_lit) == 1) {
        switch (*size_lit) {
            case 'G':
            case 'g':
                units = 1024 * 1024 * 1024;
                break;
            case 'M':
            case 'm':
                units = 1024 * 1024;
                break;
            case 'K':
            case 'k':
                units = 1024;
                break;
            default:
                return EXIT_FAILURE;
        }
    } else if (strlen(size_lit) == 0) {
        units = 1;
    } else {
        return EXIT_FAILURE;
    }

    return size * units;
}

int AsyParamTopo(TopoMeta &topoMeta)
{
    cmd::CmdExtParam asyParam;
    cmd::Pod* pod;
    cmd::Server* server;
    const char* filename = "asyParam.prototxt";
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        std::cout << "asyParam.prototxt open failed" << std::endl;
        return EXIT_FAILURE;
    }
    FileInputStream* input = new FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, &asyParam);
    cmd::AsymmetryTopo* topo = asyParam.mutable_topo();
    topoMeta.resize(topo->pods_size());
    for (int i = 0; i < topo->pods_size(); i++) {
        topoMeta[i].resize(topo->pods(i).servers_size());
        for(int j = 0; j < topo->pods(i).servers_size(); j++) {
            topoMeta[i][j].resize(topo->pods(i).servers(j).phyids_size());
            for(int k = 0; k < topo->pods(i).servers(j).phyids_size(); k++) {
                topoMeta[i][j][k] = topo->pods(i).servers(j).phyids(k);
            }
        }
    }
    delete input;
    close(fd);
    return 0;
}

void CheckerCmd::PrintArgs(u64 count) {
    for (const auto& pair : testCmdTypeMap) {
        if (pair.second == uiParam.opType) {
            std::cout << "[opType]: " << pair.first <<  ", ";
        }
    }
    for (const auto& pair : testDtypeMap) {
        if (pair.second == uiParam.dataType) {
            std::cout << "[dataType]: " << pair.first <<  ", ";
        }
    }
    for (const auto& pair : testOpModeMap) {
        if (pair.second == uiParam.opMode) {
            std::cout << "[OpMode]: " << pair.first <<  ", ";
        }
    }
    for (const auto& pair : testDevTypeMap) {
        if (pair.second == uiParam.devtype) {
            std::cout << "[devtype]: " << pair.first <<  ", ";
        }
    }
    if (uiParam.opType == CheckerOpType::REDUCE
        || uiParam.opType == CheckerOpType::SCATTER
        || uiParam.opType == CheckerOpType::BROADCAST) {
        std::cout << "[root]: " << uiParam.root <<  ", ";
    }
#ifdef HCCL_ALG_ANALYZER_DAVID
    std::cout << "[count]: " << count << ", [algName]: " << Hccl::g_algNamev2 << std::endl;
#else
    std::cout << "[count]: " << count << ", [algName]: " << hccl::g_algName << std::endl;
#endif
}

int PrintHelp() {
    printf("USAGE: ./hccl_alg_analyzer_test \n\t"
    "[-p,--npus <npus used for superPodNum>] \n\t"
    "[-s,--npus <npus used for serverNum>] \n\t"
    "[-r,--npus <npus used for rankNum>] \n\t"
    "[-d,--deviceType 1.0 : <310p1/310p3/910/910b/910_93> 2.0 : <910_95>] \n\t"
    "[-t,--opType <broadcast/allreduce/reduce/send/receive/allgather/reducescatter/alltoallv/alltoallvc/alltoall/"
    "scatter/batchsendrecv/reducescatterV/allGatherV>] \n\t"
    "[-m,--opMode <opbase/offload>] \n\t"
    "[-a,--root <root rank>] \n\t"
    "[-o,--op <sum/prod/min/max>] \n\t"
    "[-b,--minbytes <min size in bytes>] \n\t"
    "[-e,--maxbytes <max size in bytes>] \n\t"
    "[-j,--stepbytes <increment size>] \n\t"
    "[-x,--stepfactor <increment factor>] \n\t"
    "[-n,--datatype <int8/int16/int32/fp16/fp32/int64/uint64/uint8/uint16/uint32/fp64/bfp16>] \n\t"
    "[-c,--rankMemCheckFlag <0:false/1:true>] \n\t"
    "[-q,--asyTopoFlag <0:false/1:true>] \n\t"
    "[-u,--taskPrintFlag <0:false/1:true>] \n\t"
    "[-g,--algname <algorithm selection> ccu 2d case use <-g> algname] \n\t"
    "[-z,--zeroCopyFlag <0:false/1:true>] \n\t"
    "[-h,--help]\n");
    return 0;
}
}  // namespace HCCL