#include "hstudio_func.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <dirent.h>
#include <stdio.h>
#include <sys/types.h>
#include "simple_param.h"
#include "checker_def.h"

using namespace hccl;

void ScanFiles(std::string dirPath, std::vector<std::string> &files, std::string suffix)
{
    DIR *dp = nullptr;
    struct dirent *fp = nullptr;

    dp = opendir(dirPath.c_str());  // 打开当前目录
    if (dp == nullptr) {
        return;
    }

    while ((fp = readdir(dp)) != nullptr) {
        if (fp->d_name[0] == '.') {
            continue;
        }
        if (fp->d_type == DT_REG) {
            char *ptr = strchr(fp->d_name, '.');
            if ((ptr == nullptr) || ((strcmp(ptr, suffix.c_str())) != 0)) {
                continue; // 后缀名不符合的跳过
            }
            string fullName = dirPath + "/" + fp->d_name;
            files.push_back(fullName);
        } else if (fp->d_type == DT_DIR) {
            string subDir = dirPath + "/" + fp->d_name;
            ScanFiles(subDir, files, suffix);
        }
    }
    closedir(dp);
}

static std::map<std::string, AlgCaseInfo> g_algTestCaseTable;

bool GetAllAlgTestCases(std::vector<std::string> &names)
{
    for (auto itor : g_algTestCaseTable) {
        names.push_back(itor.first);
    }
    return true;
}

bool GetAlgTestCaseInfo(std::string name, AlgCaseInfo &caseInfo)
{
    auto itor = g_algTestCaseTable.find(name);
    if (itor == g_algTestCaseTable.end()) {
        return false;
    }
    caseInfo = itor->second;
    return true;
}

bool RegAlgTestCase(const AlgCaseInfo &caseInfo)
{
    std::cout << "------------------ RegAlgTestCase:: " << caseInfo.name_ << " ----------------------------"
              << std::endl;
    // caseInfo.parameter_.Print();
    g_algTestCaseTable[caseInfo.name_] = caseInfo;
    return true;
}

bool CreateAlgTestCase(std::string name, AlgCaseParaS &para)
{
    auto itor = g_algTestCaseTable.find(name);
    if (itor != g_algTestCaseTable.end()) {
        return false;
    }

    AlgCaseInfo caseInfo;
    caseInfo.name_ = name;
    caseInfo.parameter_ = para;
    RegAlgTestCase(caseInfo);
    return true;
}

CheckerOpType HcclCMDTypeStr2Enum(std::string str)
{
    const std::map<std::string, CheckerOpType> mapping = {{"HCCL_CMD_INVALID", CheckerOpType::INVALID},
        {"HCCL_CMD_BROADCAST", CheckerOpType::BROADCAST},
        {"HCCL_CMD_ALLREDUCE", CheckerOpType::ALLREDUCE},
        {"HCCL_CMD_REDUCE", CheckerOpType::REDUCE},
        {"HCCL_CMD_SEND", CheckerOpType::SEND},
        {"HCCL_CMD_RECEIVE", CheckerOpType::RECEIVE},
        {"HCCL_CMD_ALLGATHER", CheckerOpType::ALLGATHER},
        {"HCCL_CMD_REDUCE_SCATTER", CheckerOpType::REDUCE_SCATTER},
        {"HCCL_CMD_ALLTOALLV", CheckerOpType::ALLTOALLV},
        {"HCCL_CMD_ALLTOALLVC", CheckerOpType::ALLTOALLVC},
        {"HCCL_CMD_ALLTOALL", CheckerOpType::ALLTOALL},
        {"HCCL_CMD_GATHER", CheckerOpType::GATHER},
        {"HCCL_CMD_SCATTER", CheckerOpType::SCATTER},
        {"HCCL_CMD_BATCH_SEND_RECV", CheckerOpType::BATCH_SEND_RECV},
        {"HCCL_CMD_BATCH_PUT", CheckerOpType::BATCH_PUT},
        {"HCCL_CMD_BATCH_GET", CheckerOpType::BATCH_GET},
        {"HCCL_CMD_ALLGATHER_V", CheckerOpType::ALLGATHER_V},
        {"HCCL_CMD_REDUCE_SCATTER_V", CheckerOpType::REDUCE_SCATTER_V},
        {"HCCL_CMD_ALL", CheckerOpType::ALL},
        {"HCCL_CMD_MAX", CheckerOpType::MAX}};

    auto itor = mapping.find(str);
    if (itor == mapping.end()) {
        return CheckerOpType::INVALID;
    }

    return itor->second;
}

CheckerOpMode OpModeStr2Enum(std::string str)
{
    const std::map<std::string, CheckerOpMode> mapping = {{"OPBASE", CheckerOpMode::OPBASE}, {"OFFLOAD", CheckerOpMode::OFFLOAD}};

    auto itor = mapping.find(str);
    if (itor == mapping.end()) {
        return CheckerOpMode::OPBASE;
    }

    return itor->second;
}

CheckerReduceOp HcclReduceOpStr2Enum(std::string str)
{
    const std::map<std::string, CheckerReduceOp> mapping = {{"HCCL_REDUCE_SUM", CheckerReduceOp::REDUCE_SUM},
        {"HCCL_REDUCE_PROD", CheckerReduceOp::REDUCE_PROD},
        {"HCCL_REDUCE_MAX", CheckerReduceOp::REDUCE_MAX},
        {"HCCL_REDUCE_MIN", CheckerReduceOp::REDUCE_MIN},
        {"HCCL_REDUCE_RESERVED", CheckerReduceOp::REDUCE_RESERVED}};

    auto itor = mapping.find(str);
    if (itor == mapping.end()) {
        return CheckerReduceOp::REDUCE_RESERVED;
    }

    return itor->second;
}

CheckerDevType DevTypeStr2Enum(std::string str)
{
    const std::map<std::string, CheckerDevType> mapping = {{"DEV_TYPE_910", CheckerDevType::DEV_TYPE_910},
        {"DEV_TYPE_310P3", CheckerDevType::DEV_TYPE_310P3},
        {"DEV_TYPE_910B", CheckerDevType::DEV_TYPE_910B},
        {"DEV_TYPE_310P1", CheckerDevType::DEV_TYPE_310P1},
        {"DEV_TYPE_910_93", CheckerDevType::DEV_TYPE_910_93},
        {"DEV_TYPE_NOSOC", CheckerDevType::DEV_TYPE_NOSOC},
        {"DEV_TYPE_COUNT", CheckerDevType::DEV_TYPE_COUNT}};

    auto itor = mapping.find(str);
    if (itor == mapping.end()) {
        return CheckerDevType::DEV_TYPE_COUNT;
    }

    return itor->second;
}

CheckerDataType HcclDataTypeStr2Enum(std::string str)
{
    const std::map<std::string, CheckerDataType> mapping = {
        {"HCCL_DATA_TYPE_INT8", CheckerDataType::DATA_TYPE_INT8},
        {"HCCL_DATA_TYPE_INT16", CheckerDataType::DATA_TYPE_INT16},
        {"HCCL_DATA_TYPE_INT32", CheckerDataType::DATA_TYPE_INT32},
        {"HCCL_DATA_TYPE_FP16", CheckerDataType::DATA_TYPE_FP16},
        {"HCCL_DATA_TYPE_FP32", CheckerDataType::DATA_TYPE_FP32},
        {"HCCL_DATA_TYPE_INT64", CheckerDataType::DATA_TYPE_INT64},
        {"HCCL_DATA_TYPE_UINT64", CheckerDataType::DATA_TYPE_UINT64},
        {"HCCL_DATA_TYPE_UINT8", CheckerDataType::DATA_TYPE_UINT8},
        {"HCCL_DATA_TYPE_UINT16", CheckerDataType::DATA_TYPE_UINT16},
        {"HCCL_DATA_TYPE_UINT32", CheckerDataType::DATA_TYPE_UINT32},
        {"HCCL_DATA_TYPE_FP64", CheckerDataType::DATA_TYPE_FP64},
        {"HCCL_DATA_TYPE_BFP16", CheckerDataType::DATA_TYPE_BFP16},
        {"HCCL_DATA_TYPE_INT128", CheckerDataType::DATA_TYPE_INT128},
        {"HCCL_DATA_TYPE_RESERVED", CheckerDataType::DATA_TYPE_RESERVED}};

    auto itor = mapping.find(str);
    if (itor == mapping.end()) {
        return CheckerDataType::DATA_TYPE_RESERVED;
    }

    return itor->second;
}

std::string HcclResult2Str(HcclResult errCode)
{
    const std::map<HcclResult, std::string> mapping = {
        {HcclResult::HCCL_SUCCESS, "success"},
        {HcclResult::HCCL_E_PARA, "parameter error"},
        {HcclResult::HCCL_E_PTR, "empty pointer"},
        {HcclResult::HCCL_E_MEMORY, "memory error"},
        {HcclResult::HCCL_E_INTERNAL, "internal error"},
        {HcclResult::HCCL_E_NOT_SUPPORT, "not support feature"},
        {HcclResult::HCCL_E_NOT_FOUND, "not found specific resource"},
        {HcclResult::HCCL_E_UNAVAIL, "resource unavailable"},
        {HcclResult::HCCL_E_SYSCALL, "call system interface error"},
        {HcclResult::HCCL_E_TIMEOUT, "timeout"},
        {HcclResult::HCCL_E_OPEN_FILE_FAILURE, "open file fail"},
        {HcclResult::HCCL_E_TCP_CONNECT, "tcp connect fail"},
        {HcclResult::HCCL_E_ROCE_CONNECT, "roce connect fail"},
        {HcclResult::HCCL_E_TCP_TRANSFER, "tcp transfer fail"},
        {HcclResult::HCCL_E_ROCE_TRANSFER, "roce transfer fail"},
        {HcclResult::HCCL_E_RUNTIME, "call runtime api fail"},
        {HcclResult::HCCL_E_DRV, "call driver api fail"},
        {HcclResult::HCCL_E_PROFILING, "call profiling api fail"},
        {HcclResult::HCCL_E_CCE, "call cce api fail"},
        {HcclResult::HCCL_E_NETWORK, "call network api fail"},
        {HcclResult::HCCL_E_AGAIN, "try again"},
        {HcclResult::HCCL_E_REMOTE, "error cqe"},
        {HcclResult::HCCL_E_RESERVED, "reserved"}
    };

    auto itor = mapping.find(errCode);
    if (itor == mapping.end()) {
        return "unknown";
    }

    return itor->second;
}

bool ParseOneCasePara(const nlohmann::json &caseObj, AlgCaseInfo& tmpCaseInfo)
{
    if (caseObj.at("topoConfigMode") == "ASYMMETRIC") {
        tmpCaseInfo.parameter_.isAsymmetricTopo = true;
        auto itorAsymTopo = caseObj.find("asymmetric_topo");
        if ((itorAsymTopo == caseObj.end()) || (!itorAsymTopo.value().is_array())) {
            std::cout << "invalid asymmetric_topo" << std::endl;
            return false;
        }
        for (const auto &itSuper : itorAsymTopo.value()) {
            if (!itSuper.is_array()) {
                std::cout << "invalid superPod" << std::endl;
                return false;
            }
            std::vector<std::vector<uint>> oneSuper;
            for (const auto &itServer : itSuper) {
                if (!itServer.is_array()) {
                    std::cout << "invalid server" << std::endl;
                    return false;
                }
                std::vector<uint> oneServer;
                for (const auto &oneRank : itServer) {
                    oneServer.push_back(static_cast<uint>(oneRank));
                }
                oneSuper.push_back(oneServer);
            }
            tmpCaseInfo.parameter_.asymmetricTopo.super_server_ranks.push_back(oneSuper);
        }
    } else {
        tmpCaseInfo.parameter_.isAsymmetricTopo = false;
        auto itorSymTopo = caseObj.find("symmetric_topo");
        if ((itorSymTopo == caseObj.end()) || (!itorSymTopo.value().is_array()) || (itorSymTopo.value().size() != 3)) {
            std::cout << "invalid symmetric_topo" << std::endl;
            return false;
        }
        tmpCaseInfo.parameter_.symmetricTopo.superPodNum = itorSymTopo.value().at(0);
        tmpCaseInfo.parameter_.symmetricTopo.serverNum = itorSymTopo.value().at(1);
        tmpCaseInfo.parameter_.symmetricTopo.rankNum = itorSymTopo.value().at(2);
    }

    tmpCaseInfo.parameter_.opType = HcclCMDTypeStr2Enum(caseObj.at("opType"));
    tmpCaseInfo.parameter_.opMode = OpModeStr2Enum(caseObj.at("opMode"));

    if (caseObj.contains("reduceType")) {
        tmpCaseInfo.parameter_.reduceType = HcclReduceOpStr2Enum(caseObj.at("reduceType"));
    }

    tmpCaseInfo.parameter_.devtype = DevTypeStr2Enum(caseObj.at("devtype"));
    if (caseObj.contains("algName")) {
        tmpCaseInfo.parameter_.algName = caseObj.at("algName");
    }
    tmpCaseInfo.parameter_.count = caseObj.at("count");
    tmpCaseInfo.parameter_.dataType = HcclDataTypeStr2Enum(caseObj.at("dataType"));

    if (caseObj.contains("rootRank")) {
        tmpCaseInfo.parameter_.root = caseObj.at("rootRank");
    }

    if (caseObj.contains("srcRank")) {
        tmpCaseInfo.parameter_.srcRank = caseObj.at("srcRank");
    }

    if (caseObj.contains("dstRank")) {
        tmpCaseInfo.parameter_.dstRank = caseObj.at("dstRank");
    }

    // 环境变量解析
    if (caseObj.contains("env")) {
        for (const auto& varObj : caseObj.at("env")) {
            tmpCaseInfo.parameter_.envVars[varObj.at("key")] = varObj.at("value");
        }
    }

    return true;
}

bool InitialAlgTest(std::string cfgpath, std::string filter)
{
    std::vector<std::string> files;
    ScanFiles(cfgpath, files, ".json");
    if (files.size() == 0) {
        return true;
    }

    for (auto file : files) {
        nlohmann::json doc;
        std::ifstream infile(file);
        infile >> doc;
        infile.close();

        const nlohmann::json::iterator  itorCaseList = doc.find("caselist");
        if ((itorCaseList == doc.end()) || (!itorCaseList.value().is_array())) {
            continue; // 识别文件格式，根节点有caselist列表才解析，其它json文件跟本业务无关直接跳过
        }

        for (const nlohmann::json &caseObj : itorCaseList.value()) {
            AlgCaseInfo tmpCaseInfo;
            tmpCaseInfo.name_ = caseObj.at("testSuite");
            tmpCaseInfo.name_ += ".";
            tmpCaseInfo.name_ += caseObj.at("testCase");
            if ((filter != "") && (tmpCaseInfo.name_ != filter)) {  // 如果指定只解析单个用例，跳过其它
                continue;
            }

            if (!ParseOneCasePara(caseObj, tmpCaseInfo)) {
                return false;
            }

            (void)RegAlgTestCase(tmpCaseInfo);
            if (filter != "") {  // 只解析指定的单个用例，执行一次便是结束
                return true;
            }
        }
    }

    return true;
}

static std::vector<u64> HStudioGenerateSendCountMatrix(u64 count, u32 rankSize)
{
    std::vector<u64> sendCountMatrix(rankSize * rankSize, count);
    return sendCountMatrix;
}

HcclResult ExecAlgTestCases(std::string name)
{
    AlgCaseInfo caseInfo;
    if (!GetAlgTestCaseInfo(name, caseInfo)) {
        std::cout << "testcase not exist: " << name << std::endl;
        return HCCL_E_OPEN_FILE_FAILURE;
    }
    AlgCaseParaS *para = &caseInfo.parameter_;

    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    if (para->isAsymmetricTopo) {
        topoMeta = para->asymmetricTopo.super_server_ranks;
    } else {
        gen.GenTopoMeta(topoMeta, para->symmetricTopo.superPodNum, para->symmetricTopo.serverNum, para->symmetricTopo.rankNum);
    }

    u32 rankNum = 0;
    for (auto& podMeta : topoMeta) {
        for (auto& serverMeta : podMeta) {
            rankNum += serverMeta.size();
        }
    }

    // config env vars
    for (const auto& envIter : para->envVars) {
        setenv(envIter.first.c_str(), envIter.second.c_str(), 1);
    }

    SimpleParam uiParam;
    uiParam.opType = para->opType;
    uiParam.algName = para->algName;
    uiParam.opMode = para->opMode;
    uiParam.reduceType = para->reduceType;
    uiParam.devtype = para->devtype;
    uiParam.is310P3V = para->is310P3V;
    uiParam.root = para->root;
    uiParam.dstRank = para->dstRank;
    uiParam.srcRank = para->srcRank;
    uiParam.count = para->count;
    uiParam.dataType = para->dataType;

    CheckerOpParam testOpParam;
    if (GenTestOpParams(rankNum, uiParam, testOpParam) != HCCL_SUCCESS) {
        std::cout << "invalid param" << std::endl;
        return HCCL_E_PARA;
    }

    Checker checker;
    checker.EnableGraphicDump();
    return checker.Check(testOpParam, topoMeta);
}

std::ofstream *g_outputFile = nullptr;

bool ProcCmdStart()
{
    std::vector<std::string> caseNames;
    (void)GetAllAlgTestCases(caseNames);

    *g_outputFile << "  case count : " << caseNames.size() << std::endl;
    *g_outputFile << "  case list :" << std::endl;
    for (auto &name : caseNames) {
        *g_outputFile << "    " << name << std::endl;
    }
    return true;
}

bool ProcCmdRun(std::string casename)
{
    std::vector<std::string> caseNames;
    (void)GetAllAlgTestCases(caseNames);
    std::cout << "------------------ ExecAlgTestCases:: " << casename << " ----------------------------" << std::endl;
    HcclResult ret = ExecAlgTestCases(casename);
    if (ret == HCCL_SUCCESS) {
        *g_outputFile << "  run case : " << casename << " OK" << std::endl;
    } else {
        *g_outputFile << "  run case : " << casename << " FAIL, ret = " << static_cast<int>(ret) << ", reason: " << HcclResult2Str(ret) << std::endl;
    }

    return true;
}

int ProcStudioCmd(std::string cmd, std::string casename)
{
    std::ofstream file;
    file.open("./output.txt", ios::app);
    if ((!file.is_open())) {
        std::cout << "open output file failed" << std::endl;
        return HSTUDIO_FAIL;
    }
    g_outputFile = &file;

    *g_outputFile << "HCCL STUDIO receive cmd : " << cmd << " " << casename << std::endl;

    bool ret = true;
    if (cmd == "-start") {
        ret = ProcCmdStart();
    } else if (cmd == "-run") {
        ret = ProcCmdRun(casename);
    } else {
        ret = false;
    }

    file.close();
    g_outputFile = nullptr;
    return ret ? HSTUDIO_OK : HSTUDIO_FAIL;
}

int HStudioMain(int argc, char **argv)
{
    if (argc == (PARA_INDEX_MAXNUM - 1)) {  // ./insight_adapter -start ./parameter_files
        (void)InitialAlgTest(argv[PARA_INDEX_CFGPATH]);
        return ProcStudioCmd(argv[PARA_INDEX_CMDTYPE]);
    } else if (argc == PARA_INDEX_MAXNUM) {  // ./insight_adapter -run ./parameter_files AllToAllTest.alltoall_test_910B_opbase
        (void)InitialAlgTest(argv[PARA_INDEX_CFGPATH], argv[PARA_INDEX_CASENAME]);
        return ProcStudioCmd(argv[PARA_INDEX_CMDTYPE], argv[PARA_INDEX_CASENAME]);
    }

    std::cout << "invalid argc for hccl studio cmd, argc = " << argc << std::endl;
    return HSTUDIO_FAIL;
}