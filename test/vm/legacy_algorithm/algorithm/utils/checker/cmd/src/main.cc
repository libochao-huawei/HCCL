#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include "CheckerCmd.h"

using namespace hccl;
using namespace checker;

int main(int argc, char *argv[])
{
    std::unique_ptr<CheckerCmd> checkerCmd(new CheckerCmd());
    if(checkerCmd == nullptr) {
        printf("checkerCmd is null!\n");
        return EXIT_FAILURE;
    }
    if (checkerCmd->ParseCmdLine(argc, argv) != EXIT_SUCCESS) {
        printf("ParseCmdLine failed!\n");
        return EXIT_FAILURE;
    }
    if (checkerCmd->CheckCmdLine() != EXIT_SUCCESS) {
        printf("CheckCmdLine failed!\n");
        return EXIT_FAILURE;
    }
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    if (checkerCmd->asyParamFlag) {
        CHECKER_WARNING_LOG("asyParamFlag is set to be true, Asymmetric topo will be used.");
        AsyParamTopo(topoMeta);
    } else {
        gen.GenTopoMeta(topoMeta, checkerCmd->superPodNum, checkerCmd->serverNum, checkerCmd->rankNum);
    }
    u32 rankSize = GetRankNumFormTopoMeta(topoMeta);
    u64 dataSize = checkerCmd->data->minBytes;
    for (; dataSize <= checkerCmd->data->maxBytes;
        (checkerCmd->stepfactorFlag == false ? dataSize += checkerCmd->data->dataStepBytes : dataSize *= checkerCmd->data->dataStepFactor)) {
        struct CheckerOpParam testOpParam;
        checkerCmd->uiParam.count = dataSize / CHECK_SIZE_TABLE[checkerCmd->uiParam.dataType];
        if (GenTestOpParams(rankSize, checkerCmd->uiParam, testOpParam) != HCCL_SUCCESS) {
            std::cout << "invalid param" << std::endl;
            return HCCL_E_PARA;
        }

        Checker checker;
        if (!(checkerCmd->rankMemCheckFlag)) {
            checker.CloseRankMemCheck();
        }

        if (checkerCmd->taskPrintFlag) {
            checker.EnableTaskPrint();
        }

        if (checkerCmd->zeroCopyFlag) {
            testOpParam.supportZeroCopy = checkerCmd->zeroCopyFlag;
            testOpParam.isZeroCopy = checkerCmd->zeroCopyFlag;
        }
#ifdef HCCL_ALG_ANALYZER_DAVID
        HcclResult ret = checker.CheckA5Aicpu(testOpParam, topoMeta);
#else
        HcclResult ret = checker.Check(testOpParam, topoMeta);
#endif
        checkerCmd->PrintArgs(checkerCmd->uiParam.count);
        if (ret != HCCL_SUCCESS) {
            std::cout << "[check failed]" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "[check success]" << std::endl;
        std::cout << "---------------------------------------" << std::endl;
    }
    return EXIT_SUCCESS;
}