#ifndef OPS_HCCL_SRC_OPS_INC_COLL_OMNIPIPEDATASLICECALC
#define OPS_HCCL_SRC_OPS_INC_COLL_OMNIPIPEDATASLICECALC
#include <math.h>
#include<stdint.h>
#include <vector>
#include<string>
#include<sstream>
#include"template_utils.h"
#include "alg_template_base.h"
namespace ops_hccl{
constexpr u64 MAX_STEP_NUM = 5;
struct StepSliceInfo
{
    BuffInfo buffInfo;
    u64 count{0};
    u64 sliceSize{0};
    u64 inputSliceStride{0};
    u64 outputSliceStride{0};
    std::vector<u64> inputOmniPipeSliceStride;
    std::vector<u64> outputOmniPipeSliceStride;
};

struct OmniPipeSliceInfo
{
    std::vector<StepSliceInfo> dataSliceLevel0; //x轴每步数据偏移信息
    std::vector<StepSliceInfo> dataSliceLevel1; //y轴每步数据偏移信息
    std::vector<StepSliceInfo> dataSliceLevel2; //z轴每步数据偏移信息
    std::vector <std::vector <std::vector<u64>>> axlesReduceDstAddr;
    bool isEmpty() {
        if (dataSliceLevel0.empty() && dataSliceLevel1.empty() && dataSliceLevel2.empty()) {
            return true;
        }
        return false;
    }
    std::string toString() {
        std::ostringstream oss; // 使用字符串流来构建输出字符串

        // 输出 dataSliceLevel0 的内容
        oss << "dataSliceLevel0:\n";
        for (const auto& slice : dataSliceLevel0) {
            oss << "  - count: " << slice.count
                << ", sliceSize: " << slice.sliceSize
                << ", inputSliceStride: " << slice.inputSliceStride
                << ", outputSliceStride: " << slice.outputSliceStride
                << ", inputOmniPipeSliceStride: [";
            for (size_t i = 0; i < slice.inputOmniPipeSliceStride.size(); ++i) {
                oss << slice.inputOmniPipeSliceStride[i];
                if (i != slice.inputOmniPipeSliceStride.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], outputOmniPipeSliceStride: [";
            for (size_t i = 0; i < slice.outputOmniPipeSliceStride.size(); ++i) {
                oss << slice.outputOmniPipeSliceStride[i];
                if (i != slice.outputOmniPipeSliceStride.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "]\n";
        }

        // 输出 dataSliceLevel1 的内容
        oss << "dataSliceLevel1:\n";
        for (const auto& slice : dataSliceLevel1) {
            oss << "  - count: " << slice.count
                << ", sliceSize: " << slice.sliceSize
                << ", inputSliceStride: " << slice.inputSliceStride
                << ", outputSliceStride: " << slice.outputSliceStride
                << ", inputOmniPipeSliceStride: [";
            for (size_t i = 0; i < slice.inputOmniPipeSliceStride.size(); ++i) {
                oss << slice.inputOmniPipeSliceStride[i];
                if (i != slice.inputOmniPipeSliceStride.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], outputOmniPipeSliceStride: [";
            for (size_t i = 0; i < slice.outputOmniPipeSliceStride.size(); ++i) {
                oss << slice.outputOmniPipeSliceStride[i];
                if (i != slice.outputOmniPipeSliceStride.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "]\n";
        }

        // 输出 dataSliceLevel2 的内容
        oss << "dataSliceLevel2:\n";
        for (const auto& slice : dataSliceLevel2) {
            oss << "  - count: " << slice.count
                << ", sliceSize: " << slice.sliceSize
                << ", inputSliceStride: " << slice.inputSliceStride
                << ", outputSliceStride: " << slice.outputSliceStride
                << ", inputOmniPipeSliceStride: [";
            for (size_t i = 0; i < slice.inputOmniPipeSliceStride.size(); ++i) {
                oss << slice.inputOmniPipeSliceStride[i];
                if (i != slice.inputOmniPipeSliceStride.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], outputOmniPipeSliceStride: [";
            for (size_t i = 0; i < slice.outputOmniPipeSliceStride.size(); ++i) {
                oss << slice.outputOmniPipeSliceStride[i];
                if (i != slice.outputOmniPipeSliceStride.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "]\n";
        }

        return oss.str(); // 返回构建好的字符串
    }
};

// 计算SliceInfo的入参
struct OmniPipeSliceParam {
    std::vector<u64> levelRankSize;                     // 依次为三个维度的rankSize
    std::vector<EndpointAttrBwCoeff> endpointAttrBw;    // 依次为三个维度的平均带宽
    u64 dataSizePerLoop{0};                             // 一次loop数据量大小
    u64 dataTypeSize{0};                                // 数据类型大小
    u64 dataWholeSize{0};                               // 如果是aicpu的单算子模式，和dataSize一样，每次都在ccl做。其他模式都是总数据量。
    std::vector<u64> levelRankId;                       // 依次为本rank在三个维度的rankID
    std::vector<u64> levelAlgType;                      // 依次为三个维度的算法类型，MESH是1 or NHR是0
    OpMode opMode;
    CommEngine engine;
    std::string toString() {
        std::ostringstream oss;
        // 输出 levelRankSize
        oss << "levelRankSize: [";
        for (size_t i = 0; i < levelRankSize.size(); ++i) {
            oss << levelRankSize[i];
            if (i != levelRankSize.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]\n";
        // 输出 endpointAttrBw
        oss << "endpointAttrBw: [";
        for (size_t i = 0; i < endpointAttrBw.size(); ++i) {
            oss << endpointAttrBw[i];
            if (i != endpointAttrBw.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]\n";
        // 输出 dataSizePerLoop, dataTypeSize, dataWholeSize
        oss << "dataSizePerLoop: " << dataSizePerLoop << "\n";
        oss << "dataTypeSize: " << dataTypeSize << "\n";
        oss << "dataWholeSize: " << dataWholeSize << "\n";
        // 输出 levelRankId
        oss << "levelRankId: [";
        for (size_t i = 0; i < levelRankId.size(); ++i) {
            oss << levelRankId[i];
            if (i != levelRankId.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]\n";
        // 输出 levelAlgType
        oss << "levelAlgType: [";
        for (size_t i = 0; i < levelAlgType.size(); ++i) {
            oss << levelAlgType[i];
            if (i != levelAlgType.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]\n";
        return oss.str();
    }
};

// 计算ScratchInfo的入参，只给RS用
struct OmniPipeScratchParam {
    std::vector<u64> levelRankSize;                     // 依次为三个维度的rankSize
    std::vector<EndpointAttrBwCoeff> endpointAttrBw;    // 依次为三个维度的平均带宽
    u64 dataSize{0};                                    // 数据量大小，是单卡做完的数据量
    u64 dataTypeSize{0};                                // 数据类型大小
    u64 maxTmpMemSize{0};                               // 最大scratch大小
    std::vector<u64> levelAlgType;                      // 依次为三个维度的算法类型，MESH是1 or NHR是0
    OpMode opMode;
    CommEngine engine;
    std::string toString() {
        std::ostringstream oss;
        // 输出 levelRankSize
        oss << "levelRankSize: [";
        for (size_t i = 0; i < levelRankSize.size(); ++i) {
            oss << levelRankSize[i];
            if (i != levelRankSize.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]\n";

        // 输出 endpointAttrBw
        oss << "endpointAttrBw: [";
        for (size_t i = 0; i < endpointAttrBw.size(); ++i) {
            oss << endpointAttrBw[i];
            if (i != endpointAttrBw.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]\n";
        // 输出 dataSize
        oss << "dataSize: " << dataSize << "\n";
        // 输出 dataTypeSize
        oss << "dataTypeSize: " << dataTypeSize << "\n";
        // 输出 maxTmpMemSize
        oss << "maxTmpMemSize: " << maxTmpMemSize << "\n";
        // 输出 levelAlgType
        oss << "levelAlgType: [";
        for (size_t i = 0; i < levelAlgType.size(); ++i) {
            oss << levelAlgType[i];
            if (i != levelAlgType.size() - 1) {
                oss << ", ";
            }
        }
        oss << "]\n";
        // // 输出 opMode
        // oss << "opMode: ";
        // switch (opMode) {
        //     case OpMode::Mode1:
        //         oss << "Mode1";
        //         break;
        //     case OpMode::Mode2:
        //         oss << "Mode2";
        //         break;
        //     default:
        //         oss << "Unknown";
        // }
        // oss << "\n";
        // // 输出 engine
        // oss << "engine: ";
        // switch (engine) {
        //     case CommEngine::Engine1:
        //         oss << "Engine1";
        //         break;
        //     case CommEngine::Engine2:
        //         oss << "Engine2";
        //         break;
        //     default:
        //         oss << "Unknown";
        // }
        oss << "\n";

        return oss.str();
    }
};

void BuffInfoAssign(BuffInfo &bi,u64 inBuffBaseOff,u64 outBuffBaseOff,u64 hcclBuffBaseOff=0);
void StepSliceInfoAssign(StepSliceInfo &ssi,BuffInfo buffInfo,u64 count, u64 sliceSize,u64 inputSliceStride,u64 outputSliceStride);
double SumDataSize(u64* datastart,u64 num);

double CalcBandwidth2D(double xB,double yB,u64 xRankSize,u64 yRankSize,int maxStepNum);
void CalAllgather2DOffset(u64* xAGOffect,u64*yAGOffect,u64 stepNum,u64 xRankSize,u64 yRankSize, u64*xAGDataSize,u64*yAGDataSize);
u64 CalAllgatherDataSizeRatio2D(double* xStepP2pDataSize,double* yStepP2pDataSize,double xB,double yB,u64 j,u64 i,double dataSize,u64 maxStep);
u64 CalAllgatherDataSize2D(u64* xStepP2pDataSize,u64* yStepP2pDataSize,double xB,double yB,u64 xRankSize,u64 yRankSize,u64 dataSizeEachRank,u64 maxStep);
OmniPipeSliceInfo CalcAGOmniPipeSliceInfo(OmniPipeSliceParam &omniPipeSliceParam);

std::vector<u64> CalScratchSize(u64*xRSDataSize,u64*yRSDataSize,u64*zRSDataSize ,std::vector<u64> levelRankSize,u64 cornerStep,u64 outerStepNum,u64 innerStepNum,u64 maxStepNum ,std::vector<u64> levelAlgType,CommEngine engine);
std::vector <std::vector<u64>> CalRSDataSizeStep(u64*xRSDataSize,u64*yRSDataSize,u64*zRSDataSize ,std::vector<u64> levelRankSize,u64 cornerStep,u64 outerStepNum,u64 innerStepNum,u64 maxStepNum);
void CalReducescatter2DOffset(u64* xRSOffect,u64*yRSOffect,u64 stepNum,u64 xRankSize,u64 yRankSize, u64*xRSDataSize,u64*yRSDataSize);
u64 CalReducescatterDataSizeRatio2D(double* xStepP2pDataSize,double* yStepP2pDataSize,double xB,double yB,u64 j,u64 i,double dataSize,u64 maxStep);
u64 CalReducescatterDataSize2D(u64* xStepP2pDataSize,u64* yStepP2pDataSize,double xB,double yB,u64 xRankSize,u64 yRankSize,u64 dataSizeEachRank,u64 maxStep);
std::vector<u64> CalcOmniPipeScratchInfo(OmniPipeScratchParam &omniPipeScratchParam);
OmniPipeSliceInfo CalcRSOmniPipeSliceInfo(OmniPipeSliceParam &omniPipeSliceParam);
}
#endif