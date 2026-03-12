#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>
#include "common_interface.h"
#include <iostream>

namespace Hccl {
    void DumpCcuProfilingInfo(const std::vector<CcuProfilingInfo> &profInfos)
    {
        auto dumpLinkInfo = [] (const CcuProfilingInfo &info) -> void {
            for (int i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
                if (info.channelId[i] == INVALID_VALUE_CHANNELID) {
                    continue;
                }
                std::cout << "channelId(" << info.channelId[i] << "), remoteRankId(" << info.remoteRankId[i] << ")." << std::endl;
            }
        };
        std::cout<<"============================Dump CCU Profiling Info BEGIN============================\n";
        for (int i = 0; i < profInfos.size(); i++) {
            if (profInfos[i].type == CcuProfilinType::CCU_TASK_PROFILING) {
                std::cout << "SQE Profiling Info: ctxSignautre(" << profInfos[i].name << "), dieId("
                        << static_cast<int>(profInfos[i].dieId) << "), missionId("
                        << static_cast<int>(profInfos[i].missionId) << "), instrId("
                        << static_cast<int>(profInfos[i].instrId) << ")" << std::endl;
            } else if (profInfos[i].type == CcuProfilinType::CCU_WAITCKE_PROFILING) {
                std::cout << "Microcode SetCKE Profiling Info:  name(" << profInfos[i].name << "), dieId("
                        << static_cast<int>(profInfos[i].dieId) << "), missionId("
                        << static_cast<int>(profInfos[i].missionId) << "), instrId("
                        << static_cast<int>(profInfos[i].instrId) << "), ckeId(" << profInfos[i].ckeId << "), mask("
                        << profInfos[i].mask << ")" << std::endl;
                dumpLinkInfo(profInfos[i]);
            } else if (profInfos[i].type == CcuProfilinType::CCU_LOOPGROUP_PROFILING) {
                std::cout << "Microcode LoopGroup Profiling Info:  name(" << profInfos[i].name << "), dieId("
                        << static_cast<int>(profInfos[i].dieId) << "), missionId("
                        << static_cast<int>(profInfos[i].missionId) << "), instrId("
                        << static_cast<int>(profInfos[i].instrId) << "), reduceOpType("
                        << static_cast<int>(profInfos[i].reduceOpType) << "), inputDataType("
                        << static_cast<int>(profInfos[i].inputDataType) << "), outputDataType("
                        << static_cast<int>(profInfos[i].outputDataType) << "), dataSize(" << profInfos[i].dataSize
                        << ")" << std::endl;
                dumpLinkInfo(profInfos[i]);
            }
        }
        std::cout<<"============================Dump CCU Profiling Info END============================\n";
    }

    void CheckProfilingInfo(const std::vector<CcuProfilingInfo> &profInfos, uint32_t sqeCnt, uint32_t ckeCnt, uint32_t loopGroupCnt)
    {
        EXPECT_EQ(profInfos.size(), sqeCnt + ckeCnt + loopGroupCnt);

        if (sqeCnt != 0) {
            EXPECT_EQ(profInfos[0].type, CcuProfilinType::CCU_TASK_PROFILING);
        }

        uint32_t ckeCount = 0;
        uint32_t lgCount = 0;
        for (int i = 0; i < profInfos.size(); i++) {
            if (profInfos[i].type == CcuProfilinType::CCU_WAITCKE_PROFILING) {
                ckeCount++;
            } else if (profInfos[i].type == CcuProfilinType::CCU_LOOPGROUP_PROFILING) {
                lgCount++;
            }
        }
        EXPECT_EQ(ckeCount, ckeCnt);
        EXPECT_EQ(lgCount, loopGroupCnt);
    }
}