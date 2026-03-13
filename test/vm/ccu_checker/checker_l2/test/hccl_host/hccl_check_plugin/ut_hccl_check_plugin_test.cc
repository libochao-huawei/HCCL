#include "gtest/gtest.h"
#define private public
#include "hccl_check_plugin_manager.h"
#undef private
#include "hccl_check_plugin.h"

#include <string>

using namespace HcclSim;

class HcclCheckPluginSuite : public ::testing::Test {
protected:
    void SetUp() {
    }
    void TearDown() {
    }
};

const std::string BASE_PATH = "../hccl_host/hccl_check_plugin/solib";

// 正常流程 注册-使用-卸载
TEST_F(HcclCheckPluginSuite, HcclPluginManagerBase)
{
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    HcclPlugin plguin(BASE_PATH + "/libTTT.so");
    HcclVmResult ret;
    ret = pluginManager.RegisterPlugin(plguin, "TTT");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);

    std::vector<std::string> tags = pluginManager.GetRegisteredPluginTags();
    EXPECT_EQ(tags[0], "TTT");

    std::vector<HcclVmResult> retVec;
    retVec = pluginManager.RunPlugins({"TTT"});
    EXPECT_EQ(retVec[0], HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
    retVec = pluginManager.UnloadPlugins({"TTT"});
    EXPECT_EQ(retVec[0], HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
}

// 注册失败
TEST_F(HcclCheckPluginSuite, HcclPluginManagerRegisterFail)
{
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    HcclPlugin plguin(BASE_PATH + "/libUnknown.so");
    HcclVmResult ret;
    ret = pluginManager.RegisterPlugin(plguin, "Unknown");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_ERROR_PARSE_CMD);
}

// 重复注册
TEST_F(HcclCheckPluginSuite, HcclPluginManagerMultiRegister)
{
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    HcclPlugin plguin(BASE_PATH + "/libTTT.so");
    HcclVmResult ret;
    ret = pluginManager.RegisterPlugin(plguin, "TTT");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
    ret = pluginManager.RegisterPlugin(plguin, "TTT");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_ERROR_PARSE_CMD);

    std::vector<HcclVmResult> retVec;
    retVec = pluginManager.UnloadPlugins({"TTT"});
    EXPECT_EQ(retVec[0], HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
}

// 加载失败 不存在Tag
TEST_F(HcclCheckPluginSuite, HcclPluginManagerLoadFail)
{
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    HcclVmResult ret;
    ret = pluginManager.LoadPlugins("Unknown");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_ERROR_PARSE_CMD);
}

// 执行失败 缺失函数
TEST_F(HcclCheckPluginSuite, HcclPluginManagerRunSymFail)
{
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    HcclPlugin plguin_MTT(BASE_PATH + "/libMTT.so");
    HcclVmResult ret;
    ret = pluginManager.RegisterPlugin(plguin_MTT, "MTT");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);

    HcclPlugin plguin_TMT(BASE_PATH + "/libTMT.so");
    ret = pluginManager.RegisterPlugin(plguin_TMT, "TMT");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);

    HcclPlugin plguin_TTM(BASE_PATH + "/libTTM.so");
    ret = pluginManager.RegisterPlugin(plguin_TTM, "TTM");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);

    std::vector<HcclVmResult> retVec;
    retVec = pluginManager.RunPlugins({"MTT", "TMT", "TTM"});
    EXPECT_EQ(retVec[0], HcclVmResult::HCCL_E_PTR);
    EXPECT_EQ(retVec[1], HcclVmResult::HCCL_E_PTR);
    EXPECT_EQ(retVec[2], HcclVmResult::HCCL_E_PTR);

    retVec = pluginManager.UnloadPlugins({"MTT", "TMT", "TTM"});
    EXPECT_EQ(retVec[0], HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
    EXPECT_EQ(retVec[1], HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
    EXPECT_EQ(retVec[2], HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
}

// 执行失败 插件内部失败
TEST_F(HcclCheckPluginSuite, HcclPluginManagerRunSoFail)
{
    HcclPluginManager &pluginManager = HcclPluginManager::GetInstance();
    HcclPlugin plguin_FTT(BASE_PATH + "/libFTT.so");
    HcclVmResult ret;
    ret = pluginManager.RegisterPlugin(plguin_FTT, "FTT");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);

    HcclPlugin plguin_TFT(BASE_PATH + "/libTFT.so");
    ret = pluginManager.RegisterPlugin(plguin_TFT, "TFT");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);

    HcclPlugin plguin_TTF(BASE_PATH + "/libTTF.so");
    ret = pluginManager.RegisterPlugin(plguin_TTF, "TTF");
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);

    std::vector<HcclVmResult> retVec;
    retVec = pluginManager.RunPlugins({"FTT", "TFT", "TTF"});
    EXPECT_EQ(retVec[0], HcclVmResult::HCCL_SIM_HOST_ERROR_PARSE_CMD);
    EXPECT_EQ(retVec[1], HcclVmResult::HCCL_SIM_HOST_ERROR_PARSE_CMD);
    EXPECT_EQ(retVec[2], HcclVmResult::HCCL_SIM_HOST_ERROR_PARSE_CMD);

    retVec = pluginManager.UnloadPlugins({"FTT", "TFT", "TTF"});
    EXPECT_EQ(retVec[0], HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
    EXPECT_EQ(retVec[1], HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
    EXPECT_EQ(retVec[2], HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD);
}

