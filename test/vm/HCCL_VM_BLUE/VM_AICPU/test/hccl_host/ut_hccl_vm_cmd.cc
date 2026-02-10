#include "gtest/gtest.h"
// #include <mockcpp/mockcpp.hpp>
// #include <mockcpp/mokc.h>
#include "hccl_vm.h"
#include "hccl_shm_pub.h"


class UtHcclVmCmdTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << " DemoTest tests set up." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << " DemoTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        TopoMeta singlePod = {
            { // 第一个 SuperPodMeta
                {1, 2, 3, 4} // 第一个 ServerMeta
            }
        };
        std::cout << "A Test case in  DemoTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        // GlobalMockObject::verify();
        std::cout << "A Test case in  DemoTest TearDown" << std::endl;
    }
};

TEST_F(UtHcclVmCmdTest, ValidateInputs)
{
    EXPECT_EQ("111", "111");
}