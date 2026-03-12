#include "gtest/gtest.h"
// #include <mockcpp/mockcpp.hpp>
// #include <mockcpp/mokc.h>

class DemoTest : public testing::Test {
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
        std::cout << "A Test case in  DemoTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        // GlobalMockObject::verify();
        std::cout << "A Test case in  DemoTest TearDown" << std::endl;
    }
};

TEST_F(DemoTest, test1)
{
    EXPECT_EQ("111", "111");
}