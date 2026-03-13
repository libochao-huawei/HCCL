#include "gtest/gtest.h"
#include <iostream>

GTEST_API_ int main(int argc, char **argv) {
    // testcase调试代码，只跑特定的用例
    // testing::GTEST_FLAG(filter) = "VirRuntimeAllreduceTest.*";
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}