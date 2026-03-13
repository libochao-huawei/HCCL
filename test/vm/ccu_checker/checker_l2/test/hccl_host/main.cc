#include <gtest/gtest.h>

int main(int argc, char *argv[]) {
    // testcase调试代码，只跑特定的用例
    // testing::GTEST_FLAG(filter) = "ST_SCATTER_TEST.*";
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}