#include "gtest/gtest.h"
#include "hstudio_func.h"

GTEST_API_ int main(int argc, char **argv) {
    return HStudioMain(argc, argv); // 可视化工具调用分支，命令参数隔离，原gtest用例无影响
}