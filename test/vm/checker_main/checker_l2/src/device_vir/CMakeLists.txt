cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    message(FATAL_ERROR "Error: ASCEND_TOOLKIT_HOME environment variable is not set. Please set it to the Ascend Toolkit installation path.")
    endif()
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_TOOLKIT_HOME})
endif()

message(STATUS "set ASCEND_CANN_PACKAGE_PATH to ${ASCEND_CANN_PACKAGE_PATH}")

# 自动扫描当前目录下的所有 .cc 文件
file(GLOB_RECURSE SRC_SOURCES "*.cc")

# 添加可执行文件
add_executable(device_vir ${SRC_SOURCES})

set(TOP_DIR           ${PROJECT_SOURCE_DIR})
# 设置头文件包含路径（关键添加：${CMAKE_SOURCE_DIR}/hccl_common）
target_include_directories(device_vir PUBLIC
    ${TOP_DIR}/include
    ${TOP_DIR}/include/runnerdb
    ${TOP_DIR}/include/store

    # driver头文件
    ${TOP_DIR}/drivers/network/inc/network
    ${TOP_DIR}/drivers/network/orion/hcomm_dev/inc/network

    # cann包头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts

    # hccl头文件
    ${TOP_DIR}/../../../hccl/src/ops/inc/
    ${TOP_DIR}/../../../hccl/src/common/
    ${TOP_DIR}/../../../hcomm/include/hccl/
)

target_compile_options(device_vir PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
    -D_GLIBCXX_USE_CXX11_ABI=0
)

# 链接依赖库
target_link_libraries(device_vir PRIVATE
    dl
    rt
    runnerdb
    store
)

# 设置可执行程序输出目录
set_target_properties(device_vir PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output
)