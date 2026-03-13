cmake_minimum_required(VERSION 3.14)

set(HCCL_VM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(HCCL_COMMON_PATH ${HCCL_VM_PATH}/hccl_common)
set(HCCL_HOST_PATH ${HCCL_VM_PATH}/hccl_host)
set(HCCL_IPC_PATH ${HCCL_VM_PATH}/hccl_ipc)
set(HCCL_SHM_PATH ${HCCL_VM_PATH}/hccl_shm)
set(HCCL_PROXY_PATH ${HCCL_VM_PATH}/hccl_proxy)
set(HCCL_VIR_RUNTIME_PATH ${HCCL_VM_PATH}/hccl_virtual_runtime)

# 自动扫描当前目录下的所有 .cc 文件
file(GLOB_RECURSE SRC_SOURCES "*.cc")

file(GLOB_RECURSE VRT_SOURCES "${HCCL_VIR_RUNTIME_PATH}/*.cc")
# 剔除hccl_virtual_runtime_main.cc文件
list(REMOVE_ITEM VRT_SOURCES "${HCCL_VIR_RUNTIME_PATH}/hccl_virtual_runtime_main.cc")

# 新增Virtual Runitme的源文件
list(APPEND SRC_SOURCES ${VRT_SOURCES})
list(APPEND SRC_SOURCES ${HCCL_COMMON_PATH}/hccl_comm_interface.cc)

# 添加可执行文件
add_executable(hccl_virtual_runtime_test ${SRC_SOURCES})
set_target_properties(hccl_virtual_runtime_test PROPERTIES COMPILE_FLAGS "-g -O0")
# 设置头文件包含路径（包含 lib 子目录）
target_include_directories(hccl_virtual_runtime_test PRIVATE
    ${HCCL_COMMON_PATH}
    ${HCCL_SHM_PATH}
    ${HCCL_IPC_PATH}
    ${HCCL_VIR_RUNTIME_PATH}
    ${HCCL_VIR_RUNTIME_PATH}/hccl_util
    ${HCCL_VIR_RUNTIME_PATH}/hccl_virtual_worker
    ../googletest/googlemock/include
)

link_directories(/usr/lib/x86_64-linux-gnu/)

# 链接依赖库
target_link_libraries(hccl_virtual_runtime_test PRIVATE
    gtest_main
    gmock_main
)

# 设置输出目录
set_target_properties(hccl_virtual_runtime_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output/bin
)