cmake_minimum_required(VERSION 3.14)

set(HOME "${CMAKE_CURRENT_LIST_DIR}")
set(PROCESS_TYPE "hccl_shm_test")

set(SRC_FILES
    "${HOME}/*.cc"
    "${HOME}/../../hccl_shm/*.cc"
)

file(GLOB_RECURSE CC_FILES ${SRC_FILES})

set(SOURCE_LIST ${CC_FILES})
set_source_files_properties(${SOURCE_LIST} PROPERTIES LANGUAGE CXX)

set(INCLUDE_DIRS
    "${HOME}/../../hccl_common"
    "${HOME}/../../hccl_shm"
    "../googletest/googlemock/include"
)
include_directories(${INCLUDE_DIRS})

add_executable(${PROCESS_TYPE} ${SOURCE_LIST})

target_link_libraries(${PROCESS_TYPE} PRIVATE
    gtest_main
    gmock_main
    boost_system
    boost_thread
    boost_atomic
    rt
)
set_target_properties(${PROCESS_TYPE} PROPERTIES COMPILE_FLAGS "-g -O0")

# 设置输出目录
set_target_properties(${PROCESS_TYPE} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)