# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(HCCL_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(HCOMM_ROOT_DIR ${HCCL_ROOT_DIR}/../hcomm)

find_path(MC2_HCCL_TYPES_INCLUDE_DIR
    NAMES hccl/hccl_types.h
    HINTS
        ${HCOMM_ROOT_DIR}/include
        ${CMAKE_SOURCE_DIR}/../hcomm/include
        ${CMAKE_SOURCE_DIR}/../../hcomm/include
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/hcomm/include
        ${ASCEND_INSTALL_PATH}/include
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/hcomm/include
        $ENV{ASCEND_INSTALL_PATH}/include
        $ENV{ASCEND_TOOLKIT_HOME}/include
        $ENV{ASCEND_TOOLKIT_HOME}/hcomm/include
)

if(NOT MC2_HCCL_TYPES_INCLUDE_DIR)
    message(FATAL_ERROR
        "Cannot find header 'hccl/hccl_types.h' for hccl_mc2 UT. "
        "Please make sure hcomm headers are available, e.g. set one of: "
        "HCOMM sibling repo, ASCEND_HOME_PATH, ASCEND_INSTALL_PATH, ASCEND_TOOLKIT_HOME.")
endif()
message(STATUS "[hccl_mc2_ut] hccl_types include root: ${MC2_HCCL_TYPES_INCLUDE_DIR}")

find_path(MC2_DLOG_INCLUDE_DIR
    NAMES dlog_pub.h
    HINTS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HCOMM_ROOT_DIR}/test/ut/depends/pkg_inc/base
        ${HCOMM_ROOT_DIR}/src/platform/hccp/external_depends/inc/log
        ${ASCEND_HOME_PATH}/pkg_inc/base
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
        $ENV{ASCEND_HOME_PATH}/pkg_inc/base
        $ENV{ASCEND_INSTALL_PATH}/pkg_inc/base
        $ENV{ASCEND_TOOLKIT_HOME}/pkg_inc/base
)
if(NOT MC2_DLOG_INCLUDE_DIR)
    message(FATAL_ERROR
        "Cannot find header 'dlog_pub.h' for hccl_mc2 UT. "
        "Please make sure base headers are available (e.g. pkg_inc/base).")
endif()
message(STATUS "[hccl_mc2_ut] dlog include root: ${MC2_DLOG_INCLUDE_DIR}")

set(MC2_FALLBACK_INCLUDE_DIR_CANDIDATES
    ${MC2_HCCL_TYPES_INCLUDE_DIR}
    ${MC2_DLOG_INCLUDE_DIR}
    ${HCOMM_ROOT_DIR}/include
    ${HCOMM_ROOT_DIR}/pkg_inc
    ${HCOMM_ROOT_DIR}/pkg_inc/base
    ${HCOMM_ROOT_DIR}/inc
    ${HCOMM_ROOT_DIR}/src/platform/hccp/external_depends/inc/log
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/hcomm/include
    ${ASCEND_INSTALL_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/base
    ${ASCEND_HOME_PATH}/pkg_inc/hccl
    ${ASCEND_HOME_PATH}/pkg_inc/hcomm
    ${ASCEND_HOME_PATH}/pkg_inc/hcomm/ccu
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/hcomm/include
    $ENV{ASCEND_INSTALL_PATH}/include
    $ENV{ASCEND_HOME_PATH}/pkg_inc/base
    $ENV{ASCEND_INSTALL_PATH}/pkg_inc/base
    $ENV{ASCEND_TOOLKIT_HOME}/pkg_inc/base
    $ENV{ASCEND_TOOLKIT_HOME}/include
    $ENV{ASCEND_TOOLKIT_HOME}/hcomm/include
)
set(MC2_FALLBACK_INCLUDE_DIRS)
foreach(path ${MC2_FALLBACK_INCLUDE_DIR_CANDIDATES})
    if(IS_DIRECTORY ${path})
        list(APPEND MC2_FALLBACK_INCLUDE_DIRS ${path})
    endif()
endforeach()
list(REMOVE_DUPLICATES MC2_FALLBACK_INCLUDE_DIRS)

file(GLOB_RECURSE OP_COMMON_PATHS LIST_DIRECTORIES true
    ${HCCL_ROOT_DIR}/src/ops/op_common/*
)
set(OP_COMMON_INCLUDE_DIRS)
foreach(path ${OP_COMMON_PATHS})
    if(IS_DIRECTORY ${path})
        list(APPEND OP_COMMON_INCLUDE_DIRS ${path})
    endif()
endforeach()
list(APPEND OP_COMMON_INCLUDE_DIRS ${HCCL_ROOT_DIR}/src/ops/op_common)
list(REMOVE_DUPLICATES OP_COMMON_INCLUDE_DIRS)

file(GLOB_RECURSE HCOMM_PATHS LIST_DIRECTORIES true
    ${HCOMM_ROOT_DIR}/include/*
    ${HCOMM_ROOT_DIR}/pkg_inc/*
    ${HCOMM_ROOT_DIR}/inc/*
    ${HCOMM_ROOT_DIR}/test/ut/depends/include/*
    ${HCOMM_ROOT_DIR}/test/ut/depends/pkg_inc/*
)
set(HCOMM_INCLUDE_DIRS)
foreach(path ${HCOMM_PATHS})
    if(IS_DIRECTORY ${path})
        list(APPEND HCOMM_INCLUDE_DIRS ${path})
    endif()
endforeach()
list(APPEND HCOMM_INCLUDE_DIRS
    ${HCOMM_ROOT_DIR}/include
    ${HCOMM_ROOT_DIR}/pkg_inc
    ${HCOMM_ROOT_DIR}/inc
    ${HCOMM_ROOT_DIR}/test/ut/depends/include
    ${HCOMM_ROOT_DIR}/test/ut/depends/pkg_inc
)
list(REMOVE_DUPLICATES HCOMM_INCLUDE_DIRS)

set(HCCL_MC2_UT_SRCS
    hccl_mc2_test.cc
    hccl_mc2_stub.cc
    ${HCCL_ROOT_DIR}/src/common/hccl_mc2.cc
)

add_executable(hccl_mc2_test ${HCCL_MC2_UT_SRCS})

target_include_directories(hccl_mc2_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${UT_COMMON_INCLUDE_LIST}
    ${HCCL_ROOT_DIR}/include
    ${HCCL_ROOT_DIR}/src/common
    ${HCCL_ROOT_DIR}/src/common/hcomm_dlsym
    ${MC2_FALLBACK_INCLUDE_DIRS}
    ${OP_COMMON_INCLUDE_DIRS}
    ${HCOMM_INCLUDE_DIRS}
)

target_compile_options(hccl_mc2_test PRIVATE
    -U_FORTIFY_SOURCE
    -w
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)

target_link_libraries(hccl_mc2_test PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -lpthread
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lgcov
    -lrt
    -lpthread
)

run_llt_test(
    TARGET hccl_mc2_test
    TASK_NUM 1
)
