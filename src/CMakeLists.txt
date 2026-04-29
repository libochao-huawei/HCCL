# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(hccl)

if(BUILD_OPEN_PROJECT)
    message(STATUS "open TOP_DIR = ${TOP_DIR}")
    set(OPS_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
else()
    message(STATUS "close TOP_DIR = ${TOP_DIR}")
    set(OPS_BASE_DIR ${TOP_DIR}/hccl/)
    set(INSTALL_AICPU_KERNEL_JSON_DIR ${INSTALL_LIBRARY_DIR})
endif()

set(INCLUDE_LIST
    ${OPS_BASE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hcomm_dlsym

    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/inc

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/channel
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/executor/registry

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/selector

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/wrapper
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/aiv
    # ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/hostdpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_common/topo
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/interface_graph_mode

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/broadcast/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce_scatter_v/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/algo/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_gather_v/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_to_all_v/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/all_reduce/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/send/template

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/recv/template

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/selector
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aicpu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/aiv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/reduce/template/ccu/kernel

    ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv/executor
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/batch_send_recv/selector
)

if(BUILD_OPEN_PROJECT)
    list(APPEND INCLUDE_LIST
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/acl

        # hcomm头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm

        # runtime头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/
        ${ASCEND_CANN_PACKAGE_PATH}/include/base/
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
        ${ASCEND_CANN_PACKAGE_PATH}/include/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/include/external/
        ${ASCEND_CANN_PACKAGE_PATH}/include/platform/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # mmpa头文件
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/mmpa/

        # driver头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/

        # 包间接口
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/dump/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace/
    )
else()
    list(APPEND INCLUDE_LIST
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/aicpu/tsd
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/inc/external
        ${TOP_DIR}/inc/external/acl
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/inc/aicpu/aicpu_schedule/aicpu_sharder
        ${TOP_DIR}/abl/qos/qosmng
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/abl/libc_sec/include
        ${TOP_DIR}/inc/aicpu/common
        ${TOP_DIR}/ace/csruntime/inc/tsch
        ${TOP_DIR}/asl/hss/inc
        ${TOP_DIR}/asl/hss/inc/pub
        ${TOP_DIR}/asl/hss/common/hash_table
        ${TOP_DIR}/metadef
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/metadef/inc
        ${TOP_DIR}/metadef/inc/common
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/include/dfx
        ${TOP_DIR}/runtime/pkg_inc
        ${TOP_DIR}/runtime/pkg_inc/base
        ${TOP_DIR}/runtime/pkg_inc/mmpa

        ${TOP_DIR}/open_source/json/include

        ${TOP_DIR}/hcomm/inc
        ${TOP_DIR}/hcomm/inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc
        ${TOP_DIR}/hcomm/pkg_inc/hccl
        ${TOP_DIR}/hcomm/pkg_inc/hcomm/ccu
        ${TOP_DIR}/hcomm/include
        ${TOP_DIR}/hcomm/include/hccl
    )
endif()

if((DEVICE_MODE AND BUILD_OPEN_PROJECT) OR (NOT BUILD_OPEN_PROJECT))
    #添加scatter_aipu_kernel库 在device侧使用
    include(scatter_aicpu_kernel.cmake)
endif()

add_definitions(-DORION_MODE)

if(STATIC_MODE)
    add_library(hccl STATIC)
    set_target_properties(hccl PROPERTIES
        OUTPUT_NAME "hccl_static"
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(hccl SHARED)
endif()

if(NOT KERNEL_MODE)
    # 基于ini生成json文件
    SET(HCCL_CMAKE_DIR ${OPS_BASE_DIR}/cmake/)
    message(STATUS "HCCL_CMAKE_DIR = ${HCCL_CMAKE_DIR}")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
        COMMAND ${HI_PYTHON} ${HCCL_CMAKE_DIR}/scripts/parser_ini.py ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/scatter_aicpu_kernel.ini ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_custom_target(aicpu_kernel_json DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
        DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
    add_dependencies(hccl aicpu_kernel_json)
endif()

set(TARGET_NAME hccl)

set(HCCT_TARGET ${TARGET_NAME})

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        OPEN_BUILD_PROJECT
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
else()
    target_compile_definitions(${TARGET_NAME} PRIVATE
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
endif()

target_include_directories(${TARGET_NAME} PRIVATE
    ${INCLUDE_LIST}
)

target_compile_definitions(${TARGET_NAME} PRIVATE
    -DHOST_COMPILE
)

target_compile_options(${TARGET_NAME} PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-O3 -g>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
    -fstack-protector-all
)

# libhccl
target_link_directories(${TARGET_NAME} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

if(NOT STATIC_MODE)
    add_dependencies(${TARGET_NAME} hccl_compat)
endif()

if(BUILD_OPEN_PROJECT)
    target_link_libraries(${TARGET_NAME} PRIVATE
        -Wl,--no-as-needed
        hcomm
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
        mmpa
        -Wl,--no-as-needed
    )
else()
    target_link_libraries(${TARGET_NAME} PRIVATE
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:npu_runtime_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        -Wl,--no-as-needed
        hcomm
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
        mmpa
        -Wl,--no-as-needed
        ofed_headers
    )
endif()

if(NOT STATIC_MODE)
    target_link_options(${TARGET_NAME} PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )
endif()

target_link_directories(${TARGET_NAME} PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

if(STATIC_MODE)
    target_link_libraries(hccl PRIVATE
        hcomm
        acl_rt
        c_sec
        unified_dlog
        mmpa
    )
else()
    if(BUILD_OPEN_PROJECT)
        target_link_libraries(hccl PRIVATE
            -Wl,--no-as-needed
            hcomm
            acl_rt
            c_sec
            unified_dlog
            mmpa
            -Wl,--no-as-needed
        )
    else()
        target_link_libraries(hccl PRIVATE
            $<BUILD_INTERFACE:slog_headers>
            $<BUILD_INTERFACE:msprof_headers>
            $<BUILD_INTERFACE:npu_runtime_headers>
            $<BUILD_INTERFACE:mmpa_headers>
            -Wl,--no-as-needed
            hcomm
            acl_rt
            c_sec
            unified_dlog
            mmpa
            -Wl,--no-as-needed
            ofed_headers
        )
    endif()
endif()

if(STATIC_MODE)
    install(TARGETS hccl
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
else()
    install(TARGETS hccl
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
endif()

if((DEVICE_MODE AND BUILD_OPEN_PROJECT) OR (NOT BUILD_OPEN_PROJECT))
    if(STATIC_MODE)
        install(TARGETS scatter_aicpu_kernel
            LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
            ${INSTALL_OPTIONAL}
            COMPONENT hccl
        )
    else()
        install(TARGETS scatter_aicpu_kernel
            LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
            ${INSTALL_OPTIONAL}
            COMPONENT hccl
        )
    endif()
endif()

add_subdirectory(common)
add_subdirectory(ops)

# 定义AIV算子聚合目标
set(AIV_TARGETS
    hccl_aiv_all_gather_op_910_95
    hccl_aiv_all_reduce_op_910_95
    hccl_aiv_all_to_all_v_op_910_95
    hccl_aiv_all_to_all_op_910_95
    hccl_aiv_broadcast_op_910_95
    hccl_aiv_reduce_op_910_95
    hccl_aiv_reduce_scatter_op_910_95
    hccl_aiv_scatter_op_910_95
    hccl_aiv_send_op_910_95
    hccl_aiv_recv_op_910_95
)

add_custom_target(aiv_all_targets)
foreach(target ${AIV_TARGETS})
    if(TARGET ${target})
        add_dependencies(aiv_all_targets ${target})
    endif()
endforeach()

if(DEVICE_MODE AND KERNEL_MODE)
    set(CCL_KERNEL_TAR_DIR ${OPS_BASE_DIR}/build_device/ccl_kernel_tar_pkg/aicpu_kernels_device)
    add_custom_command(
        TARGET scatter_aicpu_kernel
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CCL_KERNEL_TAR_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:scatter_aicpu_kernel> ${CCL_KERNEL_TAR_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hccl_kernel_compat> ${CCL_KERNEL_TAR_DIR}
        COMMENT "Copying libscatter_aicpu_kernel.so&libhccl_kernel_compat.so to ${CCL_KERNEL_TAR_DIR}"
    )

    set(AICPU_CUSTOM_DIR ${OPS_BASE_DIR}/build_device/aicpu_custom/)
    add_custom_command(
        TARGET scatter_aicpu_kernel
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${AICPU_CUSTOM_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:scatter_aicpu_kernel> ${AICPU_CUSTOM_DIR}/libscatter_aicpu_kernel.so
        COMMENT "Copying libscatter_aicpu_kernel.so to ${AICPU_CUSTOM_DIR}/libscatter_aicpu_kernel.so"
    )
endif()

# 静态库特殊处理：嵌入二进制文件
if(STATIC_MODE)
    # 添加静态库恢复代码
    set(STATIC_RESTORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/common/static_restore.c)
    if(NOT EXISTS ${STATIC_RESTORE_SRC})
        file(WRITE ${STATIC_RESTORE_SRC} "/* Static restore implementation will be generated during build */\n")
    endif()

    # 创建自定义目标来处理静态库的最终打包
    add_custom_target(package_static_lib DEPENDS hccl)

    # 添加后构建步骤来嵌入二进制文件
    add_custom_command(
        TARGET package_static_lib
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Processing static library embedding..."
    )
endif()
