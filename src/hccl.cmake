# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(STATIC_MODE)
    add_library(hccl STATIC)
    set_target_properties(hccl PROPERTIES
        OUTPUT_NAME "hccl_static"
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(hccl SHARED)
endif()

# 基于ini生成json文件
SET(HCCL_CMAKE_DIR ${OPS_BASE_DIR}/cmake/)
message(STATUS "HCCL_CMAKE_DIR = ${HCCL_CMAKE_DIR}")
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
    COMMAND ${HI_PYTHON}
            ${HCCL_CMAKE_DIR}/scripts/parser_ini.py
            ${CMAKE_CURRENT_SOURCE_DIR}/ops/scatter/scatter_aicpu_kernel.ini
            ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(aicpu_kernel_json DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libscatter_aicpu_kernel.json
    DESTINATION ${INSTALL_AICPU_KERNEL_JSON_DIR}/config
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
add_dependencies(hccl aicpu_kernel_json)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccl PRIVATE
        OPEN_BUILD_PROJECT
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
else()
    target_compile_definitions(hccl PRIVATE
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
    )
endif()

target_include_directories(hccl PRIVATE
    ${INCLUDE_LIST}
)

target_compile_definitions(hccl PRIVATE
    -DHOST_COMPILE
)

hccl_apply_cann_compat(hccl)

if(HCCL_CANN_COMPAT_850)
    target_compile_definitions(hccl PRIVATE HCCL_CANN_COMPAT_850)
endif()

target_compile_options(hccl PRIVATE
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
target_link_directories(hccl PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

if(NOT STATIC_MODE)
    add_dependencies(hccl hccl_compat)
endif()

if(BUILD_OPEN_PROJECT)
    target_link_libraries(hccl PRIVATE
        -Wl,--no-as-needed
        hcomm
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
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
        hccl_compat
        acl_rt
        c_sec
        unified_dlog
        -Wl,--no-as-needed
        ofed_headers
    )
endif()

if(NOT STATIC_MODE)
    target_link_options(hccl PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )
endif()

target_link_directories(hccl PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

if(STATIC_MODE)
    target_link_libraries(hccl PRIVATE
        hcomm
        acl_rt
        c_sec
        unified_dlog
    )
else()
    if(BUILD_OPEN_PROJECT)
        target_link_libraries(hccl PRIVATE
            -Wl,--no-as-needed
            hcomm
            acl_rt
            c_sec
            unified_dlog
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
            -Wl,--no-as-needed
            ofed_headers
        )
    endif()
endif()

if(STATIC_MODE)
    # 后处理阶段再安装
else()
    install(TARGETS hccl
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ${INSTALL_OPTIONAL}
        COMPONENT hccl
    )
endif()