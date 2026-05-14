# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# makeself.cmake - 自定义 makeself 打包脚本

message(STATUS "CPACK_CMAKE_SOURCE_DIR = ${CPACK_CMAKE_SOURCE_DIR}")
message(STATUS "CPACK_CMAKE_CURRENT_SOURCE_DIR = ${CPACK_CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CPACK_CMAKE_BINARY_DIR = ${CPACK_CMAKE_BINARY_DIR}")
message(STATUS "CPACK_CMAKE_INSTALL_PREFIX = ${CPACK_CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_COMMAND = ${CMAKE_COMMAND}")
message(STATUS "CPACK_TEMPORARY_DIRECTORY = ${CPACK_TEMPORARY_DIRECTORY}")
message(STATUS "CPACK_VERSION_SRC = ${CPACK_VERSION_SRC}")

set(CMAKE_SYSTEM_PROCESSOR ${CPACK_TARGET_ARCH})
# 设置 makeself 路径
set(MAKESELF_EXE ${CPACK_MAKESELF_PATH}/makeself.sh)
set(MAKESELF_HEADER_EXE ${CPACK_MAKESELF_PATH}/makeself-header.sh)
if(NOT MAKESELF_EXE)
    message(FATAL_ERROR "makeself not found!")
endif()

# 创建临时安装目录
set(STAGING_DIR "${CPACK_CMAKE_BINARY_DIR}/_CPack_Packages/makeself_staging")
if(NOT CPACK_CANN_NO_CLEAN)
    file(REMOVE_RECURSE "${STAGING_DIR}")
endif()
file(MAKE_DIRECTORY "${STAGING_DIR}")

# 执行安装到临时目录
if(CPACK_CANN_INSTALL_COMPONENT)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --install "${CPACK_CMAKE_BINARY_DIR}" --prefix "${STAGING_DIR}" --component "${CPACK_PACKAGE_NAME}"
        RESULT_VARIABLE INSTALL_RESULT
    )
else()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --install "${CPACK_CMAKE_BINARY_DIR}" --prefix "${STAGING_DIR}"
        RESULT_VARIABLE INSTALL_RESULT
    )
endif()

if(NOT INSTALL_RESULT EQUAL 0)
    message(FATAL_ERROR "Installation to staging directory failed: ${INSTALL_RESULT}")
endif()

if(CPACK_ENABLE_DEVICE)
    # 解压子工程包
    execute_process(
        COMMAND tar --keep-old-files -zxpf "${STAGING_DIR}/device-${CPACK_PACKAGE_NAME}.tar.gz" -C "${STAGING_DIR}"
        RESULT_VARIABLE RETCODE
    )
    if(RETCODE)
        message(FATAL_ERROR "Extract device-${CPACK_PACKAGE_NAME}.tar.gz failed, return code is ${RETCODE}.")
    endif()

    # 刪除子工程压缩包，避免打到run包中
    file(REMOVE "${STAGING_DIR}/device-${CPACK_PACKAGE_NAME}.tar.gz")
endif()

# 生成安装配置文件
execute_process(
    COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/package.py --pkg_name ${CPACK_PACKAGE_PARAM_NAME} --chip_name ${CPACK_SOC} --os_arch linux-${CMAKE_SYSTEM_PROCESSOR} --version_dir ${CPACK_VERSION} --delivery_dir ${CPACK_CMAKE_BINARY_DIR} --source_dir ${CPACK_CMAKE_SOURCE_DIR}
    WORKING_DIRECTORY ${CPACK_CMAKE_BINARY_DIR}
    OUTPUT_VARIABLE result
    ERROR_VARIABLE error
    RESULT_VARIABLE code
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT code EQUAL 0)
    message(FATAL_ERROR "Filelist generation failed: ${result} ${error}")
else ()
    message(STATUS "Filelist generated successfully: ${result}")
endif ()

# 统一修正文件权限
if(EXISTS "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/conf/path.cfg")
    execute_process(COMMAND chmod 440 "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/conf/path.cfg")
endif()
if(EXISTS "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/bin")
    execute_process(COMMAND find "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/bin" -type f -exec chmod 550 {} +)
endif()
if(EXISTS "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64")
    execute_process(COMMAND find "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib")
    execute_process(COMMAND find "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/devlib" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
    execute_process(COMMAND find "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/include" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/device/lib64")
    execute_process(COMMAND find "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/device/lib64" -type f -exec chmod 550 {} +)
endif()
if(EXISTS "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc")
    execute_process(COMMAND find "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/version")
    execute_process(COMMAND find "${STAGING_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/version" -type f -exec chmod 440 {} +)
endif()
if(EXISTS "${STAGING_DIR}/opp/built-in/op_impl")
    execute_process(COMMAND find "${STAGING_DIR}/opp/built-in/op_impl" -type f -exec chmod 440 {} +)
endif()

# makeself打包
file(STRINGS ${CPACK_CMAKE_BINARY_DIR}/makeself.txt script_output)
string(REPLACE " " ";" makeself_param_string "${script_output}")
string(REGEX MATCH "cann.*\\.run" package_name "${makeself_param_string}")

list(LENGTH makeself_param_string LIST_LENGTH)
math(EXPR INSERT_INDEX "${LIST_LENGTH} - 2")
list(INSERT makeself_param_string ${INSERT_INDEX} "${STAGING_DIR}")

message(STATUS "script output: ${script_output}")
message(STATUS "makeself: ${makeself_param_string}")
message(STATUS "package: ${package_name}")

if(NOT DEFINED CPACK_HELP_HEADER_PATH)
    set(CPACK_HELP_HEADER_PATH "share/info/${CPACK_PACKAGE_PARAM_NAME}/script/help.info")
endif()

if(NOT DEFINED CPACK_INSTALL_PATH)
    set(CPACK_INSTALL_PATH "share/info/${CPACK_PACKAGE_PARAM_NAME}/script/install.sh")
endif()


execute_process(COMMAND bash ${MAKESELF_EXE}
        --header ${MAKESELF_HEADER_EXE}
        --help-header ${CPACK_HELP_HEADER_PATH}
        ${makeself_param_string} ${CPACK_INSTALL_PATH}
        WORKING_DIRECTORY ${STAGING_DIR}
        RESULT_VARIABLE EXEC_RESULT
        ERROR_VARIABLE  EXEC_ERROR
)

if(NOT EXEC_RESULT EQUAL 0)
    message(FATAL_ERROR "makeself packaging failed: ${EXEC_ERROR}")
endif()

if(NOT DEFINED CPACK_BUILD_MODE)
    set(CPACK_BUILD_MODE "DIR_MOVE")
endif()

if(CPACK_BUILD_MODE STREQUAL "RUN_COPY")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CPACK_CMAKE_INSTALL_PREFIX}
        RESULT_VARIABLE MKDIR_INSTALL_PREFIX
    )

    if(MKDIR_INSTALL_PREFIX EQUAL 0)
        execute_process(
            COMMAND find . -name "cann-*.run"
            COMMAND xargs -I {} cp {} ${CPACK_CMAKE_INSTALL_PREFIX}/
            WORKING_DIRECTORY ${STAGING_DIR}
            RESULT_VARIABLE EXEC_RESULT
            ERROR_VARIABLE  EXEC_ERROR
        )

        if(NOT "${EXEC_RESULT}" STREQUAL "0")	 
            message(FATAL_ERROR "Failed to copy run files: ${EXEC_ERROR}")
        else()
            message(STATUS "Build pkg success: ${CPACK_CMAKE_INSTALL_PREFIX}/${package_name}")
        endif()
    else()
        message(FATAL_ERROR "Failed to mkdir new directory: ${CPACK_CMAKE_INSTALL_PREFIX}")
    endif()
endif()