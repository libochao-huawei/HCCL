# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

unset(json_FOUND CACHE)
unset(JSON_INCLUDE CACHE)

set(JSON_FILE "include.zip")
set(JSON_URL "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/${JSON_FILE}")
set(JSON_PKG_PATH ${CANN_3RD_LIB_PATH}/${JSON_FILE})
set(JSON_SRC_PATH ${CANN_3RD_LIB_PATH}/json)

find_path(JSON_INCLUDE
    NAMES nlohmann/json.hpp
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${JSON_SRC_PATH}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(json
    FOUND_VAR
    json_FOUND
    REQUIRED_VARS
    JSON_INCLUDE
)

if(json_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[ThirdParty] nlohmann_json found in ${JSON_SRC_PATH}, and not force rebuild cann third_party")
else()
    if(EXISTS ${JSON_PKG_PATH})
        # 离线编译场景，优先使用已下载的包
        message(STATUS "[ThirdParty] Found local nlohmann_json package: ${JSON_PKG_PATH}")
        set(JSON_PROJECT_URL ${JSON_PKG_PATH})
    else()
        # 下载并编译安装
        message(STATUS "[ThirdParty] Downloading nlohmann_json from ${JSON_URL}")
        set(JSON_PROJECT_URL ${JSON_URL})
    endif()

    include(ExternalProject)
    ExternalProject_Add(third_party_json
        URL ${JSON_PROJECT_URL}
        URL_HASH SHA256=a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d
        TLS_VERIFY OFF
        DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}
        DOWNLOAD_NO_PROGRESS TRUE
        SOURCE_DIR ${JSON_SRC_PATH}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
    )
endif()

# 创建导入的目标
add_library(json INTERFACE)
set(THIRD_PARTY_NLOHMANN_PATH ${JSON_SRC_PATH}/include)
target_include_directories(json INTERFACE
    ${THIRD_PARTY_NLOHMANN_PATH}
)
add_dependencies(json third_party_json)
