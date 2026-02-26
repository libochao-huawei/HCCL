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

set(BOOST_NAME "boost")
set(BOOST_FILE "boost_1_87_0.tar.gz")
set(BOOST_URL "https://gitcode.com/cann-src-third-party/boost/releases/download/v1.87.0/${BOOST_FILE}")
set(BOOST_PKG_PATH ${CANN_3RD_LIB_PATH}/${BOOST_FILE})
set(BOOST_SRC_PATH ${PROJECT_SOURCE_DIR}/build/third_party/boost)

if(NOT EXISTS ${BOOST_SRC_PATH}/boost/config.hpp)
    if(EXISTS ${BOOST_PKG_PATH})
        # 离线编译场景，优先使用已下载的包
        message(STATUS "[ThirdParty] Found local boost package: ${BOOST_PKG_PATH}")
        set(BOOST_PROJECT_URL ${BOOST_PKG_PATH})
    else()
        # 下载并解压
        message(STATUS "[ThirdParty] Downloading ${BOOST_NAME} from ${BOOST_URL}")
        set(BOOST_PROJECT_URL ${BOOST_URL})
    endif()

    include(ExternalProject)
    ExternalProject_Add(third_party_boost
        URL ${BOOST_PROJECT_URL}
        URL_HASH SHA256=f55c340aa49763b1925ccf02b2e83f35fdcf634c9d5164a2acb87540173c741d
        DOWNLOAD_NO_EXTRACT FALSE
        DOWNLOAD_NO_PROGRESS TRUE
        DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}
        SOURCE_DIR ${BOOST_SRC_PATH}
        CONFIGURE_COMMAND ""    # 无需编译，只需解压
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
endif()
