# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

unset(mockcpp_FOUND CACHE)
unset(MOCKCPP_INCLUDE CACHE)
unset(MOCKCPP_STATIC_LIBRARY CACHE)

set(MOCKCPP_FILE "mockcpp-2.7.tar.gz")
set(MOCKCPP_URL "https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h4/${MOCKCPP_FILE}")
set(MOCKCPP_PKG_PATH ${CANN_3RD_LIB_PATH}/${MOCKCPP_FILE})
set(MOCKCPP_INSTALL_PATH ${CANN_3RD_LIB_PATH}/mockcpp)

set(MOCKCPP_PATCH_FILE "mockcpp-2.7_py3.patch")
set(MOCKCPP_PATCH_URL "https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h4/${MOCKCPP_PATCH_FILE}")
set(MOCKCPP_PATCH_PATH ${CANN_3RD_LIB_PATH}/${MOCKCPP_PATCH_FILE})

# 查找目录下是否已经安装，避免重复编译安装
message(STATUS "[ThirdParty] MOCKCPP_INSTALL_PATH=${MOCKCPP_INSTALL_PATH}")
find_path(MOCKCPP_INCLUDE
    NAMES mockcpp/mockcpp.hpp
    PATH_SUFFIXES include
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${MOCKCPP_INSTALL_PATH}
)
find_library(MOCKCPP_STATIC_LIBRARY
    NAMES libmockcpp.a
    PATH_SUFFIXES lib
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH
    PATHS ${MOCKCPP_INSTALL_PATH}
)

# 是否全部找到 mockcpp 的头文件和二进制
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mockcpp
    FOUND_VAR
    mockcpp_FOUND
    REQUIRED_VARS
    MOCKCPP_INCLUDE
    MOCKCPP_STATIC_LIBRARY
)
message(STATUS "[ThirdParty] Found MockCpp: ${mockcpp_FOUND}")

if(mockcpp_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[ThirdParty] MockCpp found in ${MOCKCPP_INSTALL_PATH}, and not force rebuild cann third_party")
else()
    # 编译 mockcpp 需要 boost 库
    include(${CMAKE_CURRENT_LIST_DIR}/boost.cmake)

    # mockcpp 补丁
    if(EXISTS ${MOCKCPP_PATCH_PATH})
        message(STATUS "[ThirdParty] Found local mockcpp patch package: ${MOCKCPP_PATCH_PATH}")
        set(MOCKCPP_PATCH_PROJECT_URL ${MOCKCPP_PATCH_PATH})
    else()
        message(STATUS "[ThirdParty] Downloading mockcpp patch from ${MOCKCPP_PATCH_URL}")
        set(MOCKCPP_PATCH_PROJECT_URL ${MOCKCPP_PATCH_URL})
    endif()

    if(EXISTS ${MOCKCPP_PKG_PATH})
        # 离线编译场景，优先使用已下载的包
        message(STATUS "[ThirdParty] Found local mockcpp package: ${MOCKCPP_PKG_PATH}")
        set(MOCKCPP_PROJECT_URL ${MOCKCPP_PKG_PATH})
    else()
        # 下载并编译安装
        message(STATUS "[ThirdParty] Downloading mockcpp from ${MOCKCPP_URL}")
        set(MOCKCPP_PROJECT_URL ${MOCKCPP_URL})
    endif()

    # 编译选项设置
    set(MOCKCPP_CFLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
    set(MOCKCPP_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

    set(MOCKCPP_OPTS
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_CXX_FLAGS=${MOCKCPP_CXXFLAGS}
        -DCMAKE_C_FLAGS=${MOCKCPP_CFLAGS}
        -DCMAKE_INSTALL_PREFIX=${MOCKCPP_INSTALL_PATH}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DBOOST_INCLUDE_DIRS=${BOOST_SRC_PATH}
        -DCMAKE_SHARED_LINKER_FLAGS=""
        -DCMAKE_EXE_LINKER_FLAGS=""
        -DBUILD_32_BIT_TARGET_BY_64_BIT_COMPILER=OFF
        -DBUILD_TESTING=OFF
    )

    include(ExternalProject)
    ExternalProject_Add(third_party_mockcpp_patch
        URL ${MOCKCPP_PATCH_PROJECT_URL}
        URL_HASH SHA256=600c0a263182b1f988e77bb907666d24a72d6ea624a52212d61750384745327d
        TLS_VERIFY OFF
        DOWNLOAD_NO_EXTRACT TRUE
        DOWNLOAD_NO_PROGRESS TRUE
        DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}
        PATCH_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    ExternalProject_Add(third_party_mockcpp
        URL ${MOCKCPP_PROJECT_URL}
        URL_HASH SHA256=73ab0a8b6d1052361c2cebd85e022c0396f928d2e077bf132790ae3be766f603
        TLS_VERIFY OFF
        DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}
        DOWNLOAD_NO_PROGRESS TRUE
        PATCH_COMMAND patch -p1 < ${MOCKCPP_PATCH_PATH}     # 应用 patch
        CONFIGURE_COMMAND ${CMAKE_COMMAND} ${MOCKCPP_OPTS} <SOURCE_DIR>
        BUILD_COMMAND $(MAKE)
        INSTALL_COMMAND $(MAKE) install
        DEPENDS third_party_boost third_party_mockcpp_patch
    )
endif()

# 创建导入的目标
add_library(mockcpp STATIC IMPORTED)
add_dependencies(mockcpp third_party_mockcpp)

if(NOT EXISTS ${MOCKCPP_INSTALL_PATH}/include)
    file(MAKE_DIRECTORY "${MOCKCPP_INSTALL_PATH}/include")
endif()

set_target_properties(mockcpp PROPERTIES
    IMPORTED_LOCATION ${MOCKCPP_INSTALL_PATH}/lib/libmockcpp.a
    INTERFACE_INCLUDE_DIRECTORIES ${MOCKCPP_INSTALL_PATH}/include
)
