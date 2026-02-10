add_subdirectory(log)
add_subdirectory(ipc)
add_subdirectory(proxy)
add_subdirectory(runnerdb)
add_subdirectory(store)
add_subdirectory(plugin/solver/virtual_runtime)

add_executable(hccl-vm 
    main.cc
    hccl_vm_subcmd.cc
    hccl_data_dump.cc
    gen_device_topofile.cc
    hccl_plugin_manager.cc
    hccl_plugin.cc
    binary_data_operator.cc
    cmd_table_handle.cc
)

message(STATUS "ASCEND_CANN_PACKAGE_PATH12 = ${ASCEND_CANN_PACKAGE_PATH}")
target_include_directories(hccl-vm PUBLIC
    # hccl vm头文件
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/src/store/hccl_shm
    ${HCOMM_VM_ROOT_PATH}/src/dep
    ${HCOMM_VM_ROOT_PATH}/src/host
    ${HCOMM_VM_ROOT_PATH}/src/host/hccl_cmd
    ${HCOMM_VM_ROOT_PATH}/src/host/hccl_plugin
    ${HCOMM_VM_ROOT_PATH}/src/runnerdb

    # hcomm头文件
    ${TOP_DIR}/ace/comop/hccl/orion/common
    ${TOP_DIR}/ace/comop/hccl/orion/common/utils
    ${TOP_DIR}/ace/comop/hccl/orion/common/types
    ${TOP_DIR}/ace/comop/hccl/orion/common/exception


    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/common
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/pub_inc
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/buffer
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/resource/buffer/aicpu
    ${TOP_DIR}/ace/comop/hccl/orion/unified_platform/external_system

    # cann包头文件
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/toolchain

    # cann包头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts
        

        # hccl_vm头文件
        ${HCOMM_VM_ROOT_PATH}/include
        ${HCOMM_VM_ROOT_PATH}/include/runnerdb
        ${HCOMM_VM_ROOT_PATH}/include/store
        ${HCOMM_VM_ROOT_PATH}/src/ipc
        ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_cmd
        ${HCOMM_VM_ROOT_PATH}/src/proxy/level2/src

        
        # HCCP头文件
        ${TOP_DIR}/drivers/network/inc/network
        ${TOP_DIR}/drivers/network/orion/hcomm_dev/inc/network

        # orion仓头文件

        # hccl_proxy模块内部头文件
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/topomodel
        ${CMAKE_CURRENT_SOURCE_DIR}/proxy
)

target_link_libraries(hccl-vm
    PRIVATE
    ipc
    runnerdb
    store
    log
)

target_compile_options(hccl-vm PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_include_directories(hccl-vm PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
)

set(YAMLCPP_REQ_VERSION "0.8.0")
set(ZLIB_REQ_VERSION "1.3.1")
set(JSON_REQ_VERSION "3.11.3")

# 链接第三方库
include(FetchContent)

# yaml-cpp 0.8.0
message(STATUS "Checking for yaml-cpp...")
find_package(yaml-cpp ${YAMLCPP_REQ_VERSION} QUIET)

if(yaml-cpp_FOUND)
    message(STATUS "Found System yaml-cpp: ${yaml-cpp_VERSION}")
else()
    message(STATUS "System yaml-cpp not found. Fetching v${YAMLCPP_REQ_VERSION} online...")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    
    # 禁止编译 yaml-cpp 的测试和工具，加快构建速度
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        yaml-cpp
        URL https://cmc.cloudartifact.szv.dragon.tools.huawei.com/artifactory/opensource_general/yaml-cpp/0.8.0/package/yaml-cpp-0.8.0.tar.gz
        TLS_VERIFY OFF
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(yaml-cpp)
    message(STATUS "yaml-cpp fetched and configured.")
endif()

# zlib 1.3.1
message(STATUS "Searching for zlib...")
find_package(ZLIB ${ZLIB_REQ_VERSION} QUIET)
if(ZLIB_FOUND)
    message(STATUS "Found System zlib: ${ZLIB_VERSION}")
else()
    # 此时 ZLIB_VERSION 为空，我们手动打印将要下载的版本
    message(STATUS "System zlib not found. Fetching v${ZLIB_REQ_VERSION} online...")

    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    # set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    include(FetchContent)
    FetchContent_Declare(
        zlib
        URL https://cmc.cloudartifact.szv.dragon.tools.huawei.com/artifactory/opensource_general/zlib/1.3.1/package/zlib-1.3.1.tar.gz
        TLS_VERIFY OFF
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    FetchContent_GetProperties(zlib)
    if(NOT zlib_POPULATED)
        FetchContent_Populate(zlib)
        # 使用 EXCLUDE_FROM_ALL 阻止其 install 指令生效
        add_subdirectory(${zlib_SOURCE_DIR} ${zlib_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    if(TARGET zlibstatic AND NOT TARGET ZLIB::ZLIB)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
    elseif(TARGET zlib AND NOT TARGET ZLIB::ZLIB)
        add_library(ZLIB::ZLIB ALIAS zlib)
    endif()
    
    # 手动给变量赋值，防止后续逻辑用到它时为空
    set(ZLIB_VERSION ${ZLIB_REQ_VERSION})
    message(STATUS "zlib v${ZLIB_VERSION} fetched and configured.")
endif()

# Boost 1.87.0
include(ExternalProject)
set(BOOST_VERSION "1.87.0")
set(BOOST_URL "https://cmc.cloudartifact.szv.dragon.tools.huawei.com/artifactory/opensource_general/Boost/1.87.0/package/boost_1_87_0.tar.gz")

set(BOOST_PREFIX      "${CMAKE_BINARY_DIR}/external/boost")
set(BOOST_SOURCE_DIR  "${BOOST_PREFIX}/src/boost_external")
set(BOOST_INSTALL_DIR "${BOOST_PREFIX}/install")
set(BOOST_INCLUDE_DIR "${BOOST_INSTALL_DIR}/include")
set(BOOST_LIB_DIR     "${BOOST_INSTALL_DIR}/lib")

# --- 2. 关键：预先创建目录以绕过 CMake 配置阶段的路径检查 ---
if(NOT EXISTS "${BOOST_INCLUDE_DIR}")
    file(MAKE_DIRECTORY "${BOOST_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${BOOST_LIB_DIR}")
    file(MAKE_DIRECTORY "${BOOST_LIB_DIR}")
endif()

# --- 3. 获取 CPU 核心数加速编译 ---

# --- 4. 定义外部项目构建流程 ---
ExternalProject_Add(boost_external
    URL               ${BOOST_URL}
    PREFIX            ${BOOST_PREFIX}
    SOURCE_DIR        ${BOOST_SOURCE_DIR}
    TLS_VERIFY        OFF
    DOWNLOAD_EXTRACT_TIMESTAMP  TRUE
    # 下载完成后进入源码目录执行 bootstrap
    BUILD_IN_SOURCE   1
    CONFIGURE_COMMAND <SOURCE_DIR>/bootstrap.sh --prefix=${BOOST_INSTALL_DIR}
    # 编译并安装指定组件到 INSTALL_DIR
    BUILD_COMMAND     <SOURCE_DIR>/b2 install 
                      --with-system 
                      --with-thread 
                      --with-atomic 
                      link=static 
                      variant=release 
                      threading=multi 
                      -j16
    INSTALL_COMMAND   ""
    # 定义产物，告诉 CMake 什么时候认为这个任务完成了
    BUILD_BYPRODUCTS  "${BOOST_LIB_DIR}/libboost_system.a"
                      "${BOOST_LIB_DIR}/libboost_thread.a"
                      "${BOOST_LIB_DIR}/libboost_atomic.a"
)

# --- 5. 创建导入目标 (Imported Targets) ---
# 定义一个辅助宏来简化 Target 的创建
macro(import_boost_lib name)
    add_library(Boost::${name} STATIC IMPORTED GLOBAL)
    set_target_properties(Boost::${name} PROPERTIES
        IMPORTED_LOCATION             "${BOOST_LIB_DIR}/libboost_${name}.a"
        INTERFACE_INCLUDE_DIRECTORIES "${BOOST_INCLUDE_DIR}"
    )
    # 建立依赖关系：确保先运行编译任务，再链接库
    add_dependencies(Boost::${name} boost_external)
endmacro()

import_boost_lib(system)
import_boost_lib(thread)
import_boost_lib(atomic)

target_link_libraries(hccl-vm PRIVATE CLI11::CLI11)
target_link_libraries(hccl-vm PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(hccl-vm PRIVATE ZLIB::ZLIB)
target_link_libraries(hccl-vm PRIVATE JSON::JSON)

install(TARGETS hccl-vm
    DESTINATION "."
    # 文件的权限 (755)
    PERMISSIONS 
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)