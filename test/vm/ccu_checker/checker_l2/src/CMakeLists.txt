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
)

message(STATUS "ASCEND_CANN_PACKAGE_PATH12 = ${ASCEND_CANN_PACKAGE_PATH}")
target_include_directories(hccl-vm PUBLIC
    # hccl vm头文件
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/include/runnerdb
    ${HCOMM_VM_ROOT_PATH}/src/store/hccl_shm
    ${HCOMM_VM_ROOT_PATH}/src/host
    ${HCOMM_VM_ROOT_PATH}/src/host/hccl_cmd
    ${HCOMM_VM_ROOT_PATH}/src/host/hccl_plugin
    ${HCOMM_VM_ROOT_PATH}/src/runnerdb

    ## cann包头文件
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

target_link_libraries(hccl-vm PRIVATE CLI11::CLI11)
target_link_libraries(hccl-vm PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(hccl-vm PRIVATE ZLIB::ZLIB)
target_link_libraries(hccl-vm PRIVATE JSON::JSON)

install(TARGETS hccl-vm
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/hccl_vm_install
)
