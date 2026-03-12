cmake_minimum_required(VERSION 3.10)

project(checker_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ZLIB_REQ_VERSION "1.3.1")

message(STATUS "Searching for zlib...")
# find_package(ZLIB QUIET)

if(NOT ZLIB_FOUND)
    message(STATUS "Zlib not found. Fetching v${ZLIB_REQ_VERSION} from artifactory...")

    # 强制 zlib 只构建静态库，不构建共享库
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

    include(FetchContent)
    FetchContent_Declare(
        zlib
        URL https://cmc.cloudartifact.szv.dragon.tools.huawei.com/artifactory/opensource_general/zlib/1.3.1/package/zlib-1.3.1.tar.gz
        TLS_VERIFY OFF
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    # 使用 EXCLUDE_FROM_ALL 确保 zlib 的 install 命令不会污染你的 build_out 目录
    FetchContent_GetProperties(zlib)
    if(NOT zlib_POPULATED)
        FetchContent_Populate(zlib)
        add_subdirectory(${zlib_SOURCE_DIR} ${zlib_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    # 统一别名，方便后面 target_link_libraries 使用
    # Zlib 编译出的静态目标通常叫 zlibstatic
    if(TARGET zlibstatic)
        add_library(ZLIB::ZLIB ALIAS zlibstatic)
        # 补偿头文件路径，确保本地编译能搜到头文件
        target_include_directories(zlibstatic INTERFACE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})
    endif()
    
    set(ZLIB_LIBRARIES ZLIB::ZLIB)
endif()

# 创建checker.bin
add_executable(checker
    main.cc
)

add_subdirectory(src)

# 如果需要暴露头文件目录
target_include_directories(checker PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/checker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/mem_conflict_check
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/semantics_check
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/singletask_check
    ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/task_graph_generator
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dfx

    ${CMAKE_CURRENT_SOURCE_DIR}/header/external
    ${CMAKE_CURRENT_SOURCE_DIR}/header/internal

    # hcomm仓头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/../hcomm/pkg_inc
)

target_link_libraries(checker PRIVATE 
    ${ZLIB_LIBRARIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/libc_sec.a
    pthread
)

set_target_properties(checker PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(checker PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(checker PRIVATE -g)

target_compile_options(checker PRIVATE
       -std=c++14
       -fno-common
       -Werror
       -fno-strict-aliasing
       -pipe
       -O0
       -g
       -D_GLIBCXX_USE_CXX11_ABI=0
)
