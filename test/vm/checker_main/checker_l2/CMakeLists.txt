cmake_minimum_required(VERSION 3.10)
project(hccl_vm VERSION 1.0.0)
set(CMAKE_VERBOSE_MAKEFILE true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})

set(HCOMM_VM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    message(FATAL_ERROR "Error: ASCEND_TOOLKIT_HOME environment variable is not set. Please set it to the Ascend Toolkit installation path.")
    endif()
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_TOOLKIT_HOME})
endif()
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

set(TOP_DIR ${CMAKE_SOURCE_DIR}/../../..)
message(STATUS "TOP_DIR: ${TOP_DIR}")

set(HCOMM_CODE_ROOT      ${CMAKE_SOURCE_DIR}/../..)
set(HCOMM_CODE_COMMON    ${HCOMM_CODE_ROOT}/src/common)
set(HCOMM_CODE_PLATFORM  ${HCOMM_CODE_ROOT}/src/platform)
set(HCOMM_CODE_ALGORITHM ${HCOMM_CODE_ROOT}/src/algorithm)
set(HCOMM_CODE_FRAMEWORK ${HCOMM_CODE_ROOT}/src/framework)
message(STATUS "HCOMM_CODE_ROOT: ${HCOMM_CODE_ROOT}")

set(THRID_PARTY_DIR ${HCOMM_CODE_ROOT}/output/third_party)

# 使用一个自定义的 Cache 变量，默认值设为 OFF 或空
set(HCCL_VM_PROJECT_PREFIX_SET OFF CACHE INTERNAL "Check if install prefix is set")

if(NOT HCCL_VM_PROJECT_PREFIX_SET)
    # 只要这个逻辑没跑成功过，就设置一次
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/hccl_vm_install" CACHE PATH "Default install prefix" FORCE)
    set(HCCL_VM_PROJECT_PREFIX_SET ON CACHE INTERNAL "Check if install prefix is set")
    message(STATUS "First-time setup: Installation prefix locked to: ${CMAKE_INSTALL_PREFIX}")
endif()

set(CMAKE_INSTALL_RPATH "\$ORIGIN")

install(DIRECTORY DESTINATION "data")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/asset/"
    DESTINATION "."
    # 文件的权限 (755)
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
    # 目录的权限 (755)
    DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)

add_subdirectory(third_party)
add_subdirectory(src)

set(CPACK_PACKAGE_NAME "hccl_vm")
set(CPACK_GENERATOR "TGZ")
include(CPack)
#add_subdirectory(test)
