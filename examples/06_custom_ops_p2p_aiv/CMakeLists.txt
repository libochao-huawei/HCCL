cmake_minimum_required(VERSION 3.16.0)

find_package(ASC REQUIRED)

project(hccl_examples_custom_ops_p2p_aiv LANGUAGES CXX ASC)

if (NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()

if (NOT DEFINED CUSTOM_OPS_OPP_LIB_PATH)
    set(CUSTOM_OPS_OPP_LIB_PATH ${CMAKE_INSTALL_PREFIX}/lib)
endif()

if (NOT DEFINED CUSTOM_OPS_OPP_INC_PATH)
    set(CUSTOM_OPS_OPP_INC_PATH ${CMAKE_INSTALL_PREFIX}/include)
endif()

install(FILES
    inc/hccl_custom_p2p_aiv.h
    DESTINATION ${CUSTOM_OPS_OPP_INC_PATH}
)

add_subdirectory(op_kernel)
add_subdirectory(op_host)
add_subdirectory(testcase)
