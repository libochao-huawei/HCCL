cmake_minimum_required(VERSION 3.16.0)

project(hccl_custom_p2p_aiv_test LANGUAGES CXX)

add_executable(test_custom_p2p_aiv main.cc)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

target_include_directories(test_custom_p2p_aiv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_directories(test_custom_p2p_aiv PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(test_custom_p2p_aiv PRIVATE
    ascendcl
    hccl
    hccl_custom_p2p_aiv
)

target_compile_definitions(test_custom_p2p_aiv PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(test_custom_p2p_aiv PRIVATE
    -std=c++17
)
