cmake_minimum_required(VERSION 3.16.0)

project(hccl_custom_allgather_test LANGUAGES CXX)

add_executable(test_custom_allgather main.cc)

target_include_directories(test_custom_allgather PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_directories(test_custom_allgather PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(test_custom_allgather PRIVATE
    ascendcl
    hccl
    hccl_custom_allgather
)

target_compile_definitions(test_custom_allgather PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(test_custom_allgather PRIVATE
    -std=c++17
)
