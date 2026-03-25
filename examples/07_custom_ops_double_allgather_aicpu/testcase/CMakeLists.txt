cmake_minimum_required(VERSION 3.16.0)

project(hccl_custom_double_allgather_test LANGUAGES CXX)
find_package(MPI REQUIRED)

add_executable(test_custom_double_allgather_aicpu main.cc)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/home/kk/Ascend")
    message(WARNING "ASCEND_CANN_PACKAGE_PATH not set, defaulting to ${ASCEND_CANN_PACKAGE_PATH}")
endif()

target_include_directories(test_custom_double_allgather_aicpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${MPI_CXX_INCLUDE_PATH}
)

target_link_directories(test_custom_double_allgather_aicpu PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)

target_link_libraries(test_custom_double_allgather_aicpu PRIVATE
    ascendcl
    hccl
    hccl_custom_double_allgather
    MPI::MPI_CXX
)

target_compile_definitions(test_custom_double_allgather_aicpu PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(test_custom_double_allgather_aicpu PRIVATE -std=c++17)

