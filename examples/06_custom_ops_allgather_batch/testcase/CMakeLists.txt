cmake_minimum_required(VERSION 3.16.0)

project(hccl_custom_allgather_batch_test LANGUAGES CXX)

find_package(MPI REQUIRED)

add_executable(test_custom_allgather_batch main.cc)
add_executable(test_custom_allgather_batch_host_ut
    host_ut.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_host/host_utils.cc
)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(WARNING "ASCEND_CANN_PACKAGE_PATH not set, defaulting to ${ASCEND_CANN_PACKAGE_PATH}")
endif()

target_include_directories(test_custom_allgather_batch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${MPI_CXX_INCLUDE_PATH}
)

target_include_directories(test_custom_allgather_batch_host_ut PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)

target_link_directories(test_custom_allgather_batch PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

target_link_libraries(test_custom_allgather_batch PRIVATE
    ascendcl
    hccl
    hccl_custom_allgather_batch
    MPI::MPI_CXX
)

target_compile_definitions(test_custom_allgather_batch PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_definitions(test_custom_allgather_batch_host_ut PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HOST_COMPILE
)

target_compile_options(test_custom_allgather_batch PRIVATE
    -std=c++17
)

target_compile_options(test_custom_allgather_batch_host_ut PRIVATE
    -std=c++17
)
