cmake_minimum_required(VERSION 3.16.0)
project(allgatherbatch_testcase)

find_package(Threads REQUIRED)

add_executable(allgatherbatch_testcase
    main.cc
)
target_include_directories(allgatherbatch_testcase PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
)
target_link_directories(allgatherbatch_testcase PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/device/lib64
    ${CUSTOM_OPS_OPP_LIB_PATH}
)
target_compile_options(allgatherbatch_testcase PRIVATE
    -std=c++14
)
target_compile_definitions(allgatherbatch_testcase PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)
target_link_libraries(allgatherbatch_testcase PRIVATE
    hccl_allgatherbatch
    hccl
    hcomm
    ascendcl
    Threads::Threads
)