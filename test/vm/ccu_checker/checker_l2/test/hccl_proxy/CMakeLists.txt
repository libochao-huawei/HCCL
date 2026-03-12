cmake_minimum_required(VERSION 3.14)

set(HCCL_VM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")
message(STATUS "HCCL_VM_DIR: ${HCCL_VM_DIR}")

add_executable(hccl_proxy_test main.cc)

target_include_directories(hccl_proxy_test PRIVATE
        ${HCCL_VM_DIR}/hccl_commom
        ../googletest/googlemock/include
        ../googletest/googletest/include
)

target_link_libraries(hccl_proxy_test PRIVATE
        gtest_main
        gmock_main
        boost_system
        boost_thread
        boost_atomic
        rt
)
set_target_properties(hccl_proxy_test PROPERTIES COMPILE_FLAGS "-g -O0")

set_target_properties(hccl_proxy_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_subdirectory(demo)