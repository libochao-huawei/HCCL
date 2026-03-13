cmake_minimum_required(VERSION 3.14)
project(hcc_vm_test)
set(CMAKE_VERBOSE_MAKEFILE true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

add_subdirectory(googletest)
# add_subdirectory(hccl_virtual_runtime_test)
# add_subdirectory(hccl_ipc)
# add_subdirectory(hccl_shm_test)
# add_subdirectory(hccl_proxy)
add_subdirectory(hccl_host)

# 并发访问测试
add_executable(concurrent_access_test concurrent_access_test.cc)
target_link_libraries(concurrent_access_test PRIVATE)
set_target_properties(concurrent_access_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# SQLite 并发测试
add_executable(sqlite_concurrent_test sqlite_concurrent_test.cc)
target_link_libraries(sqlite_concurrent_test PRIVATE sqlite3)
set_target_properties(sqlite_concurrent_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# SQLite 读写并发测试
add_executable(sqlite_read_write_test sqlite_read_write_test.cc)
target_link_libraries(sqlite_read_write_test PRIVATE sqlite3)
set_target_properties(sqlite_read_write_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# RunnerDB 多进程并发测试
add_executable(runnerdb_concurrent_test runnerdb_concurrent_test.cc)
target_include_directories(runnerdb_concurrent_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/runnerdb
)
target_link_libraries(runnerdb_concurrent_test PRIVATE
    runnerdb
    store
    log
    sqlite3
)
set_target_properties(runnerdb_concurrent_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# RunnerDB 跨架构测试
add_executable(runnerdb_cross_arch_test runnerdb_cross_arch_test.cc)
target_include_directories(runnerdb_cross_arch_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/runnerdb
)
target_link_libraries(runnerdb_cross_arch_test PRIVATE
    runnerdb
    store
    log
    sqlite3
)
set_target_properties(runnerdb_cross_arch_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

enable_testing()
# add_test(NAME hccl_ipc_test COMMAND hccl_ipc_test)
# add_test(NAME hccl_proxy_test COMMAND hccl_proxy_test)
# add_test(NAME hccl_virtual_runtime_test COMMAND hccl_virtual_runtime_test)
add_test(NAME hccl_host_test COMMAND hccl_host_test)

# 字段完整性测试
add_executable(runnerdb_field_test runnerdb_field_test.cc)
target_include_directories(runnerdb_field_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/runnerdb
)
target_link_libraries(runnerdb_field_test PRIVATE
    runnerdb
    store
    log
    sqlite3
)
set_target_properties(runnerdb_field_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# RunnerDB 全量接口测试
add_executable(runnerdb_full_test runnerdb_full_test.cc)
target_include_directories(runnerdb_full_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/runnerdb
)
target_link_libraries(runnerdb_full_test PRIVATE
    runnerdb
    store
    log
    sqlite3
)
set_target_properties(runnerdb_full_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
