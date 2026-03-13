file(GLOB_RECURSE HCCL_CMD_SOURCES
        "base/*.cpp"
        "subcmds/*.cpp"
        "utils/*.cpp"
)

add_library(cmd OBJECT ${HCCL_CMD_SOURCES})

target_include_directories(cmd PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/base
        ${CMAKE_CURRENT_SOURCE_DIR}/subcmds
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../store/hccl_shm
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/runnerdb
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party
)

target_compile_options(cmd PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(cmd PRIVATE
        dl
        rt
        log
        runnerdb
        store
        ipc
)

target_link_libraries(cmd PRIVATE CLI11::CLI11)
target_link_libraries(cmd PRIVATE yaml-cpp::yaml-cpp)
target_link_libraries(cmd PRIVATE JSON::JSON)
