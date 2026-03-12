add_library(log STATIC hccl_vm_log.cc)

target_include_directories(log PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}

        ${CMAKE_SOURCE_DIR}/third_party
)

target_link_libraries(log PRIVATE ZLIB::ZLIB)

target_compile_options(log PRIVATE
        -std=c++14
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)
