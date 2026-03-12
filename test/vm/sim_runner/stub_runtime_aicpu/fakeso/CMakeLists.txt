add_library(mmpa SHARED empty.cc)
target_compile_options(mmpa PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET mmpa POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libmmpa.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(aicpu_sharder SHARED empty.cc)
target_compile_options(aicpu_sharder PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET aicpu_sharder POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libaicpu_sharder.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(ascendalog SHARED empty.cc)
target_compile_options(ascendalog PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET ascendalog POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libascendalog.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(runtime SHARED empty.cc)
target_compile_options(runtime PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET runtime POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libruntime.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(qos_manager SHARED empty.cc)
target_compile_options(qos_manager PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET qos_manager POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libqos_manager.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(exe_graph SHARED empty.cc)
target_compile_options(exe_graph PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET exe_graph POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libexe_graph.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(lowering SHARED empty.cc)
target_compile_options(lowering PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET lowering POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp liblowering.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(error_manager SHARED empty.cc)
target_compile_options(error_manager PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET error_manager POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp liberror_manager.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(tsdclient SHARED empty.cc)
target_compile_options(tsdclient PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET tsdclient POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libtsdclient.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(ra SHARED empty.cc)
target_compile_options(ra PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET ra POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libra.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)

add_library(ascendcl SHARED empty.cc)
target_compile_options(ascendcl PRIVATE
    -std=c++14
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g
    -O0
)
add_custom_command(TARGET ascendcl POST_BUILD
    COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/hccl_lib
    COMMAND cp libascendcl.so ${CMAKE_INSTALL_PREFIX}/hccl_lib
)