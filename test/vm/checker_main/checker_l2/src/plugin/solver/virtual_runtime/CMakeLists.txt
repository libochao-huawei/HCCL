cmake_minimum_required(VERSION 3.14)

# # 自动扫描当前目录下的所有 .cc 文件
# file(GLOB_RECURSE SRC_SOURCES 
#     "./hccl_task_sink/*.cc"
#     "./hccl_util/*.cc"
#     "./hccl_virtual_worker/*.cc"
#     "./hccl_virtual_runtime_main.cc"
# )

# # 添加可执行文件
# add_executable(hccl_virtual_runtime ${SRC_SOURCES})
# set_target_properties(hccl_virtual_runtime PROPERTIES COMPILE_FLAGS "-g -O0")
# # 设置头文件包含路径（包含 lib 子目录）
# target_include_directories(hccl_virtual_runtime PRIVATE
#     ../hccl_common
#     ../hccl_ipc
#     ../hccl_ipc/shm
#     ../hccl_shm
#     ./hccl_util
#     ./hccl_virtual_worker
# )

# link_directories(/usr/lib/x86_64-linux-gnu/)

# # 链接依赖库
# target_link_libraries(hccl_virtual_runtime PRIVATE
#     _hccl_ipc_lib
#     hccl_shm_lib
#     Boost::system
#     Boost::thread
#     Boost::atomic
#     rt
# )

# # 设置输出目录
# set_target_properties(hccl_virtual_runtime PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output/bin
# )

# # 设置插件输出目录
#set(PLUGIN_PATH "${CMAKE_SOURCE_DIR}/build/output/bin")
#file(GLOB_RECURSE VR_LIB_SOURCES
#    "./hccl_task_sink/*.cc"
#    "./hccl_util/*.cc"
#    "./hccl_virtual_worker/*.cc"
#    hccl_virtual_runtime_main.cc
#)

# # 添加插件库
#add_executable(runner ${VR_LIB_SOURCES})
#
#set_target_properties(runner PROPERTIES COMPILE_FLAGS "-g -O0")
#
#target_include_directories(runner PRIVATE
#    ../hccl_common
#    ../hccl_ipc
#    ../hccl_ipc/shm
#    ../store/shm
#    ../hccl_shm
#    ./hccl_util
#    ./hccl_virtual_worker
#)
#
#target_link_libraries(runner PRIVATE
#    ipc
#    store
#    Boost::system
#    Boost::thread
#    Boost::atomic
#    rt
#    nlohmann_json
#)

# install(TARGETS runner
#     DESTINATION "plugin/validate/runner"
#     # 文件的权限 (755)
#     PERMISSIONS 
#         OWNER_READ OWNER_WRITE OWNER_EXECUTE 
#         GROUP_READ GROUP_EXECUTE 
#         WORLD_READ WORLD_EXECUTE
# )

install(FILES "manifest.json"
    DESTINATION "plugin/validate/runner"
    # 文件的权限 (755)
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/checker"
    DESTINATION "plugin/validate"
    # 文件的权限 (755)
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
    # 目录的权限 (755)
    DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dumper"
    DESTINATION "plugin/visualize"
    # 文件的权限 (755)
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
    # 目录的权限 (755)
    DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)