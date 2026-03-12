
file(GLOB_RECURSE HCCL_SOURCES "*.cc")

# 添加共享库
add_library(ipc SHARED ${HCCL_SOURCES})
target_compile_options(ipc PRIVATE
    -fPIC
    -g
    -O0
)
# 设置头文件包含路径（如果需要）
target_include_directories(ipc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/shm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# set(DEP_INC
#     "/usr/include"
# )
# include_directories(${DEP_INC})

target_link_libraries(ipc PUBLIC
        Boost::system
        Boost::thread
        Boost::atomic
        rt
)

install(TARGETS ipc
    DESTINATION "."
    # 文件的权限 (755)
    PERMISSIONS 
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)
