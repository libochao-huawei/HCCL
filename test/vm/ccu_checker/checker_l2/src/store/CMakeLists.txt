file(GLOB_RECURSE HCCL_SOURCES "*.cc")

# 添加共享库
add_library(store SHARED ${HCCL_SOURCES})
target_compile_options(store PRIVATE
    -fPIC
    -g
    -O0
)

# 设置头文件包含路径（改为绝对路径，基于项目根目录）
target_include_directories(store PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# 链接依赖库（补充 boost_interprocess，之前遗漏了！）
target_link_libraries(store PRIVATE
    boost_system
    boost_thread 
    boost_atomic
    rt

    log
)

install(TARGETS store
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/hccl_vm_install
)
