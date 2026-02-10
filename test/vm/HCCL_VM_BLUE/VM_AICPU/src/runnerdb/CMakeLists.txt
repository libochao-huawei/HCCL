add_library(runnerdb SHARED "")

target_sources(runnerdb PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_runner_db.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sim_runner_common.cc
)

target_include_directories(runnerdb PRIVATE
        # hccl_proxy依赖的HCCL业务代码头文件, 这里的头文件有要求: 开源用户可获取
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
)
target_include_directories(runnerdb PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/runnerdb
)

target_include_directories(runnerdb PRIVATE
        # hccl_proxy依赖的Checker L2其他模块的头文件
        ${HCOMM_VM_ROOT_PATH}/include
        ${HCOMM_VM_ROOT_PATH}/src/store/hccl_shm
        ${HCOMM_VM_ROOT_PATH}/src/runnerdb
)

target_compile_options(runnerdb PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O0
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(runnerdb PRIVATE
        store
        log
)

install(TARGETS runnerdb
    DESTINATION "."
    # 文件的权限 (755)
    PERMISSIONS 
        OWNER_READ OWNER_WRITE OWNER_EXECUTE 
        GROUP_READ GROUP_EXECUTE 
        WORLD_READ WORLD_EXECUTE
)
