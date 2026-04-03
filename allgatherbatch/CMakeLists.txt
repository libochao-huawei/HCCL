cmake_minimum_required(VERSION 3.16.0)
project(hccl_allgatherbatch_custom_ops)

set(ALLGATHERBATCH_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
)

add_library(hccl_allgatherbatch SHARED)
target_include_directories(hccl_allgatherbatch PRIVATE
    ${ALLGATHERBATCH_INCLUDE_DIRS}
)
target_compile_options(hccl_allgatherbatch PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -g -O0
    -std=c++14
    -fstack-protector-all
)
target_compile_definitions(hccl_allgatherbatch PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    -DHOST_COMPILE
)
target_link_options(hccl_allgatherbatch PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_libraries(hccl_allgatherbatch PRIVATE
    -Wl,--no-as-needed
    hcomm
    c_sec
    unified_dlog
    -Wl,--no-as-needed
)
install(TARGETS hccl_allgatherbatch
    LIBRARY DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)
install(FILES
    inc/allgather_batch.h
    DESTINATION ${CUSTOM_OPS_OPP_INC_PATH}
)

set(AICPU_KERNEL_TAR_FILE ${CMAKE_BINARY_DIR}/aicpu_allgatherbatch.tar.gz)
set(AICPU_KERNEL_JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel_aicpu/liballgatherbatch_aicpu_kernel.json)
if(KERNEL_MODE)
    add_library(allgatherbatch_aicpu_kernel SHARED)
    target_include_directories(allgatherbatch_aicpu_kernel PRIVATE
        ${ALLGATHERBATCH_INCLUDE_DIRS}
    )
    target_compile_definitions(allgatherbatch_aicpu_kernel PRIVATE
        -DAICPU_COMPILE
    )
    target_compile_options(allgatherbatch_aicpu_kernel PRIVATE
        -g
        -O0
    )
    target_link_libraries(allgatherbatch_aicpu_kernel PRIVATE
        -Wl,--no-as-needed
        ccl_kernel
        -Wl,--no-as-needed
    )
    register_custom_aicpu_target(allgatherbatch_aicpu_kernel)
    package_aicpu_kernel(
        allgatherbatch_aicpu_kernel
        ${AICPU_KERNEL_TAR_FILE}
        ${AICPU_KERNEL_JSON_FILE}
    )
else()
    install(FILES
        ${BUILD_DEVICE_DIR}/signatures/aicpu_allgatherbatch.tar.gz
        DESTINATION ${CUSTOM_OPS_OPP_AICPU_KERNEL_PATH}
    )
    install(FILES
        ${AICPU_KERNEL_JSON_FILE}
        DESTINATION ${CUSTOM_OPS_OPP_AICPU_CONFIG_PATH}
    )
endif()

add_subdirectory(op_host)
add_subdirectory(op_kernel_aicpu)

if(NOT KERNEL_MODE)
    add_subdirectory(testcase)
endif()
