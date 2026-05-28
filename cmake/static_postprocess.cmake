# 静态库后处理：嵌入AICPU与AIV内核
set(_STATIC_POSTPROCESS_DIR "${CMAKE_CURRENT_LIST_DIR}")

# 中间静态库（纯host）
set(HCCL_STATIC_INTERMEDIATE "${CMAKE_BINARY_DIR}/src/libhccl_static.a")

# device侧AICPU包
set(HCCL_AICPU_TAR "${CMAKE_BINARY_DIR}/hccl_device/signatures/aicpu_hccl.tar.gz")

# 最终静态库（含内核）
set(HCCL_STATIC_FINAL_LIB "${CMAKE_BINARY_DIR}/libhccl_static_final.a")

add_custom_command(
    OUTPUT ${HCCL_STATIC_FINAL_LIB}
    COMMAND ${CMAKE_COMMAND}
        -D_STATIC_LIB=${HCCL_STATIC_INTERMEDIATE}
        -D_AICPU_TAR=${HCCL_AICPU_TAR}
        -D_AIV_SEARCH_DIR=${CMAKE_BINARY_DIR}/src/ops
        -D_EXTRACT_DIR=${CMAKE_BINARY_DIR}/static_extract
        -D_FINAL_STATIC_LIB=${HCCL_STATIC_FINAL_LIB}
        -P ${_STATIC_POSTPROCESS_DIR}/_static_repack.cmake
    DEPENDS hccl aiv_all_targets hccl_device
)

add_custom_target(hccl_static_final ALL DEPENDS ${HCCL_STATIC_FINAL_LIB})

# 安装最终静态库
install(FILES ${HCCL_STATIC_FINAL_LIB}
    DESTINATION ${INSTALL_LIBRARY_DIR}
    RENAME libhccl_static.a
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)
