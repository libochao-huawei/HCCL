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
    COMMENT "后处理：嵌入AICPU与AIV内核"
)

add_custom_target(hccl_static_final ALL DEPENDS ${HCCL_STATIC_FINAL_LIB})

# 安装最终静态库
install(FILES ${HCCL_STATIC_FINAL_LIB}
    DESTINATION ${INSTALL_LIBRARY_DIR}
    RENAME libhccl_static.a
    ${INSTALL_OPTIONAL}
    COMPONENT hccl
)

# 兼容旧输出路径: build_out/lib/libhccl_static.a
set(HCCL_STATIC_COMPAT_LIB "${CMAKE_INSTALL_PREFIX}/lib/libhccl_static.a")
add_custom_command(
    OUTPUT ${HCCL_STATIC_COMPAT_LIB}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/lib
    COMMAND ${CMAKE_COMMAND} -E copy ${HCCL_STATIC_FINAL_LIB} ${HCCL_STATIC_COMPAT_LIB}
    DEPENDS ${HCCL_STATIC_FINAL_LIB}
    COMMENT "复制到兼容路径"
)
add_custom_target(hccl_static_compat ALL DEPENDS ${HCCL_STATIC_COMPAT_LIB})

# 兼容旧打包格式: build_out/cann-hccl-static_*.tar.gz
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i386|i686")
    set(HCCL_STATIC_TAR_ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|armv8l|armv7l")
    set(HCCL_STATIC_TAR_ARCH "aarch64")
else()
    set(HCCL_STATIC_TAR_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(HCCL_STATIC_PKG_DIR "${CMAKE_BINARY_DIR}/static_package")
set(HCCL_STATIC_TAR_NAME "cann-hccl-static_${VERSION_INFO}_linux-${HCCL_STATIC_TAR_ARCH}.tar.gz")
set(HCCL_STATIC_TAR "${CMAKE_INSTALL_PREFIX}/${HCCL_STATIC_TAR_NAME}")

add_custom_command(
    OUTPUT ${HCCL_STATIC_TAR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HCCL_STATIC_PKG_DIR}/include
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HCCL_STATIC_PKG_DIR}/lib64
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${HCCL_STATIC_PKG_DIR}/include
    COMMAND ${CMAKE_COMMAND} -E copy ${HCCL_STATIC_FINAL_LIB} ${HCCL_STATIC_PKG_DIR}/lib64/libhccl_static.a
    COMMAND ${CMAKE_COMMAND} -E tar czf ${HCCL_STATIC_TAR} include lib64
    WORKING_DIRECTORY ${HCCL_STATIC_PKG_DIR}
    DEPENDS ${HCCL_STATIC_FINAL_LIB}
    COMMENT "打包静态库tar.gz"
)
add_custom_target(hccl_static_package ALL DEPENDS ${HCCL_STATIC_TAR})
