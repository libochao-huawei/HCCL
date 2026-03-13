# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Manual compilation of ASC kernel to object file for dynamic loading
set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/launch_kernel_asc.asc)
set(KERNEL_TMP_OBJ ${CMAKE_CURRENT_BINARY_DIR}/hccl_custom_allgather_kernels_tmp.o)
set(KERNEL_OBJ ${CMAKE_CURRENT_BINARY_DIR}/hccl_custom_allgather_kernels.o)

# Include paths for ASC compiler
set(ASC_INCLUDE_FLAGS 
    -I${CMAKE_CURRENT_SOURCE_DIR}/../inc
    -I${ASCEND_CANN_PACKAGE_PATH}/include
    -I${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    -I${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
)

# Use CMAKE_ASC_COMPILER if available, otherwise default to 'asc'
if(NOT CMAKE_ASC_COMPILER)
    set(CMAKE_ASC_COMPILER "asc")
endif()

# Find linker
find_program(LLD_LINKER ld.lld PATHS "${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin" NO_DEFAULT_PATH)
if(NOT LLD_LINKER)
    set(LLD_LINKER "ld.lld")
endif()

# Compile ASC kernel
add_custom_command(
    OUTPUT ${KERNEL_OBJ}
    # 1. Compile to temporary object file
    COMMAND ${CMAKE_ASC_COMPILER} -c -o ${KERNEL_TMP_OBJ} ${KERNEL_SRC} --npu-arch=dav-2201 ${ASC_INCLUDE_FLAGS}
    # 2. Link object file (essential for aclrtBinaryLoad)
    COMMAND ${LLD_LINKER} -m aicorelinux -Ttext=0 ${KERNEL_TMP_OBJ} -static -o ${KERNEL_OBJ}
    DEPENDS ${KERNEL_SRC}
    COMMENT "Compiling and Linking ASC kernel: ${KERNEL_SRC} -> ${KERNEL_OBJ}"
    VERBATIM
)

add_custom_target(generate_kernel_obj ALL DEPENDS ${KERNEL_OBJ})

# Install the generated .o file to lib directory
install(FILES ${KERNEL_OBJ}
    DESTINATION ${CUSTOM_OPS_OPP_LIB_PATH}
)

# Copy to testcase directory for runtime loading convenience
add_custom_command(TARGET generate_kernel_obj POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${KERNEL_OBJ}
        ${CMAKE_SOURCE_DIR}/testcase/hccl_custom_allgather_kernels.o
)
