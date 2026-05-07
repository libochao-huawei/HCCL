# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_kernel_compat SHARED
		hcomm_device_dlsym.cc
		hcomm_primitives_dl.cc
		hcomm_diag_dl.cc
		hcomm_device_profiling_dl.cc
		hccl_device_comm_dl.cc
	)

if(HCCL_CANN_COMPAT_850)
	target_compile_definitions(hccl_kernel_compat PRIVATE HCCL_CANN_COMPAT_850)
endif()

target_include_directories(hccl_kernel_compat PRIVATE
	${INCLUDE_LIST}
)

target_compile_options(hccl_kernel_compat PRIVATE
	$<$<CONFIG:Debug>:-g>
	$<$<CONFIG:Release>:-O3>
	-fstack-protector-all
)

target_link_options(hccl_kernel_compat PRIVATE
	-Wl,-z,relro
	-Wl,-z,now
	-Wl,-z,noexecstack
	$<$<CONFIG:Release>:-s>
)

target_link_directories(hccl_kernel_compat PRIVATE
${ASCEND_CANN_PACKAGE_PATH}/devlib/device
)

install(TARGETS hccl_kernel_compat
	LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} 
	${INSTALL_OPTIONAL}
	COMPONENT hccl
)