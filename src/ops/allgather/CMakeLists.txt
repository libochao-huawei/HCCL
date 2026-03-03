# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 先构建子目录中的算法实现，再回到本目录注册对外 op 源文件。
add_subdirectory(algo)

# 当前目录需要编入 hccl 目标的源码列表。
set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/allgather_op.cc
)

# 把 allgather op 实现加入主库 hccl。
target_sources(hccl PRIVATE
    ${src_list}
)
