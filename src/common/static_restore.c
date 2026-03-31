/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <fcntl.h>
#include <pwd.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 声明嵌入的二进制数据符号 */
extern char _binary_aicpu_hccl_tar_gz_start[];
extern char _binary_aicpu_hccl_tar_gz_end[];
/* 注意：size符号实际由链接器定义为_binary_<filename>_size，但我们可以通过end-start计算 */



/* 辅助函数：递归创建目录 */
static int mkdir_p(const char *path) {
    char tmp[1024];
    char *p = NULL;
    size_t len;
    
    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    
    /* 移除末尾的斜杠 */
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = 0;
    }
    
    /* 逐级创建目录 */
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (access(tmp, F_OK) != 0) {
                if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                    return -1;
                }
            }
            *p = '/';
        }
    }
    
    /* 创建最后一级目录 */
    if (access(tmp, F_OK) != 0) {
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
            return -1;
        }
    }
    
    return 0;
}

/* 恢复AICPU tar包的构造函数 */
__attribute__((constructor))
static void restore_aicpu_tar() {
    char target_path[1024];
    FILE *fp = NULL;
    size_t tar_size;
    
    /* 计算tar包大小 */
    tar_size = (size_t)(_binary_aicpu_hccl_tar_gz_end - _binary_aicpu_hccl_tar_gz_start);
    if (tar_size == 0) {
        /* 没有嵌入的tar包，可能是非静态构建 */
        return;
    }
    
    /* 确定目标路径：优先使用环境变量，然后使用默认路径 */
    const char *base_path = getenv("ASCEND_HOME_PATH");
    if (base_path == NULL) {
        base_path = "/usr/local/Ascend/ascend-toolkit/latest";
    }
    
    /* 构建完整目标路径 */
    snprintf(target_path, sizeof(target_path), 
             "%s/opp/built_in/op_impl/aicpu/kernel/aicpu_hccl.tar.gz",
             base_path);
    
    /* 检查文件是否已经存在且大小匹配 */
    if (access(target_path, F_OK) == 0) {
        struct stat st;
        if (stat(target_path, &st) == 0 && (size_t)st.st_size == tar_size) {
            /* 文件已存在且大小匹配，跳过恢复 */
            return;
        }
    }
    
    /* 创建目标目录 */
    char *dir_end = strrchr(target_path, '/');
    if (dir_end != NULL) {
        *dir_end = '\0';
        if (mkdir_p(target_path) != 0) {
            fprintf(stderr, "Warning: Failed to create directory: %s\n", target_path);
            return;
        }
        *dir_end = '/';
    }
    
    /* 写入tar包文件 */
    fp = fopen(target_path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Warning: Failed to open file for writing: %s\n", target_path);
        return;
    }
    
    size_t written = fwrite(_binary_aicpu_hccl_tar_gz_start, 1, tar_size, fp);
    fclose(fp);
    
    if (written != tar_size) {
        fprintf(stderr, "Warning: Failed to write tar file, expected %zu bytes, wrote %zu bytes\n",
                tar_size, written);
        unlink(target_path);  /* 删除不完整的文件 */
        return;
    }
    
    /* 设置文件权限 */
    chmod(target_path, 0644);
    
    /* 输出恢复成功信息（调试用） */
    fprintf(stderr, "AICPU tar package restored to: %s (%zu bytes)\n", 
            target_path, tar_size);
}

/* 获取AIV算子文件的函数（供外部调用） */
void *get_aiv_kernel_data(const char *kernel_name, size_t *size) {
    /* 简化实现：直接返回NULL，实际符号由链接器生成 */
    if (size != NULL) {
        *size = 0;
    }
    return NULL;
}

#ifdef __cplusplus
}
#endif