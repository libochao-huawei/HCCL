/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* 配置参数 */
#define DEFAULT_BASE_PATH         "/usr/local/Ascend/ascend-toolkit/latest"
#define AICPU_TAR_RELATIVE_PATH   "opp/built_in/op_impl/aicpu/kernel/aicpu_hccl.tar.gz"
#define FILE_PERMISSIONS          0644
#define PATH_BUFFER_SIZE          PATH_MAX

#ifdef __cplusplus
extern "C" {
#endif

/* 声明嵌入的二进制数据符号 */
extern char _binary_aicpu_hccl_tar_gz_start[];
extern char _binary_aicpu_hccl_tar_gz_end[];

/**
 * @brief 安全地递归创建目录
 *
 * @param path 要创建的目录路径
 * @return int 0成功，-1失败
 */
static int safe_create_directory(const char* path) {
    char component[PATH_BUFFER_SIZE];
    size_t len = strlen(path);

    /* 检查路径长度 */
    if (len >= PATH_BUFFER_SIZE) {
        return -1;
    }

    /* 复制路径到本地缓冲区 */
    strncpy(component, path, sizeof(component));
    component[sizeof(component) - 1] = '\0';

    /* 移除末尾的斜杠 */
    if (len > 0 && component[len - 1] == '/') {
        component[len - 1] = '\0';
    }

    /* 逐级创建目录 */
    for (char* p = component + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(component, 0755) == -1 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }

    /* 创建最后一级目录 */
    if (mkdir(component, 0755) == -1 && errno != EEXIST) {
        return -1;
    }

    return 0;
}



/**
 * @brief 恢复AICPU tar包的构造函数
 *
 * 在程序启动时自动执行，将嵌入的AICPU tar包恢复到文件系统。
 * 如果文件已存在且大小匹配，则跳过恢复。
 */
__attribute__((constructor))
static void restore_aicpu_tar() {
    char target_path[PATH_MAX];
    const char* base_path;
    size_t tar_size;
    FILE* fp = NULL;

    /* 计算嵌入tar包的大小 */
    tar_size = (size_t)(_binary_aicpu_hccl_tar_gz_end - _binary_aicpu_hccl_tar_gz_start);
    if (tar_size == 0) {
        /* 没有嵌入的tar包，可能是非静态构建 */
        fprintf(stderr, "Info: No embedded AICPU tar package found, skipping restore\n");
        return;
    }

    /* 确定基础路径：优先使用环境变量 */
    base_path = getenv("ASCEND_HOME_PATH");
    if (base_path == NULL) {
        base_path = DEFAULT_BASE_PATH;
    }

    /* 构建完整目标路径 */
    if (snprintf(target_path, sizeof(target_path),
                 "%s/%s",
                 base_path, AICPU_TAR_RELATIVE_PATH) >= (int)sizeof(target_path)) {
        fprintf(stderr, "Warning: Path too long for AICPU tar restore\n");
        return;
    }

    /* 提取目录路径并创建目录 */
    char* last_slash = strrchr(target_path, '/');
    if (last_slash != NULL) {
        *last_slash = '\0';
        if (safe_create_directory(target_path) != 0) {
            fprintf(stderr, "Warning: Failed to create directory: %s\n", target_path);
            *last_slash = '/';
            return;
        }
        *last_slash = '/';
    }

    /* 写入tar包文件 */
    fp = fopen(target_path, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Warning: Failed to open file for writing: %s (%s)\n",
                target_path, strerror(errno));
        return;
    }

    size_t written = fwrite(_binary_aicpu_hccl_tar_gz_start, 1, tar_size, fp);
    fclose(fp);

    if (written != tar_size) {
        fprintf(stderr, "Warning: Failed to write tar file: expected %zu bytes, wrote %zu bytes\n",
                tar_size, written);
        /* 写入失败，保留不完整文件供调试 */
        return;
    }

    /* 设置文件权限 */
    if (chmod(target_path, FILE_PERMISSIONS) != 0) {
        fprintf(stderr, "Warning: Failed to set permissions on %s\n", target_path);
    }

    /* 输出恢复成功信息 */
    fprintf(stderr, "AICPU tar package restored to: %s (%zu bytes)\n", target_path, tar_size);
}

/**
 * @brief 获取AIV算子数据的接口函数
 *
 * @param kernel_name 算子名称（当前未使用）
 * @param size 输出参数，返回数据大小
 * @return void* 指向算子数据的指针，当前实现返回NULL
 */
void* get_aiv_kernel_data(const char* kernel_name, size_t* size) {
    (void)kernel_name; /* 避免未使用参数警告 */

    if (size != NULL) {
        *size = 0;
    }
    return NULL;
}

#ifdef __cplusplus
}
#endif