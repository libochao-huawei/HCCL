/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* 确保 fileno() 可用 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
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

/* CRC32 查找表 */
static uint32_t crc32_table[256];
static int crc32_table_initialized = 0;

/**
 * @brief 初始化 CRC32 查找表
 */
static void init_crc32_table(void) {
    if (crc32_table_initialized) {
        return;
    }

    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
        }
        crc32_table[i] = crc;
    }
    crc32_table_initialized = 1;
}

/**
 * @brief 计算内存数据的 CRC32 校验值
 *
 * @param data 数据指针
 * @param length 数据长度
 * @return uint32_t CRC32 校验值
 */
static uint32_t calc_crc32(const void* data, size_t length) {
    init_crc32_table();

    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < length; i++) {
        crc = (crc >> 8) ^ crc32_table[(crc ^ bytes[i]) & 0xFF];
    }

    return crc ^ 0xFFFFFFFF;
}

/**
 * @brief 检查路径是否为安全的绝对路径
 *
 * 安全路径定义：
 * 1. 非空
 * 2. 绝对路径（以/开头）
 * 3. 仅包含白名单字符（字母、数字、/_-.）
 * 4. 不包含路径穿越序列（..）
 *
 * @param path 待检查的路径
 * @return int 1 表示安全，0 表示不安全
 */
static int is_safe_path(const char* path) {
    /* 1. 路径不能为空 */
    if (path == NULL || *path == '\0') {
        fprintf(stderr, "Error: Path is empty\n");
        return 0;
    }

    /* 2. 必须是绝对路径 */
    if (path[0] != '/') {
        fprintf(stderr, "Error: Path must be absolute, got '%s'\n", path);
        return 0;
    }

    /* 3. 白名单字符校验 */
    const char* whitelist = "abcdefghijklmnopqrstuvwxyz"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "0123456789/_-.";

    for (const char* p = path; *p != '\0'; p++) {
        if (strchr(whitelist, *p) == NULL) {
            fprintf(stderr, "Error: Path contains invalid character at position %ld: '%c'\n",
                    (long)(p - path), *p);
            return 0;
        }
    }

    /* 4. 检测路径穿越攻击 (..) */
    if (strstr(path, "/../") != NULL) {
        fprintf(stderr, "Error: Path contains traversal sequence '../': '%s'\n", path);
        return 0;
    }

    /* 检查路径是否以 /.. 结尾 */
    size_t len = strlen(path);
    if (len >= 3 && strcmp(path + len - 3, "/..") == 0) {
        fprintf(stderr, "Error: Path ends with traversal sequence '/..': '%s'\n", path);
        return 0;
    }

    /* 5. 检测 /.// 路径穿越（当前目录引用） */
    if (strstr(path, "/./") != NULL) {
        fprintf(stderr, "Error: Path contains current directory sequence './': '%s'\n", path);
        return 0;
    }

    /* 检查路径是否以 /./ 结尾 */
    if (len >= 3 && strcmp(path + len - 3, "/./") == 0) {
        fprintf(stderr, "Error: Path ends with current directory sequence '/./': '%s'\n", path);
        return 0;
    }

    /* 6. 检查连续的斜杠 */
    if (strstr(path, "//") != NULL) {
        fprintf(stderr, "Error: Path contains double slash: '%s'\n", path);
        return 0;
    }

    /* 7. 检查连续的点号（除了合法的 ..） */
    if (strstr(path, "...") != NULL) {
        fprintf(stderr, "Error: Path contains invalid dot sequence: '%s'\n", path);
        return 0;
    }

    return 1;
}

/**
 * @brief 安全地复制字符串，确保目标缓冲区以 null 结尾
 *
 * @param dest 目标缓冲区
 * @param dest_size 目标缓冲区大小
 * @param src 源字符串
 * @return int 0 成功，-1 失败（源字符串过长被截断）
 */
static int safe_strcpy(char* dest, size_t dest_size, const char* src) {
    if (dest == NULL || src == NULL || dest_size == 0) {
        return -1;
    }

    size_t src_len = strlen(src);
    if (src_len >= dest_size) {
        /* 源字符串过长，进行截断 */
        memcpy(dest, src, dest_size - 1);
        dest[dest_size - 1] = '\0';
        return -1;
    }

    memcpy(dest, src, src_len + 1);
    return 0;
}

/**
 * @brief 安全地递归创建目录
 *
 * @param path 要创建的目录路径
 * @return int 0 成功，-1 失败
 */
static int safe_create_directory(const char* path) {
    char component[PATH_BUFFER_SIZE];
    size_t len;
    struct stat st;

    if (path == NULL) {
        return -1;
    }

    len = strlen(path);

    /* 检查路径长度 */
    if (len >= PATH_BUFFER_SIZE) {
        fprintf(stderr, "Error: Path too long for mkdir: %zu chars\n", len);
        return -1;
    }

    /* 安全复制路径到本地缓冲区 */
    if (safe_strcpy(component, sizeof(component), path) != 0) {
        return -1;
    }

    /* 移除末尾的斜杠 */
    if (len > 0 && component[len - 1] == '/') {
        component[len - 1] = '\0';
        len--;
    }

    /* 逐级创建目录 */
    for (char* p = component + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(component, 0755) == -1) {
                if (errno != EEXIST) {
                    fprintf(stderr, "Error: Failed to create directory '%s': %s\n",
                            component, strerror(errno));
                    return -1;
                }
                /* 目录已存在，验证它是真正的目录而非符号链接 */
                if (lstat(component, &st) == 0) {
                    if (S_ISLNK(st.st_mode)) {
                        fprintf(stderr, "Error: Path '%s' is a symbolic link, refusing to create\n",
                                component);
                        return -1;
                    }
                    if (!S_ISDIR(st.st_mode)) {
                        fprintf(stderr, "Error: Path '%s' exists but is not a directory\n",
                                component);
                        return -1;
                    }
                }
            }
            *p = '/';
        }
    }

    /* 创建最后一级目录 */
    if (mkdir(component, 0755) == -1) {
        if (errno != EEXIST) {
            fprintf(stderr, "Error: Failed to create directory '%s': %s\n",
                    component, strerror(errno));
            return -1;
        }
        /* 目录已存在，验证它是真正的目录而非符号链接 */
        if (lstat(component, &st) == 0) {
            if (S_ISLNK(st.st_mode)) {
                fprintf(stderr, "Error: Path '%s' is a symbolic link, refusing to create\n",
                        component);
                return -1;
            }
            if (!S_ISDIR(st.st_mode)) {
                fprintf(stderr, "Error: Path '%s' exists but is not a directory\n",
                        component);
                return -1;
            }
        }
    }

    return 0;
}

/**
 * @brief 获取并校验基础路径
 *
 * 优先使用环境变量 ASCEND_HOME_PATH，如果未设置或不安全则使用默认值
 *
 * @return const char* 安全的基础路径（静态缓冲区）
 */
static const char* get_safe_base_path(void) {
    static char safe_path[PATH_BUFFER_SIZE];
    const char* env_path;

    env_path = getenv("ASCEND_HOME_PATH");

    if (env_path != NULL && is_safe_path(env_path)) {
        if (safe_strcpy(safe_path, sizeof(safe_path), env_path) == 0) {
            return safe_path;
        }
    }

    /* 使用默认路径 */
    if (safe_strcpy(safe_path, sizeof(safe_path), DEFAULT_BASE_PATH) != 0) {
        return NULL;
    }

    return safe_path;
}

/**
 * @brief 构建完整的目标文件路径
 *
 * @param base_path 基础路径
 * @param relative_path 相对路径
 * @param output 输出缓冲区
 * @param output_size 输出缓冲区大小
 * @return int 0 成功，-1 失败
 */
static int build_safe_path(const char* base_path, const char* relative_path,
                          char* output, size_t output_size) {
    size_t base_len, rel_len;

    if (base_path == NULL || relative_path == NULL || output == NULL) {
        return -1;
    }

    base_len = strlen(base_path);
    rel_len = strlen(relative_path);

    /* 检查总长度是否溢出 */
    if (base_len + rel_len + 2 > output_size) {
        fprintf(stderr, "Warning: Combined path too long\n");
        return -1;
    }

    /* 安全拼接路径 */
    int ret = snprintf(output, output_size, "%s/%s", base_path, relative_path);
    if (ret < 0 || (size_t)ret >= output_size) {
        fprintf(stderr, "Warning: Path truncation occurred\n");
        return -1;
    }

    /* 校验生成的路径 */
    if (!is_safe_path(output)) {
        return -1;
    }

    return 0;
}

/**
 * @brief 打开并获取目标文件的排他锁
 *
 * 直接对目标文件本身加锁，避免单独的锁文件与目标文件权限不一致的问题。
 *
 * @param target_path 目标文件路径
 * @param file_exists 输出参数，文件是否存在
 * @return FILE* 文件指针（>=0 成功），NULL 失败
 */
static FILE* open_and_lock_target(const char* target_path, int* file_exists) {
    FILE* fp = NULL;
    int fd;

    if (file_exists != NULL) {
        *file_exists = 0;
    }

    /* 尝试以读写方式打开现有文件 */
    fp = fopen(target_path, "r+b");
    if (fp == NULL) {
        if (errno == ENOENT) {
            /* 文件不存在，需要创建 */
            if (file_exists != NULL) {
                *file_exists = 0;
            }
        } else {
            /* 其他错误（权限不足等） */
            fprintf(stderr, "Error: Cannot open target file '%s': %s\n",
                    target_path, strerror(errno));
            return NULL;
        }
    } else {
        /* 文件已存在 */
        if (file_exists != NULL) {
            *file_exists = 1;
        }
    }

    /* 如果文件不存在，创建它 */
    if (fp == NULL) {
        fp = fopen(target_path, "w+b");
        if (fp == NULL) {
            fprintf(stderr, "Error: Cannot create target file '%s': %s\n",
                    target_path, strerror(errno));
            return NULL;
        }
    }

    /* 获取文件描述符并加锁 */
    fd = fileno(fp);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot get file descriptor for '%s'\n", target_path);
        fclose(fp);
        return NULL;
    }

    /* 获取排他锁（阻塞式） */
    if (flock(fd, LOCK_EX) < 0) {
        fprintf(stderr, "Error: Failed to acquire lock on '%s': %s\n",
                target_path, strerror(errno));
        fclose(fp);
        return NULL;
    }

    return fp;
}

/**
 * @brief 释放文件锁并关闭文件
 *
 * @param fp 文件指针
 */
static void unlock_and_close(FILE* fp) {
    if (fp != NULL) {
        int fd = fileno(fp);
        if (fd >= 0) {
            flock(fd, LOCK_UN);
        }
        fclose(fp);
    }
}

/**
 * @brief 检查并比较文件 CRC
 *
 * @param file_path 文件路径
 * @param expected_data 期望的数据
 * @param expected_size 期望的数据大小
 * @param expected_crc 期望的 CRC 值（0 表示不传入，函数内计算）
 * @return int 1 表示文件存在且 CRC 匹配，0 表示不匹配或文件不存在
 */
static int check_file_integrity(const char* file_path, const void* expected_data,
                                size_t expected_size, uint32_t expected_crc) {
    struct stat st;
    FILE* fp = NULL;
    uint8_t* buffer = NULL;
    uint32_t existing_crc;
    uint32_t computed_crc;

    /* 检查文件是否存在 */
    if (stat(file_path, &st) != 0) {
        return 0; /* 文件不存在 */
    }

    /* 检查文件大小 */
    if ((size_t)st.st_size != expected_size) {
        fprintf(stderr, "Info: Existing file size (%ld) differs from embedded (%zu), will overwrite\n",
                (long)st.st_size, expected_size);
        return 0;
    }

    /* 读取现有文件内容 */
    fp = fopen(file_path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Warning: Cannot open existing file '%s' for comparison: %s\n",
                file_path, strerror(errno));
        return 0;
    }

    buffer = (uint8_t*)malloc(expected_size);
    if (buffer == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for integrity check\n");
        fclose(fp);
        return 0;
    }

    size_t read_size = fread(buffer, 1, expected_size, fp);
    fclose(fp);

    if (read_size != expected_size) {
        fprintf(stderr, "Error: Failed to read existing file for comparison: read %zu, expected %zu\n",
                read_size, expected_size);
        free(buffer);
        return 0;
    }

    /* 比较 CRC32 */
    existing_crc = calc_crc32(buffer, expected_size);
    free(buffer);

    /* 使用传入的 CRC 或重新计算 */
    if (expected_crc != 0) {
        computed_crc = expected_crc;
    } else {
        computed_crc = calc_crc32(expected_data, expected_size);
    }

    if (existing_crc != computed_crc) {
        fprintf(stderr, "Info: Existing file CRC (0x%08X) differs from embedded (0x%08X), will overwrite\n",
                existing_crc, computed_crc);
        return 0;
    }

    /* 文件完整且 CRC 匹配，无需恢复 */
    return 1;
}

/**
 * @brief 删除写入失败的不完整文件
 */
static void remove_incomplete_file(const char* path) {
    if (unlink(path) != 0) {
        fprintf(stderr, "Warning: Failed to remove incomplete file '%s': %s\n",
                path, strerror(errno));
    } else {
        fprintf(stderr, "Info: Removed incomplete file '%s'\n", path);
    }
}

/**
 * @brief 解析目标路径并创建父目录
 *
 * @param target_path 输出缓冲区
 * @param path_size 缓冲区大小
 * @return int 0 成功，-1 失败
 */
static int resolve_target_path(char* target_path, size_t path_size) {
    const char* base_path = get_safe_base_path();
    if (base_path == NULL) {
        fprintf(stderr, "Warning: Failed to get safe base path, using default\n");
        base_path = DEFAULT_BASE_PATH;
    }

    if (build_safe_path(base_path, AICPU_TAR_RELATIVE_PATH, target_path, path_size) != 0) {
        fprintf(stderr, "Error: Failed to build safe target path\n");
        return -1;
    }

    char* last_slash = strrchr(target_path, '/');
    if (last_slash != NULL) {
        *last_slash = '\0';
        int ret = safe_create_directory(target_path);
        *last_slash = '/';
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to create directory for: %s\n", target_path);
            return -1;
        }
    }

    return 0;
}

/**
 * @brief 将嵌入数据写入已锁定的文件并验证
 *
 * @param fp 已打开并加锁的文件指针
 * @param target_path 目标文件路径（用于错误信息和清理）
 * @param data 待写入的数据
 * @param size 数据大小
 * @param expected_crc 期望的 CRC32 校验值
 * @param file_exists 文件是否已存在（决定是否需要 truncate）
 * @return int 0 成功，-1 失败（失败时文件已被清理）
 */
static int write_and_verify_tar(FILE* fp, const char* target_path,
                                const char* data, size_t size,
                                uint32_t expected_crc, int file_exists) {
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Error: Cannot seek to beginning of '%s'\n", target_path);
        return -1;
    }

    if (file_exists && ftruncate(fileno(fp), 0) != 0) {
        fprintf(stderr, "Error: Cannot truncate '%s': %s\n", target_path, strerror(errno));
        return -1;
    }

    size_t written = fwrite(data, 1, size, fp);
    if (written != size) {
        fprintf(stderr, "Error: Failed to write tar file: expected %zu bytes, wrote %zu\n",
                size, written);
        fclose(fp);
        remove_incomplete_file(target_path);
        return -1;
    }

    if (fflush(fp) != 0) {
        fprintf(stderr, "Error: Failed to flush file '%s': %s\n", target_path, strerror(errno));
        fclose(fp);
        remove_incomplete_file(target_path);
        return -1;
    }

    uint32_t written_crc = calc_crc32(data, size);
    if (written_crc != expected_crc) {
        fprintf(stderr, "Error: Post-write CRC check failed: 0x%08X vs expected 0x%08X\n",
                written_crc, expected_crc);
        fclose(fp);
        remove_incomplete_file(target_path);
        return -1;
    }

    return 0;
}

/**
 * @brief 恢复 AICPU tar 包的构造函数
 *
 * 在程序启动时自动执行，将嵌入的 AICPU tar 包恢复到文件系统。
 * 使用文件锁确保多进程场景下的安全性，并在恢复前进行完整性校验。
 */
__attribute__((constructor))
static void restore_aicpu_tar(void) {
    char target_path[PATH_MAX];
    int file_exists = 0;

    size_t tar_size = (size_t)(_binary_aicpu_hccl_tar_gz_end - _binary_aicpu_hccl_tar_gz_start);
    if (tar_size == 0) {
        fprintf(stderr, "Info: No embedded AICPU tar package found, skipping restore\n");
        return;
    }

    uint32_t embedded_crc = calc_crc32(_binary_aicpu_hccl_tar_gz_start, tar_size);

    if (resolve_target_path(target_path, sizeof(target_path)) != 0) {
        return;
    }

    FILE* fp = open_and_lock_target(target_path, &file_exists);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to acquire file lock on '%s'. Aborting restore.\n",
                target_path);
        return;
    }

    if (file_exists && check_file_integrity(target_path, _binary_aicpu_hccl_tar_gz_start,
                                            tar_size, embedded_crc)) {
        fprintf(stderr, "Info: AICPU tar file already exists with matching CRC, skipping restore\n");
        unlock_and_close(fp);
        return;
    }

    if (write_and_verify_tar(fp, target_path, _binary_aicpu_hccl_tar_gz_start,
                             tar_size, embedded_crc, file_exists) != 0) {
        return;
    }

    unlock_and_close(fp);

    if (chmod(target_path, FILE_PERMISSIONS) != 0) {
        fprintf(stderr, "Warning: Failed to set permissions on %s: %s\n",
                target_path, strerror(errno));
    }

    fprintf(stderr, "Info: AICPU tar package restored to: %s (%zu bytes, CRC: 0x%08X)\n",
            target_path, tar_size, embedded_crc);
}

/**
 * @brief 获取 AIV 算子数据的接口函数
 *
 * @param kernel_name 算子名称（当前未使用）
 * @param size 输出参数，返回数据大小
 * @return void* 指向算子数据的指针，当前实现返回 NULL
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
