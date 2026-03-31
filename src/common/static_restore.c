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

/* 声明AIV算子符号的通用宏 */
#define DECLARE_AIV_SYMBOLS(name) \
    extern char _binary_##name##_start[]; \
    extern char _binary_##name##_end[]

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
    const char *possible_paths[] = {
        "${HOME}/Ascend/ascend-toolkit/latest/opp/built_in/op_impl/aicpu/kernel",
        "${HOME}/Ascend/latest/opp/built_in/op_impl/aicpu/kernel",
        "/usr/local/Ascend/ascend-toolkit/latest/opp/built_in/op_impl/aicpu/kernel",
        "/usr/local/Ascend/latest/opp/built_in/op_impl/aicpu/kernel",
        "/usr/local/Ascend/cann/opp/built_in/op_impl/aicpu/kernel",
        NULL
    };
    
    const char *home_path = getenv("HOME");
    char target_path[1024];
    char expanded_path[1024];
    FILE *fp = NULL;
    size_t tar_size;
    int i;
    
    /* 计算tar包大小 */
    tar_size = (size_t)(_binary_aicpu_hccl_tar_gz_end - _binary_aicpu_hccl_tar_gz_start);
    if (tar_size == 0) {
        /* 没有嵌入的tar包，可能是非静态构建 */
        return;
    }
    
    /* 尝试每个可能的路径 */
    for (i = 0; possible_paths[i] != NULL; i++) {
        const char *path_template = possible_paths[i];
        
        /* 展开HOME环境变量 */
        if (strstr(path_template, "${HOME}") != NULL) {
            if (home_path == NULL) {
                /* 如果HOME环境变量不存在，使用当前用户的主目录 */
                struct passwd *pw = getpwuid(getuid());
                if (pw != NULL) {
                    home_path = pw->pw_dir;
                } else {
                    continue;  /* 无法确定HOME路径，跳过此选项 */
                }
            }
            
            /* 替换${HOME} */
            const char *home_placeholder = "${HOME}";
            size_t home_len = strlen(home_placeholder);
            const char *pos = strstr(path_template, home_placeholder);
            if (pos != NULL) {
                size_t before_len = pos - path_template;
                size_t after_len = strlen(pos + home_len);
                if (before_len + strlen(home_path) + after_len + 20 >= sizeof(expanded_path)) {
                    continue;  /* 路径太长 */
                }
                
                strncpy(expanded_path, path_template, before_len);
                expanded_path[before_len] = '\0';
                strcat(expanded_path, home_path);
                strcat(expanded_path, pos + home_len);
            } else {
                strncpy(expanded_path, path_template, sizeof(expanded_path) - 1);
                expanded_path[sizeof(expanded_path) - 1] = '\0';
            }
        } else {
            strncpy(expanded_path, path_template, sizeof(expanded_path) - 1);
            expanded_path[sizeof(expanded_path) - 1] = '\0';
        }
        
        /* 构建完整目标路径 */
        snprintf(target_path, sizeof(target_path), "%s/aicpu_hccl.tar.gz", expanded_path);
        
        /* 检查文件是否已经存在 */
        if (access(target_path, F_OK) == 0) {
            /* 文件已存在，检查大小是否匹配 */
            struct stat st;
            if (stat(target_path, &st) == 0 && (size_t)st.st_size == tar_size) {
                /* 文件已存在且大小匹配，跳过恢复 */
                return;
            }
        }
        
        /* 创建目标目录 */
        if (mkdir_p(expanded_path) != 0) {
            /* 无法创建目录，尝试下一个路径 */
            continue;
        }
        
        /* 写入tar包文件 */
        fp = fopen(target_path, "wb");
        if (fp == NULL) {
            /* 无法打开文件，尝试下一个路径 */
            continue;
        }
        
        size_t written = fwrite(_binary_aicpu_hccl_tar_gz_start, 1, tar_size, fp);
        fclose(fp);
        
        if (written != tar_size) {
            /* 写入失败，删除不完整的文件并尝试下一个路径 */
            unlink(target_path);
            continue;
        }
        
        /* 设置文件权限 */
        chmod(target_path, 0644);
        
        /* 输出恢复成功信息（调试用） */
        fprintf(stderr, "AICPU tar package restored to: %s (%zu bytes)\n", 
                target_path, tar_size);
        return;
    }
    
    /* 所有路径都失败 */
    fprintf(stderr, "Warning: Failed to restore AICPU tar package to any location\n");
}

/* 获取AIV算子文件的函数（供外部调用） */
void *get_aiv_kernel_data(const char *kernel_name, size_t *size) {
    /* 这里需要根据具体的AIV算子文件名生成符号名 */
    char clean_name[256];
    char symbol_name[256];
    size_t i, j;
    
    /* 清理kernel_name，将非字母数字字符转换为下划线 */
    j = 0;
    for (i = 0; kernel_name[i] != '\0' && j < sizeof(clean_name) - 1; i++) {
        char c = kernel_name[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
            clean_name[j++] = c;
        } else {
            if (j > 0 && clean_name[j-1] != '_') {
                clean_name[j++] = '_';
            }
        }
    }
    clean_name[j] = '\0';
    
    /* 去除末尾的下划线 */
    while (j > 0 && clean_name[j-1] == '_') {
        clean_name[--j] = '\0';
    }
    
    /* 生成符号名 */
    snprintf(symbol_name, sizeof(symbol_name), "%s", clean_name);
    
    /* 动态查找符号 - 注意：这需要运行时链接器支持 */
    /* 在实际实现中，可能需要使用dlsym或类似的动态符号查找 */
    
    /* 简化实现：预定义的AIV算子符号 */
    struct aiv_symbol {
        const char *name;
        void *start;
        void *end;
    };
    
    static const struct aiv_symbol aiv_symbols[] = {
        /* 这些符号实际由链接器生成，这里只是占位符 */
        {"hccl_aiv_all_gather_op_910_95", NULL, NULL},
        {"hccl_aiv_all_reduce_op_910_95", NULL, NULL},
        {"hccl_aiv_all_to_all_v_op_910_95", NULL, NULL},
        {"hccl_aiv_all_to_all_op_910_95", NULL, NULL},
        {"hccl_aiv_broadcast_op_910_95", NULL, NULL},
        {"hccl_aiv_reduce_op_910_95", NULL, NULL},
        {"hccl_aiv_reduce_scatter_op_910_95", NULL, NULL},
        {"hccl_aiv_scatter_op_910_95", NULL, NULL},
        {NULL, NULL, NULL}
    };
    
    for (i = 0; aiv_symbols[i].name != NULL; i++) {
        if (strcmp(aiv_symbols[i].name, symbol_name) == 0) {
            /* 在实际构建中，这些符号会被链接器填充 */
            if (size != NULL) {
                if (aiv_symbols[i].start != NULL && aiv_symbols[i].end != NULL) {
                    *size = (size_t)((char *)aiv_symbols[i].end - (char *)aiv_symbols[i].start);
                } else {
                    *size = 0;
                }
            }
            return aiv_symbols[i].start;
        }
    }
    
    /* 未找到对应的AIV算子 */
    if (size != NULL) {
        *size = 0;
    }
    return NULL;
}

#ifdef __cplusplus
}
#endif