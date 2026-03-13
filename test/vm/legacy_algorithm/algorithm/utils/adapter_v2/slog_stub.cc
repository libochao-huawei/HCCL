
#include <securec.h>
#include <stdint.h>
#include <syslog.h>
#include <stdio.h>
#include <string>
#include <sys/time.h>

#include "log.h"
#include "slog.h"
#include "slog_api.h"
#include <hccl/hccl_types.h>
#include "sal_stub.h"

using namespace std;
using namespace Hccl;
static s32 stub_log_level = DLOG_ERROR;
static s32 stub_checker_level = DLOG_ERROR;
constexpr s32  LOG_TIME_STAMP_SIZE = 27;
constexpr u32 TIME_FROM_1900 = 1900;

s32 log_level_get_stub() {
    return stub_log_level;
}

void log_level_set_stub(s32 log_level) {
	stub_log_level = log_level;
}

int CheckLogLevel(int moduleId, int logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return 1;
}

int32_t dlog_setlevel(int32_t moduleId, int32_t level, int32_t enableEvent)
{
    stub_checker_level = level;
    return 0;
}

string get_log_str_from_type_stub(s32 type)
{
    string str = "";
    switch (type) {
        case DLOG_DEBUG:
            str = "[DEBUG]";
            break;
        case DLOG_INFO:
            str = "[INFO]";
            break;
        case DLOG_WARN:
            str = "[WARNING]";
            break;
        case DLOG_ERROR:
            str = "[ERROR]";
            break;
        default:
            break;
    }
    return str;
}

// 记录日志时获取当前时间字符串
HcclResult sal_get_current_time(char *timeStr, u32 len)
{
    struct timeval tv;
    time_t tmpt;
    struct tm *now;

    if (timeStr == NULL) {
        return HCCL_E_PARA;
    }

    if (0 > gettimeofday(&tv, NULL)) {
        return HCCL_E_INTERNAL;
    }

    tmpt = (time_t)tv.tv_sec;
    now = localtime(&tmpt);
    if (now == NULL) {
        return HCCL_E_INTERNAL;
    }

    int iLen = snprintf_s(timeStr, len, len, "%04d-%02d-%02d %02d:%0d:%02d.%06u",\
        now->tm_year + TIME_FROM_1900,
        now->tm_mon + 1, now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec, (u32)tv.tv_usec);
    if (iLen == -1) {
        HCCL_WARNING("Print time failed[%d]." \
            "params: time[%s], len[%u], time_format:%04d-%02d-%02d %02d:%02d:%02d.%06u",\
            iLen, timeStr, len, now->tm_year + TIME_FROM_1900, now->tm_mon + 1, now->tm_mday,
            now->tm_hour, now->tm_min, now->tm_sec, (u32)tv.tv_usec);
    }

    return HCCL_SUCCESS;
}

void sal_dlog_printf_stub(int level,char* log_buffer)
{
    char            time_stamp[LOG_TIME_STAMP_SIZE]   = {0};   /* 缓存时间戳  */

    /* 获取时间标签 */
    (void)sal_memset(time_stamp, LOG_TIME_STAMP_SIZE, 0, sizeof(time_stamp));
    (void)sal_get_current_time(time_stamp, LOG_TIME_STAMP_SIZE);

    string str_type= get_log_str_from_type_stub(level);
    printf("[%-26s][pid:%u]%s  %s\n", time_stamp, SalGetPid(), str_type.c_str(), log_buffer);
}

void DlogInner(int moduleId, int level, const char *fmt, ...)
{
    if(level < stub_log_level){
    	return;
    }

    char            stack_log_buffer[LOG_TMPBUF_SIZE];          /* 优先使用栈中的buffer, 小而快  */
    va_list         arg;
    (void)va_start(arg, fmt);   //lint !e530
    (void)sal_memset(stack_log_buffer, LOG_TMPBUF_SIZE, 0, sizeof(stack_log_buffer));
    /*
        C库标准的vsnprintf()函数在字符串超出缓存长度后返回需要的缓存空间.
        公司的安全函数库包装后的vsnprintf_s()在字符串超出缓存长度后返回值为-1, 无法根据返回值重新申请堆内存.
    */
    sal_vsnprintf(stack_log_buffer, sizeof(stack_log_buffer), (sizeof(stack_log_buffer) - 1), fmt, arg);
    va_end(arg);

    sal_dlog_printf_stub(level, stack_log_buffer);
}

void DlogRecord(int moduleId, int level, const char *fmt, ...)
{
    if((moduleId == HCCL && level < stub_log_level) || (moduleId != HCCL && level < stub_checker_level)){
    	return;
    }
    char            stack_log_buffer[LOG_TMPBUF_SIZE];          /* 优先使用栈中的buffer, 小而快  */
    va_list         arg;
    (void)va_start(arg, fmt);   //lint !e530
    (void)sal_memset(stack_log_buffer, LOG_TMPBUF_SIZE, 0, sizeof(stack_log_buffer));
    /*
        C库标准的vsnprintf()函数在字符串超出缓存长度后返回需要的缓存空间.
        公司的安全函数库包装后的vsnprintf_s()在字符串超出缓存长度后返回值为-1, 无法根据返回值重新申请堆内存.
    */
    vsnprintf_s(stack_log_buffer, sizeof(stack_log_buffer), (sizeof(stack_log_buffer) - 1), fmt, arg);
    va_end(arg);

    sal_dlog_printf_stub(level, stack_log_buffer);
}

void initLogLevel()
{
    auto setLogEnv = [](const char* env, s32& log_level) {
        char *env_var = getenv(env);
        if (env_var) {
            // 将环境变量的值转换为整数
            if(strlen(env_var) > 1 || env_var[0] < '0' || env_var[0] > '3')
            {
                return;
            }
            log_level = static_cast<s32>(atoi(env_var));
            printf("Environment variable %s is set to: %d\n", env, log_level);
        }
    };
    setLogEnv("ASCEND_LOG_LEVEL", stub_log_level);
    setLogEnv("CHECK_LOG_LEVEL", stub_checker_level);
}
