#include <string>
#include <securec.h>
#include <stdio.h>
#include "semantics_utils.h"

using namespace Hccl;

namespace checker {

std::string GenMsg(const char* fileName, s32 lineNum, const char *fmt, ...)
{
    char buffer[LOG_TMPBUF_SIZE];
    va_list arg;
    (void)va_start(arg, fmt);
    (void)memset_s(buffer, LOG_TMPBUF_SIZE, 0, sizeof(buffer));
    vsnprintf_s(buffer, sizeof(buffer), (sizeof(buffer) - 1), fmt, arg);
    va_end(arg);

    char ret[LOG_TMPBUF_SIZE];
    snprintf_s(ret, sizeof(ret), (sizeof(ret) - 1), "[%s:%d]%s", fileName, lineNum, buffer);

    return std::string(ret);
}

}
