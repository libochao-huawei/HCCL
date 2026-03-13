#ifndef ADAPTER_V2_SAL_STUB_H
#define ADAPTER_V2_SAL_STUB_H

namespace Hccl {

extern HcclResult sal_memset(void *dest, size_t destMaxSize, int c, size_t count);
s32 SalGetPid();
s32 SalGetTid();
s32 sal_vsnprintf(char *strDest, size_t destMaxSize, size_t count, const char *format, va_list argList);

}

#endif