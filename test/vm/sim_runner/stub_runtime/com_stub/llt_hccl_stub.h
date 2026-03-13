#ifndef __LLT_HCCL_STUB_H__
#define __LLT_HCCL_STUB_H__

#include "llt_hccl_stub_pub.h"

/** 强, 弱符号别名, 用于桩函数实现 */
#if !defined(weak_alias)
#define weak_alias(name, aliasname) _weak_alias(name, aliasname)
#define _weak_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((weak, alias(#name)));
#endif

#if !defined(strong_alias)
#define strong_alias(name, aliasname) _strong_alias(name, aliasname)
#define _strong_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((alias(#name)));
#endif

#endif /* __LLT_HCCL_STUB_H__ */
