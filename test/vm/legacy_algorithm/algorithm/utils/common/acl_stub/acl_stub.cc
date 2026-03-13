#include "acl_stub.h"

aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t *value)
{
    *value = 56;
    return ACL_SUCCESS;
}