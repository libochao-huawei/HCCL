#include <iostream>
#include "hccl_shm_pub.h"

static uint64_t g_memBaseAddr = 0;

void RegisterOffset2AddrMap(uint64_t baseAddr)
{
    g_memBaseAddr = baseAddr;
}

uint64_t GetBaseAddr(uint64_t offset)
{
    std::cout<<"zhf-get base addr= "<<std::hex<<g_memBaseAddr<<std::endl;
    return g_memBaseAddr;
}