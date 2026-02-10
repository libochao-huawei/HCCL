#ifndef HCCL_COM_STUB_H
#define HCCL_COM_STUB_H

#include <map>
#include <cstdint>

void RegisterOffset2AddrMap(uint64_t baseAddr);

uint64_t GetBaseAddr(uint64_t offset);

#endif