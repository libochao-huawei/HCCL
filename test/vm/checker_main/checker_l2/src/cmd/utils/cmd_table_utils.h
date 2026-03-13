#ifndef HCCL_VM_TABLE_UTILS_H
#define HCCL_VM_TABLE_UTILS_H

#include <string>

namespace HcclSim {
void CmdTableShow(std::string &tableName);
bool CmdTableUpdate(const std::string &table, const uint64_t id, const std::string &column, const std::string &value);
}

#endif