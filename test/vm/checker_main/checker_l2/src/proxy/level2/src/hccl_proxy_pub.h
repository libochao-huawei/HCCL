#ifndef HCCL_PROXY_PUBLIC_H
#define HCCL_PROXY_PUBLIC_H

#include <vector>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <memory>

#include "acl/acl_rt.h"
#include "acl/acl_base.h"


#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "hccl_sim_world_pub.h"

extern thread_local std::pair<uint32_t, ShmNpuPos> curr_dev;

constexpr u32 HCCL_INVALID_PORT = 65536;  // topo_model.h 需要
constexpr uint32_t ROOTINFO_INDENTIFIER_MAX_LENGTH = 128; // sim_communicator.cc需要

#endif  // HCCL_PROXY_PUBLIC_H