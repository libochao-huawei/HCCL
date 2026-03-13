#ifndef HCCL_LLT_COMMON_H
#define HCCL_LLT_COMMON_H

#include "base.h"
#include <chrono>

namespace checker {

using char_t = char;
#ifdef HCCL_ALG_ANALYZER_DAVID
using RankId = s32;
#else
using RankId = u32;
#endif
using ServerId = u32;
using SuperPodId = u32;
using QId = u32;
using BlockId = u32;

using HcclUs = std::chrono::steady_clock::time_point;
#define DURATION_US(x) (std::chrono::duration_cast<std::chrono::microseconds>(x))
#define TIME_NOW() ({ std::chrono::steady_clock::now(); })

}

#endif