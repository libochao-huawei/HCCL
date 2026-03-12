#ifndef _SIM_RUNNER_OPS_H_
#define _SIM_RUNNER_OPS_H_

#include <thread>
#include "acl/acl_base.h"
#include "hccl_proxy_pub.h"
#include "sim_models.h"
#include "sim_runner_db.h"

namespace sim {

const Runner &GetCurrRunnerTls();
bool SetCurrCtxTls(uint64_t ctx);
uint64_t GetCurrRankId();
uint64_t GetRankIdByCtxId(uint64_t ctxId);


void SetLastStreamIdTls(uint64_t streamId);
void SetLastTaskIdTls(uint64_t taskId);

uint64_t GetLastStreamIdTls();
uint64_t GetLastTaskIdTls();


void SetTsDevice(int tsId);
uint32_t GetRankSize();
uint32_t GetHostSize();
uint32_t GetCurrentStreamId(uint64_t streamKey);
}
#endif