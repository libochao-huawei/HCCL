#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <boost/interprocess/ipc/message_queue.hpp>

#include "hccl_common_defs.h"
#include "hccl_vm_log.h"
#include "cmd_base_utils.h"
#include "sim_runner_db.h"

namespace HcclSim {
template <typename T>
static void PrintTable(const std::string &header, const std::vector<T> &rows,
                    std::function<std::string(const T &)> formatter)
{
    std::cout << header << std::endl;
    for (const auto &row : rows) {
        std::cout << formatter(row) << std::endl;
    }
    std::cout << std::endl;
}

void CmdTableShow(std::string &tableName)
{
    if (tableName == "Device") {
        auto tables = RunnerDB::GetByPred<sim::Device>([](auto &&) { return true; });
        PrintTable<sim::Device>("| id | server_id | logic_id | physical_id | overflow_mode | soc_version | status |",
                                tables, [](const sim::Device &d) {
                                    return "| " + std::to_string(d.id) + " | " + std::to_string(d.server_id) + " | " +
                                        std::to_string(d.logic_id) + " | " + std::to_string(d.physical_id) + " | " +
                                        std::to_string(d.overflow_mode) + " | " + std::string(d.soc_version) +
                                        " | " + std::to_string(d.status) + " |";
                                });
    } else if (tableName == "Server") {
        auto tables = RunnerDB::GetByPred<sim::Server>([](auto &&) { return true; });
        PrintTable<sim::Server>("| id | pod_id | version |", tables, [](const sim::Server &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.pod_id) + " | " +
                std::string(tmp.version) + " |";
        });
    } else if (tableName == "Host") {
        auto tables = RunnerDB::GetByPred<sim::Host>([](auto &&) { return true; });
        PrintTable<sim::Host>("| id | server_id | ip | arch |", tables, [](const sim::Host &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.server_id) + " | " +
                std::string(tmp.ip_addr) + " | " + std::to_string(tmp.arch) + " |";
        });
    } else if (tableName == "Runner") {
        auto tables = RunnerDB::GetByPred<sim::Runner>([](auto &&) { return true; });
        PrintTable<sim::Runner>("| id | host_id | pid | thread_id | timeout_config_ms | current_ctx_id |", tables,
                                [](const sim::Runner &tmp) {
                                    return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.host_id) + " | " +
                                        std::to_string(tmp.pid) + " | " + std::to_string(tmp.thread_id) + " | " +
                                        std::to_string(tmp.timeout_config_ms) + " | " +
                                        std::to_string(tmp.current_ctx_id) + " |";
                                });
    } else if (tableName == "TaskSchedulerDevice") {
        auto tables = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([](auto &&) { return true; });
        PrintTable<sim::TaskSchedulerDevice>(
            "| id | device_id | type|", tables, [](const sim::TaskSchedulerDevice &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | " +
                    std::to_string(tmp.type) + " |";
            });
    } else if (tableName == "ComputeDie") {
        auto tables = RunnerDB::GetByPred<sim::ComputeDie>([](const sim::ComputeDie &) { return true; });
        PrintTable<sim::ComputeDie>("| id | ts_id | type |", tables, [](const sim::ComputeDie &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ts_id) + " | " +
                std::to_string(tmp.type) + " |";
        });
    } else if (tableName == "DeviceStatus") {
        auto tables = RunnerDB::GetByPred<sim::DeviceStatus>([](const sim::DeviceStatus &) { return true; });
        PrintTable<sim::DeviceStatus>("| id | device_id | overflow | sync_strat | sync_timeout | capability | run_by_host | ts_core | online |", 
            tables, [](const sim::DeviceStatus &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | " +
                std::to_string(tmp.overflow_status) + " | " + std::to_string(tmp.synchronize_strategy) + " | " +
                std::to_string(tmp.synchronize_timeout) + " | " + std::to_string(tmp.capability_mask) + " | " +
                std::to_string(tmp.run_by_host) + " | " + std::to_string(tmp.ts_core) + " | " +
                std::to_string(tmp.online_status) + " |";
            });
    } else if (tableName == "Port") {
        auto tables = RunnerDB::GetByPred<sim::Port>([](const sim::Port &) { return true; });
        PrintTable<sim::Port>(
            "| id | device_id | ccu_id | func_id | rdma_handle | name | ip_addr | protocol | status |", tables,
            [](const sim::Port &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | " +
                    std::to_string(tmp.ccu_id) + " | " + std::to_string(tmp.func_id) + " | " +
                    std::to_string(tmp.rdma_handle) + " | " + std::string(tmp.name) + " | " +
                    std::string(tmp.ip_addr) + " | " + std::to_string(tmp.protocol) + " | " +
                    std::to_string(tmp.status) + " |";
            });
    } else if (tableName == "Ccu") {
        auto tables = RunnerDB::GetByPred<sim::Ccu>([](const sim::Ccu &) { return true; });
        PrintTable<sim::Ccu>("| id | device_id | die_id | status |", tables, [](const sim::Ccu &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | " +
                std::to_string(tmp.die_id) + " | " + std::to_string(tmp.status) + " |";
        });
    } else if (tableName == "CcuResource") {
        auto tables = RunnerDB::GetByPred<sim::CcuResource>([](const sim::CcuResource &) { return true; });
        PrintTable<sim::CcuResource>("| id | ccu_id | instr_cnt |", tables,
                                    [](const sim::CcuResource &tmp) {
                                        return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ccu_id) +
                                                " | " + std::to_string(tmp.instr_cnt) + " | ";
                                    });
    } else if (tableName == "DeviceConnection") {
        auto tables = RunnerDB::GetByPred<sim::DeviceConnection>([](const sim::DeviceConnection &) { return true; });
        PrintTable<sim::DeviceConnection>(
            "| id | src_dev_id | dst_dev_id | link_type | access_by_remote |", tables,
            [](const sim::DeviceConnection &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.src_dev_id) + " | " +
                    std::to_string(tmp.dst_dev_id) + " | " + std::to_string(tmp.link_type) + " | " +
                    std::to_string(tmp.access_by_remote) + " |";
            });
    } else if (tableName == "EndPointPair") {
        auto tables = RunnerDB::GetByPred<sim::EndPointPair>([](const sim::EndPointPair &) { return true; });
        PrintTable<sim::EndPointPair>("| id | src_port | dst_port | type |", tables, [](const sim::EndPointPair &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.src_port) + " | " +
                std::to_string(tmp.dst_port) + " | " + std::to_string(tmp.type) + " |";
        });
    } else if (tableName == "CcuChannel") {
        auto tables = RunnerDB::GetByPred<sim::CcuChannel>([](const sim::CcuChannel &) { return true; });
        PrintTable<sim::CcuChannel>("| id | end_point_pair_id | channel_id | src_rank | dst_rank | src_die | dst_die |",
                                    tables, [](const sim::CcuChannel &tmp) {
                                        return "| " + std::to_string(tmp.id) + " | " +
                                            std::to_string(tmp.end_point_pair_id) + " | " +
                                            std::to_string(tmp.channel_id) + " | " + std::to_string(tmp.src_rank) +
                                            " | " + std::to_string(tmp.dst_rank) + " | " +
                                            std::to_string(tmp.src_die) + " | " + std::to_string(tmp.dst_die) + " |";
                                    });
    } else if (tableName == "Context") {
        auto tables = RunnerDB::GetByPred<sim::Context>([](const sim::Context &) { return true; });
        PrintTable<sim::Context>(
            "| id | run_id | device_id | is_default | ref_cnt | float_overflow_addr | capture_mode |", tables,
            [](const sim::Context &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.run_id) + " | " +
                    std::to_string(tmp.device_id) + " | " + std::to_string(tmp.is_default) + " | " +
                    std::to_string(tmp.ref_cnt) + " | " + std::to_string(tmp.float_overflow_addr) + " | " +
                    std::to_string(tmp.capture_mode) + " |";
            });
    } else if (tableName == "Stream") {
        auto tables = RunnerDB::GetByPred<sim::Stream>([](const sim::Stream &) { return true; });
        PrintTable<sim::Stream>(
            "| id | ctx_id | sq_base_addr | is_primary_default | is_other_default | priority | schedule_strategy | "
            "failure_mode | user_tag | overflow_switch | activated | capture_status | task_complete_status |",
            tables, [](const sim::Stream &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_id) + " | " +
                    std::to_string(tmp.sq_base_addr) + " | " + std::to_string(tmp.is_primary_default) + " | " +
                    std::to_string(tmp.is_other_default) + " | " + std::to_string(tmp.priority) + " | " +
                    std::to_string(tmp.schedule_strategy) + " | " + std::to_string(tmp.failure_mode) + " | " +
                    std::to_string(tmp.user_tag) + " | " + std::to_string(tmp.overflow_switch) + " | " +
                    std::to_string(tmp.activated) + " | " + std::to_string(tmp.capture_status) + " | " +
                    std::to_string(tmp.task_complete_status) + " |";
            });
    } else if (tableName == "Task") {
        auto tables = RunnerDB::GetByPred<sim::Task>([](const sim::Task &) { return true; });
        PrintTable<sim::Task>("| id | stream_id | cid | seq_number | type |", tables, [](const sim::Task &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.stream_id) + " | " +
                std::to_string(tmp.cid) + " | " + std::to_string(tmp.seq_number) + " | " + std::to_string(tmp.type) +
                " |";
        });
    } else if (tableName == "EventSyncTask") {
        auto tables = RunnerDB::GetByPred<sim::EventSyncTask>([](const sim::EventSyncTask &) { return true; });
        PrintTable<sim::EventSyncTask>(
            "| id | excute_time_ms | finish_time_ms | op_timeout_s |", tables, [](const sim::EventSyncTask &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.excute_time_ms) + " | " +
                       std::to_string(tmp.finish_time_ms) + " | " + std::to_string(tmp.op_timeout_s) + " |";
            });
    } else if (tableName == "Notify") {
        auto tables = RunnerDB::GetByPred<sim::Notify>([](const sim::Notify &) { return true; });
        PrintTable<sim::Notify>(
            "| id | create_ctx_id | device_notify_seq | value |", tables, [](const sim::Notify &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.create_ctx_id) + " | " +
                    std::to_string(tmp.device_notify_seq) + " | " + std::to_string(tmp.value) + " |";
            });
    } else if (tableName == "IpcNotify") {
        auto tables = RunnerDB::GetByPred<sim::IpcNotify>([](const sim::IpcNotify &) { return true; });
        PrintTable<sim::IpcNotify>(" id | notify_id | name_or_key | create_pid |", tables,
                                [](const sim::IpcNotify &tmp) {
                                    return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.notify_id) +
                                            " | " + std::string(reinterpret_cast<const char *>(tmp.name_or_key)) +
                                            " | " + std::to_string(tmp.create_pid) + " |";
                                });
    } else if (tableName == "IpcNotifyVistorList") {
        auto tables =
            RunnerDB::GetByPred<sim::IpcNotifyVistorList>([](const sim::IpcNotifyVistorList &) { return true; });
        PrintTable<sim::IpcNotifyVistorList>(
            "| ipc_id | visitor_pid |", tables, [](const sim::IpcNotifyVistorList &tmp) {
                return "| " + std::to_string(tmp.ipc_id) + " | " + std::to_string(tmp.visitor_pid) + " |";
            });
    } else if (tableName == "NotifyRecordTask") {
        auto tables = RunnerDB::GetByPred<sim::NotifyRecordTask>([](const sim::NotifyRecordTask &) { return true; });
        PrintTable<sim::NotifyRecordTask>("| notify_id |", tables, [](const sim::NotifyRecordTask &tmp) {
            return "| " + std::to_string(tmp.notify_id) + " |";
        });
    } else if (tableName == "NotifyWaitTask") {
        auto tables = RunnerDB::GetByPred<sim::NotifyWaitTask>([](const sim::NotifyWaitTask &) { return true; });
        PrintTable<sim::NotifyWaitTask>("| notify_id |", tables, [](const sim::NotifyWaitTask &tmp) {
            return "| " + std::to_string(tmp.notify_id) + " |";
        });
    } else if (tableName == "Event") {
        auto tables = RunnerDB::GetByPred<sim::Event>([](const sim::Event &) { return true; });
        PrintTable<sim::Event>("| id | create_ctx_id | event_flag | device_res_seq | created_time | status |", tables,
                            [](const sim::Event &tmp) {
                                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.create_ctx_id) +
                                        " | " + std::to_string(tmp.event_flag) + " | " +
                                        std::to_string(tmp.device_res_seq) + " | " +
                                        std::to_string(tmp.created_time) + " | " + std::to_string(tmp.status) + " |";
                            });
    } else if (tableName == "PhyMem") {
        auto tables = RunnerDB::GetByPred<sim::PhyMemBlock>([](const sim::PhyMemBlock &) { return true; });
        PrintTable<sim::PhyMemBlock>(
            "| id | device_id | addr | size | type | ref_count |", tables, [](const sim::PhyMemBlock &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | " +
                    std::to_string(tmp.addr) + " | " + std::to_string(tmp.size) + " | " + std::to_string(tmp.type) +
                    " | " + std::to_string(tmp.ref_count) + " |";
            });
    } else if (tableName == "VirMem") {
        auto tables = RunnerDB::GetByPred<sim::VirtualMemBlock>([](const sim::VirtualMemBlock &) { return true; });
        PrintTable<sim::VirtualMemBlock>(
            "| id | start_ptr | size | ctx_id | phy_mem_id | owner_pid | src_type | policy |", tables,
            [](const sim::VirtualMemBlock &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.start_ptr) + " | " +
                    std::to_string(tmp.size) + " | " + std::to_string(tmp.ctx_id) + " | " +
                    std::to_string(tmp.phy_mem_id) + " | " + std::to_string(tmp.owner_pid) + " | " +
                    std::to_string(tmp.src_type) + " | " + std::to_string(tmp.policy) + " |";
            });
    } else if (tableName == "IpcMemRecord") {
        auto tables = RunnerDB::GetByPred<sim::IpcMemRecord>([](const sim::IpcMemRecord &) { return true; });
        PrintTable<sim::IpcMemRecord>("| id | vir_mem_id | create_pid |", tables, [](const sim::IpcMemRecord &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.vir_mem_id) + " | " +
                std::to_string(tmp.create_pid) + " |";
        });
    } else if (tableName == "IpcMemWhiteList") {
        auto tables = RunnerDB::GetByPred<sim::IpcMemWhiteList>([](const sim::IpcMemWhiteList &) { return true; });
        PrintTable<sim::IpcMemWhiteList>(
            "| id | name_or_key | pid | create_pid |", tables, [](const sim::IpcMemWhiteList &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.name_or_key) + " | " +
                    std::to_string(tmp.pid) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "FdMemRecord") {
        auto tables = RunnerDB::GetByPred<sim::FdMemRecord>([](const sim::FdMemRecord &) { return true; });
        PrintTable<sim::FdMemRecord>(
            "| id | vir_mem_id | phy_mem_id | create_pid |", tables, [](const sim::FdMemRecord &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.vir_mem_id) + " | " +
                    std::to_string(tmp.phy_mem_id) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "FdMemWhiteList") {
        auto tables = RunnerDB::GetByPred<sim::FdMemWhiteList>([](const sim::FdMemWhiteList &) { return true; });
        PrintTable<sim::FdMemWhiteList>(
            "| id | name_or_key | pid | create_pid |", tables, [](const sim::FdMemWhiteList &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.name_or_key) + " | " +
                    std::to_string(tmp.pid) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "RaSocket") {
        auto tables = RunnerDB::GetByPred<sim::RaSocket>([](const sim::RaSocket &) { return true; });
        PrintTable<sim::RaSocket>("| id | device_id | role | state |", tables, [](const sim::RaSocket &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | " +
                std::to_string(tmp.role) + " | " + std::to_string(tmp.state) + " |";
        });
    } else if (tableName == "RaSocketPair") {
        auto tables = RunnerDB::GetByPred<sim::RaSocketPair>([](const sim::RaSocketPair &) { return true; });
        PrintTable<sim::RaSocketPair>("| id | server_id | client_id | ref_cnt |", tables, [](const sim::RaSocketPair &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.server_id) + " | " +
                std::to_string(tmp.client_id) + std::to_string(tmp.ref_cnt) + " |";
        });
    } else if (tableName == "MemoryLayout") {
        auto tables = RunnerDB::GetByPred<sim::MemoryLayout>([](const sim::MemoryLayout &) { return true; });
        PrintTable<sim::MemoryLayout>("| id | rank_id | base_addr | buf_type | reserved | size | global_offset |",
                                    tables, [](const sim::MemoryLayout &tmp) {
                                        return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.rank_id) +
                                                " | " + std::to_string(tmp.base_addr) + " | " +
                                                std::to_string(tmp.buf_type) + " | " + std::to_string(tmp.reserved) +
                                                " | " + std::to_string(tmp.size) + " | " +
                                                std::to_string(tmp.global_offset) + " |";
                                    });
    } else if (tableName == "SimModelData") {
        auto tables = RunnerDB::GetByPred<sim::SimModelData>([](const sim::SimModelData &) { return true; });
        PrintTable<sim::SimModelData>(
            "| id | rank_id | src_rank | dst_rank | root | rank_size | chip_type | op_type | reduce_op | data_type | "
            "data_count |",
            tables, [](const sim::SimModelData &tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.rank_id) + " | " +
                    std::to_string(tmp.src_rank) + " | " + std::to_string(tmp.dst_rank) + " | " +
                    std::to_string(tmp.root) + " | " + std::to_string(tmp.rank_size) + " | " +
                    std::to_string(tmp.chip_type) + " | " + std::to_string(tmp.op_type) + " | " +
                    std::to_string(tmp.reduce_op) + " | " + std::to_string(tmp.data_type) + " | " +
                    std::to_string(tmp.data_count) + " |";
            });
    } else if (tableName == "Rank") {
        auto tables = RunnerDB::GetByPred<sim::Rank>([](const sim::Rank &) { return true; });
        PrintTable<sim::Rank>("| id | device_id |", tables, [](const sim::Rank &tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " |";
        });
    } else {
        std::cout << "undefine table " << tableName << std::endl;
    }
}

bool CmdTableUpdate(const std::string &table, const uint64_t id, const std::string &column, const std::string &value)
{
    if (table == "Device" && column == "soc_version") {
        RunnerDB::Update<sim::Device>(id,
                                    [value](sim::Device &d) { memcpy(d.soc_version, value.data(), value.size()); });
        return true;
    } else {
        std::cout << "undefine update " << table << " [id=" << id << "]." << column << " = \"" << value << "\"" << std::endl;
        return false;
    }
}
}