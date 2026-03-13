#ifndef SIM_SQLITE_DB_H
#define SIM_SQLITE_DB_H

#include <tuple>
#include <type_traits>
#include <stdexcept>
#include <functional>
#include <optional>
#include <typeindex>
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>

#include "sim_models.h"
#include "sim_sqlite_table.h"

// Forward declaration to allow friend access
struct SimRunnerSqliteDB;

template <typename T>
using TableKeyType = typename sim::SqliteTable<T>::KeyType;
template <typename T>
using TableValue = typename sim::SqliteTable<T>::ValueType;

struct SimRunnerSqliteDB {
private:
    sim::SqliteDatabase m_db;

    sim::SqliteTable<sim::Server> m_serverTbl;
    sim::SqliteTable<sim::Host> m_hostTbl;
    sim::SqliteTable<sim::Runner> m_runnerTbl;
    sim::SqliteTable<sim::Device> m_deviceTbl;
    sim::SqliteTable<sim::DeviceStatus> m_deviceStatusTbl;
    sim::SqliteTable<sim::Context> m_contextTbl;
    sim::SqliteTable<sim::Stream> m_streamTbl;
    sim::SqliteTable<sim::Notify> m_notifyTbl;
    sim::SqliteTable<sim::Event> m_eventTbl;
    sim::SqliteTable<sim::VirtualMemBlock> m_virMemTbl;
    sim::SqliteTable<sim::PhyMemBlock> m_phyMemTbl;
    sim::SqliteTable<sim::IpcMemRecord> m_ipcMemRecordTbl;
    sim::SqliteTable<sim::IpcMemWhiteList> m_ipcMemWhiteListTbl;
    sim::SqliteTable<sim::FdMemRecord> m_fdMemRecordTbl;
    sim::SqliteTable<sim::FdMemWhiteList> m_fdMemWhiteListTbl;
    sim::SqliteTable<sim::Task> m_taskTbl;
    sim::SqliteTable<sim::EventSyncTask> m_eventSyncTaskTbl;
    sim::SqliteTable<sim::Port> m_portTbl;
    sim::SqliteTable<sim::Ccu> m_ccuTbl;
    sim::SqliteTable<sim::CcuResource> m_ccuResTbl;
    sim::SqliteTable<sim::DeviceConnection> m_deviceConnTbl;
    sim::SqliteTable<sim::EndPointPair> m_endPointPairTbl;
    sim::SqliteTable<sim::CcuChannel> m_ccuChannelTbl;
    sim::SqliteTable<sim::TaskSchedulerDevice> m_taskSchedulerDeviceTbl;
    sim::SqliteTable<sim::ComputeDie> m_computeDieTbl;
    sim::SqliteTable<sim::RaSocket> m_raSocketTbl;
    sim::SqliteTable<sim::RaSocketPair> m_raSocketPairTbl;
    sim::SqliteTable<sim::RaDevice> m_raDeviceTbl;
    sim::SqliteTable<sim::RaQP> m_raQPTbl;
    sim::SqliteTable<sim::RaCQ> m_raCQTbl;
    sim::SqliteTable<sim::RaCQE> m_raCQETbl;
    sim::SqliteTable<sim::RaMR> m_raMRTbl;
    sim::SqliteTable<sim::MemoryLayout> m_memoryLayoutTbl;
    sim::SqliteTable<sim::SimModelData> m_simModelDataTbl;
    sim::SqliteTable<sim::Rank> m_rankTbl;
    sim::SqliteTable<sim::IpcNotify> m_ipcNotifyTbl;
    sim::SqliteTable<sim::IpcNotifyVistorList> m_ipcNotifyVistorListTbl;
    sim::SqliteTable<sim::NotifyRecordTask> m_notifyRecordTaskTbl;
    sim::SqliteTable<sim::NotifyWaitTask> m_notifyWaitTaskTbl;

    std::unordered_map<std::type_index, sim::TableBase*> m_tableMap;
    std::vector<std::string> m_tableNames;

    template <typename T>
    void RegisterTable(sim::SqliteTable<T>& table, const std::string& name) {
        m_tableMap[std::type_index(typeid(T))] = &table;
        m_tableNames.push_back(name);
    }

public:
    SimRunnerSqliteDB() 
        : m_serverTbl(m_db.GetDb(), "Server")
        , m_hostTbl(m_db.GetDb(), "Host")
        , m_runnerTbl(m_db.GetDb(), "Runner")
        , m_deviceTbl(m_db.GetDb(), "Device")
        , m_deviceStatusTbl(m_db.GetDb(), "DeviceStatus")
        , m_contextTbl(m_db.GetDb(), "Context")
        , m_streamTbl(m_db.GetDb(), "Stream")
        , m_notifyTbl(m_db.GetDb(), "Notify")
        , m_eventTbl(m_db.GetDb(), "Event")
        , m_virMemTbl(m_db.GetDb(), "VirMem")
        , m_phyMemTbl(m_db.GetDb(), "PhMem")
        , m_ipcMemRecordTbl(m_db.GetDb(), "IpcMemRecord")
        , m_ipcMemWhiteListTbl(m_db.GetDb(), "IpMemWhiteList")
        , m_fdMemRecordTbl(m_db.GetDb(), "FdMemRecord")
        , m_fdMemWhiteListTbl(m_db.GetDb(), "FdMemWhiteList")
        , m_taskTbl(m_db.GetDb(), "Task")
        , m_eventSyncTaskTbl(m_db.GetDb(), "EventSyncTask")
        , m_portTbl(m_db.GetDb(), "Port")
        , m_ccuTbl(m_db.GetDb(), "Ccu")
        , m_ccuResTbl(m_db.GetDb(), "CcuResource")
        , m_deviceConnTbl(m_db.GetDb(), "DeviceConnection")
        , m_endPointPairTbl(m_db.GetDb(), "EndPointPair")
        , m_ccuChannelTbl(m_db.GetDb(), "CcuChannel")
        , m_taskSchedulerDeviceTbl(m_db.GetDb(), "TaskSchedulerDevice")
        , m_computeDieTbl(m_db.GetDb(), "ComputeDie")
        , m_raSocketTbl(m_db.GetDb(), "RaSocket")
        , m_raSocketPairTbl(m_db.GetDb(), "RaSocketPair")
        , m_raDeviceTbl(m_db.GetDb(), "RaDevice")
        , m_raQPTbl(m_db.GetDb(), "RaQP")
        , m_raCQTbl(m_db.GetDb(), "RaCQ")
        , m_raCQETbl(m_db.GetDb(), "RaCQE")
        , m_raMRTbl(m_db.GetDb(), "RaMR")
        , m_memoryLayoutTbl(m_db.GetDb(), "MemoryLayout")
        , m_simModelDataTbl(m_db.GetDb(), "SimModelData")
        , m_rankTbl(m_db.GetDb(), "Rank")
        , m_ipcNotifyTbl(m_db.GetDb(), "IpcNotify")
        , m_ipcNotifyVistorListTbl(m_db.GetDb(), "IpcNotifyVistorList")
        , m_notifyRecordTaskTbl(m_db.GetDb(), "NotifyRecordTask")
        , m_notifyWaitTaskTbl(m_db.GetDb(), "NotifyWaitTask")
    {
        RegisterTable(m_serverTbl, "Server");
        RegisterTable(m_hostTbl, "Host");
        RegisterTable(m_runnerTbl, "Runner");
        RegisterTable(m_deviceTbl, "Device");
        RegisterTable(m_deviceStatusTbl, "DeviceStatus");
        RegisterTable(m_contextTbl, "Context");
        RegisterTable(m_streamTbl, "Stream");
        RegisterTable(m_notifyTbl, "Notify");
        RegisterTable(m_eventTbl, "Event");
        RegisterTable(m_virMemTbl, "VirMem");
        RegisterTable(m_phyMemTbl, "PhMem");
        RegisterTable(m_ipcMemRecordTbl, "IpcMemRecord");
        RegisterTable(m_ipcMemWhiteListTbl, "IpMemWhiteList");
        RegisterTable(m_fdMemRecordTbl, "FdMemRecord");
        RegisterTable(m_fdMemWhiteListTbl, "FdMemWhiteList");
        RegisterTable(m_taskTbl, "Task");
        RegisterTable(m_eventSyncTaskTbl, "EventSyncTask");
        RegisterTable(m_portTbl, "Port");
        RegisterTable(m_ccuTbl, "Ccu");
        RegisterTable(m_ccuResTbl, "CcuResource");
        RegisterTable(m_deviceConnTbl, "DeviceConnection");
        RegisterTable(m_endPointPairTbl, "EndPointPair");
        RegisterTable(m_ccuChannelTbl, "CcuChannel");
        RegisterTable(m_taskSchedulerDeviceTbl, "TaskSchedulerDevice");
        RegisterTable(m_computeDieTbl, "ComputeDie");
        RegisterTable(m_raSocketTbl, "RaSocket");
        RegisterTable(m_raSocketPairTbl, "RaSocketPair");
        RegisterTable(m_raDeviceTbl, "RaDevice");
        RegisterTable(m_raQPTbl, "RaQP");
        RegisterTable(m_raCQTbl, "RaCQ");
        RegisterTable(m_raCQETbl, "RaCQE");
        RegisterTable(m_raMRTbl, "RaMR");
        RegisterTable(m_memoryLayoutTbl, "MemoryLayout");
        RegisterTable(m_simModelDataTbl, "SimModelData");
        RegisterTable(m_rankTbl, "Rank");
        RegisterTable(m_ipcNotifyTbl, "IpcNotify");
        RegisterTable(m_ipcNotifyVistorListTbl, "IpcNotifyVistorList");
        RegisterTable(m_notifyRecordTaskTbl, "NotifyRecordTask");
        RegisterTable(m_notifyWaitTaskTbl, "NotifyWaitTask");
    }

    SimRunnerSqliteDB(const SimRunnerSqliteDB&) = delete;
    SimRunnerSqliteDB& operator=(const SimRunnerSqliteDB&) = delete;
    SimRunnerSqliteDB(SimRunnerSqliteDB&&) = delete;
    SimRunnerSqliteDB& operator=(SimRunnerSqliteDB&&) = delete;

    static SimRunnerSqliteDB& Instance() {
        static std::mutex s_mtx;
        static SimRunnerSqliteDB* s_instance = nullptr;

        std::lock_guard<std::mutex> lock(s_mtx);
        if (s_instance == nullptr) {
            s_instance = new SimRunnerSqliteDB();
        }
        return *s_instance;
    }

    template <typename T>
    sim::SqliteTable<T>& GetTable() {
        auto it = m_tableMap.find(std::type_index(typeid(T)));
        if (it == m_tableMap.end()) {
            throw std::invalid_argument("SimRunnerSqliteDB: Unsupported type - " + std::string(typeid(T).name()));
        }
        return *static_cast<sim::SqliteTable<T>*>(it->second);
    }

    template <typename T>
    std::optional<T> Find(TableKeyType<T> id) {
        return GetTable<T>().Find(id);
    }

    template <typename T>
    std::pair<TableValue<T>, bool> Query(std::function<bool(const T &)> pred) {
        return GetTable<T>().Query(pred);
    }

    template <typename T>
    std::vector<TableValue<T>> QueryList(std::function<bool(const T &)> pred) {
        return GetTable<T>().QueryList(pred);
    }

    template <typename T>
    TableKeyType<T> Add(T rec) {
        return GetTable<T>().Add(rec);
    }

    template <typename T>
    bool Update(TableKeyType<T> id, std::function<void(T &)> updater) {
        return GetTable<T>().Update(id, updater);
    }

    template <typename T>
    bool Delete(TableKeyType<T> id) {
        return GetTable<T>().Delete(id);
    }

    template <typename T>
    bool DeleteAll() {
        return GetTable<T>().DeleteAll();
    }

    std::vector<std::string> GetAllTableName() const {
        return m_tableNames;
    }

    void ClearAll() {
        m_db.ClearAllTables();
    }
};

#endif
