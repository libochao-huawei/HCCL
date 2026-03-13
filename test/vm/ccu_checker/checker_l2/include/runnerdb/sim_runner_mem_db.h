#ifndef SIM_RUNNER_MEM_DB_H
#define SIM_RUNNER_MEM_DB_H
#include <tuple>
#include <type_traits>
#include <stdexcept>
#include <functional>
#include <optional>
#include <typeindex>
#include <unordered_map>
#include <shared_mutex>

#include "sim_models.h"
#include "sim_mem_table.h"

template <typename T>
using TableKeyType = typename Table<T>::KeyType;
template <typename T>
using TableValue = typename Table<T>::ValueType;

struct SimRunnerMemDB {
private:
    Table<sim::Server> serverTbl{"server"};
    Table<sim::Host> hostTbl{"host"};
    Table<sim::Runner> runnerTbl{"runner"};
    Table<sim::Device> deviceTbl{"device"};
    Table<sim::DeviceStatus> deviceStatusTbl{"deviceStatus"};
    Table<sim::Context> contextTbl{"context"};
    Table<sim::Stream> streamTbl{"stream"};
    Table<sim::Notify> notifyTbl{"notify"};
    Table<sim::Event> eventTbl{"event"};
    Table<sim::VirtualMemBlock> virMemTbl{"vir_mem"};
    Table<sim::PhyMemBlock> phyMemTbl{"phy_mem"};
    Table<sim::IpcMemRecord> ipcMemRecordTbl{"ipcMemRecord"};
    Table<sim::IpcMemWhiteList> ipcMemWhiteListTbl{"ipMemWhiteList"};
    Table<sim::FdMemRecord> fdMemRecordTbl{"fdMemRecord"};
    Table<sim::FdMemWhiteList> fdMemWhiteListTbl{"fdMemWhiteList"};
    Table<sim::Task> taskTbl{"task"};
    Table<sim::EventSyncTask> eventSyncTaskTbl{"eventSyncTask"};
    Table<sim::Port> portTbl{"port"};
    Table<sim::Ccu> ccuTbl{"ccu"};
    Table<sim::CcuResource> ccuResTbl{"CcuResource"};
    Table<sim::DeviceConnection> deviceConnTbl{"deviceConnection"};
    Table<sim::EndPointPair> endPointPairTbl{"endPointPair"};
    Table<sim::CcuChannel> ccuChannelTbl{"ccuChannel"};
    Table<sim::TaskSchedulerDevice> taskSchedulerDeviceTbl{"taskSchedulerDevice"};
    Table<sim::ComputeDie> computeDieTbl{"ComputeDie"};
    Table<sim::RaSocket> raSocketTbl{"raSocket"};
    Table<sim::RaSocketPair> raSocketPairTbl{"raSocketPair"};

    Table<sim::MemoryLayout> memoryLayoutTbl{"MemoryLayout"};
    Table<sim::SimModelData> simModelDataTbl{"SimModelData"};
    Table<sim::Rank> rank{"Rank"};
    Table<sim::IpcNotify> ipcNotifyTbl{"IpcNotify"};
    
    std::unordered_map<std::type_index, void *> tableMap;
    std::shared_mutex tableRWMutex;

    SimRunnerMemDB()
    {
        RegisterTable<sim::Server>(serverTbl);
        RegisterTable<sim::Host>(hostTbl);
        RegisterTable<sim::Runner>(runnerTbl);
        RegisterTable<sim::Device>(deviceTbl);
        RegisterTable<sim::DeviceStatus>(deviceStatusTbl);
        RegisterTable<sim::Context>(contextTbl);
        RegisterTable<sim::Stream>(streamTbl);
        RegisterTable<sim::Notify>(notifyTbl);
        RegisterTable<sim::Event>(eventTbl);
        RegisterTable<sim::VirtualMemBlock>(virMemTbl);
        RegisterTable<sim::PhyMemBlock>(phyMemTbl);
        RegisterTable<sim::IpcMemRecord>(ipcMemRecordTbl);
        RegisterTable<sim::IpcMemWhiteList>(ipcMemWhiteListTbl);
        RegisterTable<sim::FdMemRecord>(fdMemRecordTbl);
        RegisterTable<sim::FdMemWhiteList>(fdMemWhiteListTbl);
        RegisterTable<sim::Task>(taskTbl);
        RegisterTable<sim::EventSyncTask>(eventSyncTaskTbl);
        RegisterTable<sim::Port>(portTbl);
        RegisterTable<sim::Ccu>(ccuTbl);
        RegisterTable<sim::CcuResource>(ccuResTbl);
        RegisterTable<sim::DeviceConnection>(deviceConnTbl);
        RegisterTable<sim::EndPointPair>(endPointPairTbl);
        RegisterTable<sim::CcuChannel>(ccuChannelTbl);
        RegisterTable<sim::TaskSchedulerDevice>(taskSchedulerDeviceTbl);
        RegisterTable<sim::ComputeDie>(computeDieTbl);
        RegisterTable<sim::RaSocket>(raSocketTbl);
        RegisterTable<sim::RaSocketPair>(raSocketPairTbl);
        RegisterTable<sim::MemoryLayout>(memoryLayoutTbl);
        RegisterTable<sim::SimModelData>(simModelDataTbl);
        RegisterTable<sim::Rank>(rank);
        RegisterTable<sim::IpcNotify>(ipcNotifyTbl);
    }

    SimRunnerMemDB(const SimRunnerMemDB&) = delete;
    SimRunnerMemDB& operator=(const SimRunnerMemDB&) = delete;
    SimRunnerMemDB(SimRunnerMemDB&&) = delete;
    SimRunnerMemDB& operator=(SimRunnerMemDB&&) = delete;

    template <typename T>
    void RegisterTable(Table<T>& table) {
        tableMap[std::type_index(typeid(T))] = &table;
    }

public:

    static SimRunnerMemDB& Instance() {
        static SimRunnerMemDB instance;
        return instance;
    }

    template <typename T>
    std::string Dump()
    {
        // dump 指定表
        return std::string("");
    }

    template <typename T>
    Table<T> &GetTable()
    {
        std::shared_lock<std::shared_mutex> lock(tableRWMutex);
        auto it = tableMap.find(std::type_index(typeid(T)));
        if (it == tableMap.end()) {
            throw std::invalid_argument("SimRunnerMemDB: Unsupported type - " + std::string(typeid(T).name()));
        }
        return *static_cast<Table<T> *>(it->second);
    }

    //  查找ptr
    template <typename T>
    const TableValue<T>* FindPtr(TableKeyType<T> id)
    {
        return GetTable<T>().FindPtr(id);
    }

    // 查找 
    template <typename T>
    std::optional<T> Find(TableKeyType<T> id)
    {
        return GetTable<T>().Find(id);
    }

    //  条件查找
    template <typename T>
    std::pair<TableValue<T>, bool> Query(std::function<bool(const T &)> pred)
    {
        return GetTable<T>().Query(pred);
    }

    //  条件查找
    template <typename T>
    std::vector<TableValue<T>> QueryList(std::function<bool(const T &)> pred)
    {
        return GetTable<T>().QueryList(pred);
    }
    // 插入
    template <typename T>
    TableKeyType<T> Add(T rec)
    {
        return GetTable<T>().Add(rec);
    }

    // 更新
    template <typename T>
    bool Update(TableKeyType<T> id, std::function<void(T &)> updater)
    {
        return GetTable<T>().Update(id, updater);
    }

    // 删除
    template <typename T>
    bool Delete(TableKeyType<T> id)
    {
        return GetTable<T>().Delete(id);
    }

};
#endif
