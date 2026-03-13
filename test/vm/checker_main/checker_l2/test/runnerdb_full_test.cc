#include <iostream>
#include <vector>
#include <cstring>
#include <functional>

#include "sim_models.h"
#include "sim_runner_db.h"
#include "sim_sqlite_db.h"

using namespace sim;

// 测试计数器
struct TestCounter {
    int addCount = 0;
    int updateCount = 0;
    int deleteCount = 0;
    int findCount = 0;
    int queryListCount = 0;
    int deleteAllCount = 0;
};

TestCounter g_counter;

void TestServerAdd() {
    std::string dbPath = "/tmp/runnerdb_full_test.db";
    
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "=== Server Add 测试 ===" << std::endl;
        
        Server data;
        data.pod_id = 100;
        snprintf(data.version, sizeof(data.version), "v1.0");
        
        uint64_t id = db.Add<Server>(data);
        g_counter.addCount++;
        
        std::cout << "  插入 Server: pod_id=" << data.pod_id 
                  << ", version=" << data.version 
                  << ", 生成 ID=" << id << std::endl;
        
        if (id > 0 && data.pod_id == 100 && strcmp(data.version, "v1.0") == 0) {
            std::cout << "  ✓ Add 测试通过" << std::endl;
        } else {
            std::cout << "  ✗ Add 测试失败" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "  异常：" << e.what() << std::endl;
    }
}

void TestServerUpdate() {
    std::string dbPath = "/tmp/runnerdb_full_test.db";
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "\n=== Server Update 测试 ===" << std::endl;
        
        // 先查询
        auto server = db.Find<Server>(1);
        if (!server) {
            std::cout << "  未找到 ID=1 的记录" << std::endl;
            return;
        }
        
        std::cout << "  更新前：pod_id=" << server->pod_id 
                  << ", version=" << server->version << std::endl;
        
        // 更新
        bool updated = db.Update<Server>(1, [](Server& s) {
            s.pod_id = 200;
            snprintf(s.version, sizeof(s.version), "v2.0");
        });
        g_counter.updateCount++;
        
        if (updated) {
            auto updatedServer = db.Find<Server>(1);
            std::cout << "  更新后：pod_id=" << updatedServer->pod_id 
                      << ", version=" << updatedServer->version << std::endl;
            
            if (updatedServer->pod_id == 200 && strcmp(updatedServer->version, "v2.0") == 0) {
                std::cout << "  ✓ Update 测试通过" << std::endl;
            } else {
                std::cout << "  ✗ Update 测试失败" << std::endl;
            }
        } else {
            std::cout << "  ✗ Update 失败" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "  异常：" << e.what() << std::endl;
    }
}

void TestServerFind() {
    std::string dbPath = "/tmp/runnerdb_full_test.db";
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "\n=== Server Find 测试 ===" << std::endl;
        
        // 查找存在的记录
        auto server = db.Find<Server>(1);
        g_counter.findCount++;
        
        if (server) {
            std::cout << "  找到 ID=1: pod_id=" << server->pod_id 
                      << ", version=" << server->version << std::endl;
            std::cout << "  ✓ Find(存在) 测试通过" << std::endl;
        } else {
            std::cout << "  ✗ Find(存在) 测试失败" << std::endl;
        }
        
        // 查找不存在的记录
        auto notFound = db.Find<Server>(999);
        g_counter.findCount++;
        
        if (!notFound) {
            std::cout << "  未找到 ID=999 (预期)" << std::endl;
            std::cout << "  ✓ Find(不存在) 测试通过" << std::endl;
        } else {
            std::cout << "  ✗ Find(不存在) 测试失败" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "  异常：" << e.what() << std::endl;
    }
}

void TestServerQueryList() {
    std::string dbPath = "/tmp/runnerdb_full_test.db";
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "\n=== Server QueryList 测试 ===" << std::endl;
        
        // 查询所有
        auto allServers = db.QueryList<Server>([](const Server&) { return true; });
        g_counter.queryListCount++;
        
        std::cout << "  查询所有：找到 " << allServers.size() << " 条记录" << std::endl;
        
        // 条件查询：pod_id > 150
        auto filtered = db.QueryList<Server>([](const Server& s) {
            return s.pod_id > 150;
        });
        g_counter.queryListCount++;
        
        std::cout << "  条件查询 (pod_id > 150): 找到 " << filtered.size() << " 条记录" << std::endl;
        for (const auto& s : filtered) {
            std::cout << "    ID=" << s.id << ", pod_id=" << s.pod_id << std::endl;
        }
        
        if (allServers.size() > 0 && filtered.size() >= 0) {
            std::cout << "  ✓ QueryList 测试通过" << std::endl;
        } else {
            std::cout << "  ✗ QueryList 测试失败" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "  异常：" << e.what() << std::endl;
    }
}

void TestServerDelete() {
    std::string dbPath = "/tmp/runnerdb_full_test.db";
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "\n=== Server Delete 测试 ===" << std::endl;
        
        // 先添加一条记录
        Server data;
        data.pod_id = 999;
        snprintf(data.version, sizeof(data.version), "v9.9");
        uint64_t newId = db.Add<Server>(data);
        g_counter.addCount++;
        
        std::cout << "  插入测试记录 ID=" << newId << std::endl;
        
        // 删除
        bool deleted = db.Delete<Server>(newId);
        g_counter.deleteCount++;
        
        if (deleted) {
            auto check = db.Find<Server>(newId);
            if (!check) {
                std::cout << "  ✓ Delete 测试通过 (记录已删除)" << std::endl;
            } else {
                std::cout << "  ✗ Delete 测试失败 (记录仍存在)" << std::endl;
            }
        } else {
            std::cout << "  ✗ Delete 操作失败" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "  异常：" << e.what() << std::endl;
    }
}

void TestServerDeleteAll() {
    std::string dbPath = "/tmp/runnerdb_full_test.db";
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "\n=== Server DeleteAll 测试 ===" << std::endl;
        
        // 查询删除前数量
        auto before = db.QueryList<Server>([](const Server&) { return true; });
        std::cout << "  删除前：" << before.size() << " 条记录" << std::endl;
        
        // 删除所有
        bool deleted = db.DeleteAll<Server>();
        g_counter.deleteAllCount++;
        
        // 查询删除后数量
        auto after = db.QueryList<Server>([](const Server&) { return true; });
        std::cout << "  删除后：" << after.size() << " 条记录" << std::endl;
        
        if (deleted && after.size() == 0) {
            std::cout << "  ✓ DeleteAll 测试通过" << std::endl;
        } else {
            std::cout << "  ✗ DeleteAll 测试失败" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "  异常：" << e.what() << std::endl;
    }
}

void TestHostOperations() {
    std::string dbPath = "/tmp/runnerdb_full_test.db";
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "\n=== Host 全量操作测试 ===" << std::endl;
        
        // Add
        Host host;
        host.server_id = 1000;
        snprintf(host.ip_addr, sizeof(host.ip_addr), "10.0.0.1");
        host.arch = 1;
        
        uint64_t hostId = db.Add<Host>(host);
        g_counter.addCount++;
        
        std::cout << "  Add: ID=" << hostId << ", server_id=" << host.server_id 
                  << ", ip_addr=" << host.ip_addr << ", arch=" << (int)host.arch << std::endl;
        
        // Find
        auto found = db.Find<Host>(hostId);
        g_counter.findCount++;
        
        if (found && found->server_id == 1000) {
            std::cout << "  Find: ✓ 找到记录" << std::endl;
        } else {
            std::cout << "  Find: ✗ 未找到记录" << std::endl;
        }
        
        // Update
        bool updated = db.Update<Host>(hostId, [](Host& h) {
            h.server_id = 2000;
            snprintf(h.ip_addr, sizeof(h.ip_addr), "10.0.0.2");
            h.arch = 2;
        });
        g_counter.updateCount++;
        
        if (updated) {
            auto updatedHost = db.Find<Host>(hostId);
            std::cout << "  Update: server_id=" << updatedHost->server_id 
                      << ", ip_addr=" << updatedHost->ip_addr << std::endl;
            std::cout << "  ✓ Update 成功" << std::endl;
        }
        
        // QueryList
        auto hosts = db.QueryList<Host>([](const Host&) { return true; });
        g_counter.queryListCount++;
        
        std::cout << "  QueryList: 找到 " << hosts.size() << " 条记录" << std::endl;
        
        // Delete
        bool deleted = db.Delete<Host>(hostId);
        g_counter.deleteCount++;
        
        if (deleted) {
            std::cout << "  ✓ Delete 成功" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "  异常：" << e.what() << std::endl;
    }
}

void PrintSummary() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "测试统计:" << std::endl;
    std::cout << "  Add:      " << g_counter.addCount << " 次" << std::endl;
    std::cout << "  Update:   " << g_counter.updateCount << " 次" << std::endl;
    std::cout << "  Delete:   " << g_counter.deleteCount << " 次" << std::endl;
    std::cout << "  Find:     " << g_counter.findCount << " 次" << std::endl;
    std::cout << "  QueryList: " << g_counter.queryListCount << " 次" << std::endl;
    std::cout << "  DeleteAll: " << g_counter.deleteAllCount << " 次" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main() {
    std::cout << "=== RunnerDB 全量接口测试 ===" << std::endl;
    std::cout << "数据库路径：/tmp/runnerdb_full_test.db" << std::endl;
    std::cout << std::endl;
    
    TestServerAdd();
    TestServerUpdate();
    TestServerFind();
    TestServerQueryList();
    TestServerDelete();
    TestServerDeleteAll();
    TestHostOperations();
    
    PrintSummary();
    
    // 清理
    std::remove("/tmp/runnerdb_full_test.db");
    std::remove("/tmp/runnerdb_full_test.db-wal");
    std::remove("/tmp/runnerdb_full_test.db-shm");
    
    return 0;
}
