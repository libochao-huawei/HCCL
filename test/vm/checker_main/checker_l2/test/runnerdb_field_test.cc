#include <iostream>
#include <vector>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

#include "sim_models.h"
#include "sim_runner_db.h"
#include "sim_sqlite_db.h"

using namespace sim;

void TestServerFields() {
    std::string dbPath = "/tmp/runnerdb_field_test.db";
    
    // 清理旧数据
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "=== Server 字段读写测试 ===" << std::endl;
        std::cout << "sizeof(Server): " << sizeof(Server) << " bytes" << std::endl;
        std::cout << std::endl;
        
        // 写入多个 Server 记录
        std::cout << "写入 5 条 Server 记录..." << std::endl;
        for (int i = 0; i < 5; i++) {
            Server data;
            // 注意：id 字段将由数据库自动生成
            data.pod_id = 100 + i;
            snprintf(data.version, sizeof(data.version), "v%d.0", i + 1);
            
            uint64_t insertedId = db.Add<Server>(data);
            std::cout << "  插入 Server: pod_id=" << data.pod_id 
                      << ", version=" << data.version 
                      << ", 生成 ID=" << insertedId << std::endl;
        }
        
        std::cout << std::endl << "查询所有 Server 记录..." << std::endl;
        
        // 查询所有记录
        auto results = db.QueryList<Server>([](const Server&) { return true; });
        
        std::cout << "找到 " << results.size() << " 条记录:" << std::endl;
        for (const auto& server : results) {
            std::cout << "  ID=" << server.id 
                      << ", pod_id=" << server.pod_id 
                      << ", version=" << server.version << std::endl;
        }
        
        // 验证字段完整性
        bool allValid = true;
        for (int i = 0; i < 5; i++) {
            // 通过 ID 查询
            auto found = db.Find<Server>(i + 1);
            if (!found) {
                std::cerr << "  ✗ ID=" << (i + 1) << " 未找到" << std::endl;
                allValid = false;
                continue;
            }
            
            // 验证字段值
            if (found.value().pod_id != 100 + i) {
                std::cerr << "  ✗ ID=" << (i + 1) << " pod_id 错误：期望 " << (100 + i) 
                          << ", 实际 " << found.value().pod_id << std::endl;
                allValid = false;
            }
            
            char expectedVersion[16];
            snprintf(expectedVersion, sizeof(expectedVersion), "v%d.0", i + 1);
            if (strcmp(found.value().version, expectedVersion) != 0) {
                std::cerr << "  ✗ ID=" << (i + 1) << " version 错误：期望 " << expectedVersion 
                          << ", 实际 " << found.value().version << std::endl;
                allValid = false;
            }
            
            if (found.value().pod_id == 100 + i && strcmp(found.value().version, expectedVersion) == 0) {
                std::cout << "  ✓ ID=" << (i + 1) << " 验证通过" << std::endl;
            }
        }
        
        std::cout << std::endl;
        if (allValid && results.size() == 5) {
            std::cout << "✓ Server 字段读写测试通过" << std::endl;
        } else {
            std::cout << "✗ Server 字段读写测试失败" << std::endl;
        }
        
        // 清理
        db.ClearAll();
    } catch (const std::exception& e) {
        std::cerr << "测试异常：" << e.what() << std::endl;
    }
    
    // 清理数据库文件
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
}

void TestHostFields() {
    std::string dbPath = "/tmp/runnerdb_host_field_test.db";
    
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "\n=== Host 字段读写测试 ===" << std::endl;
        std::cout << "sizeof(Host): " << sizeof(Host) << " bytes" << std::endl;
        std::cout << std::endl;
        
        // 写入 Host 记录
        std::cout << "写入 3 条 Host 记录..." << std::endl;
        for (int i = 0; i < 3; i++) {
            Host data;
            data.server_id = 200 + i;
            snprintf(data.ip_addr, sizeof(data.ip_addr), "192.168.1.%d", 10 + i);
            data.arch = static_cast<uint8_t>(i + 1);
            
            uint64_t insertedId = db.Add<Host>(data);
            std::cout << "  插入 Host: server_id=" << data.server_id 
                      << ", ip_addr=" << data.ip_addr 
                      << ", arch=" << (int)data.arch
                      << ", 生成 ID=" << insertedId << std::endl;
        }
        
        std::cout << std::endl << "查询所有 Host 记录..." << std::endl;
        
        auto results = db.QueryList<Host>([](const Host&) { return true; });
        
        std::cout << "找到 " << results.size() << " 条记录:" << std::endl;
        for (const auto& host : results) {
            std::cout << "  ID=" << host.id 
                      << ", server_id=" << host.server_id 
                      << ", ip_addr=" << host.ip_addr
                      << ", arch=" << (int)host.arch << std::endl;
        }
        
        // 验证字段完整性
        bool allValid = true;
        for (int i = 0; i < 3; i++) {
            auto found = db.Find<Host>(i + 1);
            if (!found) {
                std::cerr << "  ✗ ID=" << (i + 1) << " 未找到" << std::endl;
                allValid = false;
                continue;
            }
            
            if (found.value().server_id != 200 + i) {
                std::cerr << "  ✗ ID=" << (i + 1) << " server_id 错误" << std::endl;
                allValid = false;
            }
            
            char expectedIp[40];
            snprintf(expectedIp, sizeof(expectedIp), "192.168.1.%d", 10 + i);
            if (strcmp(found.value().ip_addr, expectedIp) != 0) {
                std::cerr << "  ✗ ID=" << (i + 1) << " ip_addr 错误" << std::endl;
                allValid = false;
            }
            
            if (found.value().arch != static_cast<uint8_t>(i + 1)) {
                std::cerr << "  ✗ ID=" << (i + 1) << " arch 错误" << std::endl;
                allValid = false;
            }
            
            if (found.value().server_id == 200 + i && 
                strcmp(found.value().ip_addr, expectedIp) == 0 &&
                found.value().arch == static_cast<uint8_t>(i + 1)) {
                std::cout << "  ✓ ID=" << (i + 1) << " 验证通过" << std::endl;
            }
        }
        
        std::cout << std::endl;
        if (allValid && results.size() == 3) {
            std::cout << "✓ Host 字段读写测试通过" << std::endl;
        } else {
            std::cout << "✗ Host 字段读写测试失败" << std::endl;
        }
        
        db.ClearAll();
    } catch (const std::exception& e) {
        std::cerr << "测试异常：" << e.what() << std::endl;
    }
    
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
}

int main() {
    TestServerFields();
    TestHostFields();
    
    return 0;
}
