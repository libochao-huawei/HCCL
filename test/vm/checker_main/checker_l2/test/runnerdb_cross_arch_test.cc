#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

#include "sim_models.h"
#include "sim_runner_db.h"
#include "sim_sqlite_db.h"

using namespace sim;

struct ServerPredicate {
    bool operator()(const Server&) const { return true; }
};

struct HostPredicate {
    bool operator()(const Host&) const { return true; }
};

void WriteServers(int processId, const std::string& dbPath) {
    SqliteDatabase::SetDbPath(dbPath);
    usleep(10000);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        for (int i = 0; i < 10; i++) {
            Server data;
            data.id = processId * 100 + i;
            data.pod_id = 1;
            snprintf(data.version, sizeof(data.version), "1.0");
            
            db.Add<Server>(data);
        }
        
        std::cout << "Process " << processId << " wrote Servers" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Process " << processId << " exception: " << e.what() << std::endl;
    }
    
    _exit(0);
}

void WriteHosts(int processId, const std::string& dbPath) {
    SqliteDatabase::SetDbPath(dbPath);
    usleep(10000);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        for (int i = 0; i < 10; i++) {
            Host data;
            data.id = processId * 100 + i;
            data.server_id = processId * 1000;
            snprintf(data.ip_addr, sizeof(data.ip_addr), "192.168.1.%d", i);
            data.arch = 1;
            
            db.Add<Host>(data);
        }
        
        std::cout << "Process " << processId << " wrote Hosts" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Process " << processId << " exception: " << e.what() << std::endl;
    }
    
    _exit(0);
}

void ReadAndVerify(const std::string& dbPath) {
    SqliteDatabase::SetDbPath(dbPath);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        std::cout << "\n=== 数据验证 ===" << std::endl;
        
        std::vector<Server> serverResults = db.QueryList<Server>(ServerPredicate());
        std::vector<Host> hostResults = db.QueryList<Host>(HostPredicate());
        
        int serverCount = (int)serverResults.size();
        int hostCount = (int)hostResults.size();
        
        std::cout << "Server 数量：" << serverCount << " (预期：20)" << std::endl;
        std::cout << "Host 数量：" << hostCount << " (预期：20)" << std::endl;
        
        bool allValid = true;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 10; j++) {
                uint64_t id = i * 100 + j;
                auto* found = db.FindPtr<Server>(id);
                if (found == nullptr || found->id != id) {
                    std::cerr << "Server 验证失败：id=" << id << std::endl;
                    allValid = false;
                }
            }
        }
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 10; j++) {
                uint64_t id = i * 100 + j;
                auto* found = db.FindPtr<Host>(id);
                if (found == nullptr || found->id != id) {
                    std::cerr << "Host 验证失败：id=" << id << std::endl;
                    allValid = false;
                }
            }
        }
        
        if (serverCount == 20 && hostCount == 20 && allValid) {
            std::cout << "✓ 数据验证通过" << std::endl;
        } else {
            std::cout << "✗ 数据验证失败" << std::endl;
        }
        
        db.ClearAll();
    } catch (const std::exception& e) {
        std::cerr << "验证异常：" << e.what() << std::endl;
    }
}

int main() {
    std::string dbPath = "/tmp/runnerdb_cross_arch_test.db";
    
    std::cout << "=== RunnerDB 跨架构数据结构测试 ===" << std::endl;
    std::cout << "sizeof(Server): " << sizeof(Server) << " bytes" << std::endl;
    std::cout << "sizeof(Host): " << sizeof(Host) << " bytes" << std::endl;
    std::cout << "数据库路径：" << dbPath << std::endl;
    std::cout << std::endl;
    
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    std::vector<pid_t> pids;
    
    pid_t pid0 = fork();
    if (pid0 == 0) {
        WriteServers(0, dbPath);
    } else if (pid0 > 0) {
        pids.push_back(pid0);
    } else {
        std::cerr << "Fork 0 failed!" << std::endl;
        return 1;
    }
    
    pid_t pid1 = fork();
    if (pid1 == 0) {
        WriteHosts(1, dbPath);
    } else if (pid1 > 0) {
        pids.push_back(pid1);
    } else {
        std::cerr << "Fork 1 failed!" << std::endl;
        return 1;
    }
    
    for (pid_t pid : pids) {
        int status;
        waitpid(pid, &status, 0);
    }
    
    ReadAndVerify(dbPath);
    
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    return 0;
}
