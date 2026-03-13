#include <iostream>
#include <vector>
#include <atomic>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>

#include "sim_models.h"
#include "sim_runner_db.h"
#include "sim_sqlite_db.h"

using namespace sim;

void WriterProcess(int processId, int iterations, const std::string& dbPath) {
    SqliteDatabase::SetDbPath(dbPath);
    usleep(10000);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        for (int i = 0; i < iterations; i++) {
            Server data;
            data.id = (uint64_t)processId * 100000 + i;
            data.pod_id = 1;
            snprintf(data.version, sizeof(data.version), "1.0");
            
            db.Add<Server>(data);
            
            usleep(100);
        }
        
        std::cout << "Writer Process " << processId << " completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Process " << processId << " exception: " << e.what() << std::endl;
    }
    
    _exit(0);
}

void ReaderProcess(int processId, int iterations, const std::string& dbPath) {
    SqliteDatabase::SetDbPath(dbPath);
    usleep(50000);
    
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        for (int i = 0; i < iterations; i++) {
            int targetId = (processId * 100 + i) % 200;
            auto* found = db.FindPtr<Server>(targetId);
            if (found != nullptr) {}
            usleep(200);
        }
        
        std::cout << "Reader Process " << processId << " completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Process " << processId << " exception: " << e.what() << std::endl;
    }
    
    _exit(0);
}

int main() {
    const int numWriters = 2;
    const int numReaders = 2;
    const int iterations = 50;
    std::string dbPath = "/tmp/runnerdb_concurrent_test.db";
    
    std::cout << "=== RunnerDB 多进程并发测试 ===" << std::endl;
    std::cout << "写进程数：" << numWriters << std::endl;
    std::cout << "读进程数：" << numReaders << std::endl;
    std::cout << "每个进程操作次数：" << iterations << std::endl;
    std::cout << "sizeof(Server): " << sizeof(Server) << " bytes" << std::endl;
    std::cout << std::endl;
    
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    std::vector<pid_t> pids;
    std::atomic<int> successCount{0};
    
    std::cout << "创建写进程..." << std::endl;
    for (int i = 0; i < numWriters; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            WriterProcess(i, iterations, dbPath);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "Fork failed!" << std::endl;
            return 1;
        }
    }
    
    usleep(50000);
    
    std::cout << "创建读进程..." << std::endl;
    for (int i = 0; i < numReaders; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            ReaderProcess(i, iterations, dbPath);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "Fork failed!" << std::endl;
            return 1;
        }
    }
    
    std::cout << "等待所有进程完成..." << std::endl;
    for (pid_t pid : pids) {
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            successCount++;
        }
    }
    
    std::cout << std::endl << "=== 最终验证 ===" << std::endl;
    std::cout << "成功完成的进程数：" << successCount << "/" << (numWriters + numReaders) << std::endl;
    
    SqliteDatabase::SetDbPath(dbPath);
    try {
        SimRunnerSqliteDB& db = SimRunnerSqliteDB::Instance();
        
        int expectedCount = numWriters * iterations;
        
        struct ServerPredicate {
            bool operator()(const Server&) const { return true; }
        };
        
        std::vector<Server> results = db.QueryList<Server>(ServerPredicate());
        int actualCount = (int)results.size();
        
        std::cout << "数据库中记录数：" << actualCount << std::endl;
        std::cout << "预期记录数：" << expectedCount << std::endl;
        
        bool allValid = true;
        for (int i = 0; i < numWriters; i++) {
            for (int j = 0; j < iterations; j++) {
                uint64_t id = (uint64_t)i * 100000 + j;
                auto* found = db.FindPtr<Server>(id);
                if (found == nullptr || found->id != id) {
                    std::cerr << "数据验证失败：id=" << id << std::endl;
                    allValid = false;
                }
            }
        }
        
        if (allValid && actualCount == expectedCount) {
            std::cout << "✓ RunnerDB 多进程并发测试通过" << std::endl;
        } else {
            std::cout << "✗ RunnerDB 多进程并发测试失败" << std::endl;
        }
        
        db.ClearAll();
    } catch (const std::exception& e) {
        std::cerr << "验证异常：" << e.what() << std::endl;
    }
    
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    return 0;
}
