#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <sqlite3.h>
#include <unistd.h>
#include <sys/wait.h>

void WorkerProcess(int processId, int iterations, const std::string& dbPath) {
    sqlite3* db;
    char* errMsg = nullptr;
    
    if (sqlite3_open_v2(dbPath.c_str(), &db, 
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI, nullptr) != SQLITE_OK) {
        std::cerr << "Process " << processId << ": Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        _exit(1);
    }
    
    // 设置 WAL 模式和超时
    sqlite3_exec(db, "PRAGMA journal_mode = WAL", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "PRAGMA busy_timeout = 5000", nullptr, nullptr, nullptr);
    
    // 创建表
    const char* createTable = "CREATE TABLE IF NOT EXISTS test_data ("
                              "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                              "process_id INTEGER, "
                              "data TEXT);";
    if (sqlite3_exec(db, createTable, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Process " << processId << ": Failed to create table: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        sqlite3_close(db);
        _exit(1);
    }
    
    // 准备插入语句
    sqlite3_stmt* insertStmt;
    const char* insertSql = "INSERT INTO test_data (process_id, data) VALUES (?, ?);";
    if (sqlite3_prepare_v2(db, insertSql, -1, &insertStmt, nullptr) != SQLITE_OK) {
        std::cerr << "Process " << processId << ": Failed to prepare insert: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        _exit(1);
    }
    
    for (int i = 0; i < iterations; i++) {
        sqlite3_bind_int(insertStmt, 1, processId);
        sqlite3_bind_text(insertStmt, 2, ("data_" + std::to_string(i)).c_str(), -1, SQLITE_STATIC);
        
        int rc = sqlite3_step(insertStmt);
        if (rc != SQLITE_DONE) {
            std::cerr << "Process " << processId << ": Insert failed at " << i << ": " << sqlite3_errmsg(db) << std::endl;
        }
        
        sqlite3_reset(insertStmt);
        usleep(100); // 模拟延迟
    }
    
    sqlite3_finalize(insertStmt);
    sqlite3_close(db);
    _exit(0);
}

int main() {
    const int numProcesses = 4;
    const int iterations = 100;
    std::string dbPath = "/tmp/hccl_sim_test.db";
    
    std::cout << "=== SQLite 多进程并发访问测试 (WAL 模式) ===" << std::endl;
    std::cout << "进程数：" << numProcesses << std::endl;
    std::cout << "每个进程写入次数：" << iterations << std::endl;
    std::cout << "数据库路径：" << dbPath << std::endl;
    std::cout << std::endl;
    
    // 清空旧数据库
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    std::vector<pid_t> pids;
    std::atomic<int> successCount{0};
    
    // 创建子进程
    for (int i = 0; i < numProcesses; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            WorkerProcess(i, iterations, dbPath);
        } else if (pid > 0) {
            pids.push_back(pid);
        } else {
            std::cerr << "Fork failed!" << std::endl;
            return 1;
        }
    }
    
    // 等待所有子进程完成
    std::cout << "等待子进程完成..." << std::endl;
    for (pid_t pid : pids) {
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            successCount++;
        }
    }
    
    // 验证结果
    std::cout << std::endl;
    std::cout << "=== 测试结果 ===" << std::endl;
    std::cout << "成功完成的进程数：" << successCount << "/" << numProcesses << std::endl;
    
    // 查询数据库
    sqlite3* db;
    if (sqlite3_open_v2(dbPath.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK) {
        sqlite3_stmt* stmt;
        const char* selectSql = "SELECT COUNT(*) FROM test_data;";
        
        if (sqlite3_prepare_v2(db, selectSql, -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                int count = sqlite3_column_int(stmt, 0);
                std::cout << "数据库总记录数：" << count << std::endl;
                std::cout << "预期记录数：" << (numProcesses * iterations) << std::endl;
                
                if (count == numProcesses * iterations) {
                    std::cout << "✓ 测试通过：所有记录都已成功写入" << std::endl;
                } else {
                    std::cout << "✗ 测试失败：记录数不匹配" << std::endl;
                }
            }
            sqlite3_finalize(stmt);
        }
        
        sqlite3_close(db);
    }
    
    // 清理
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    return 0;
}
