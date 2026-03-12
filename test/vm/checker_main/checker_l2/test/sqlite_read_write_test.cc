#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <sqlite3.h>
#include <unistd.h>
#include <sys/wait.h>

// 写进程
void WriterProcess(int processId, int iterations, const std::string& dbPath) {
    sqlite3* db;
    char* errMsg = nullptr;
    
    if (sqlite3_open_v2(dbPath.c_str(), &db, 
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI, nullptr) != SQLITE_OK) {
        std::cerr << "Writer " << processId << ": Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        _exit(1);
    }
    
    sqlite3_exec(db, "PRAGMA journal_mode = WAL", nullptr, nullptr, nullptr);
    sqlite3_exec(db, "PRAGMA busy_timeout = 5000", nullptr, nullptr, nullptr);
    
    const char* createTable = "CREATE TABLE IF NOT EXISTS test_data ("
                              "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                              "process_id INTEGER, "
                              "data TEXT);";
    sqlite3_exec(db, createTable, nullptr, nullptr, &errMsg);
    
    sqlite3_stmt* insertStmt;
    const char* insertSql = "INSERT INTO test_data (process_id, data) VALUES (?, ?);";
    sqlite3_prepare_v2(db, insertSql, -1, &insertStmt, nullptr);
    
    for (int i = 0; i < iterations; i++) {
        sqlite3_bind_int(insertStmt, 1, processId);
        sqlite3_bind_text(insertStmt, 2, ("write_data_" + std::to_string(i)).c_str(), -1, SQLITE_STATIC);
        sqlite3_step(insertStmt);
        sqlite3_reset(insertStmt);
        usleep(500);
    }
    
    sqlite3_finalize(insertStmt);
    sqlite3_close(db);
    _exit(0);
}

// 读进程
void ReaderProcess(int processId, int iterations, const std::string& dbPath) {
    sqlite3* db;
    
    if (sqlite3_open_v2(dbPath.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::cerr << "Reader " << processId << ": Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        _exit(1);
    }
    
    sqlite3_exec(db, "PRAGMA busy_timeout = 5000", nullptr, nullptr, nullptr);
    
    sqlite3_stmt* selectStmt;
    const char* selectSql = "SELECT COUNT(*) FROM test_data;";
    sqlite3_prepare_v2(db, selectSql, -1, &selectStmt, nullptr);
    
    for (int i = 0; i < iterations; i++) {
        sqlite3_reset(selectStmt);
        if (sqlite3_step(selectStmt) == SQLITE_ROW) {
            int count = sqlite3_column_int(selectStmt, 0);
            if (count > 0) {
                // 读取一些数据
                sqlite3_exec(db, "SELECT * FROM test_data LIMIT 10", nullptr, nullptr, nullptr);
            }
        }
        sqlite3_reset(selectStmt);
        usleep(300);
    }
    
    sqlite3_finalize(selectStmt);
    sqlite3_close(db);
    _exit(0);
}

int main() {
    const int numWriters = 2;
    const int numReaders = 2;
    const int iterations = 50;
    std::string dbPath = "/tmp/hccl_sim_rw_test.db";
    
    std::cout << "=== SQLite 读写并发测试 (WAL 模式) ===" << std::endl;
    std::cout << "写进程数：" << numWriters << std::endl;
    std::cout << "读进程数：" << numReaders << std::endl;
    std::cout << "每个进程操作次数：" << iterations << std::endl;
    std::cout << "数据库路径：" << dbPath << std::endl;
    std::cout << std::endl;
    
    // 清空旧数据库
    std::remove(dbPath.c_str());
    std::remove((dbPath + "-wal").c_str());
    std::remove((dbPath + "-shm").c_str());
    
    std::vector<pid_t> pids;
    std::atomic<int> successCount{0};
    
    // 创建写进程
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
    
    // 等待写进程开始
    usleep(100000);
    
    // 创建读进程
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
    
    // 等待所有进程完成
    std::cout << "等待所有进程完成..." << std::endl;
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
    std::cout << "成功完成的进程数：" << successCount << "/" << (numWriters + numReaders) << std::endl;
    
    sqlite3* db;
    if (sqlite3_open_v2(dbPath.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK) {
        sqlite3_stmt* stmt;
        const char* selectSql = "SELECT COUNT(*) FROM test_data;";
        
        if (sqlite3_prepare_v2(db, selectSql, -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                int count = sqlite3_column_int(stmt, 0);
                std::cout << "数据库总记录数：" << count << std::endl;
                std::cout << "预期记录数：" << (numWriters * iterations) << std::endl;
                
                if (count == numWriters * iterations) {
                    std::cout << "✓ 测试通过：读写并发正常" << std::endl;
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
