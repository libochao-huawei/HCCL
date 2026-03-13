#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <string>
#include <fstream>
#include <unistd.h>
#include <sys/wait.h>

// 简化的 SQLite 数据库连接结构
struct SimDb {
    std::string dbPath;
    std::mutex dbMutex;
    std::atomic<int> writeCount{0};
    std::atomic<int> readCount{0};
    
    SimDb(const std::string& path) : dbPath(path) {}
    
    void Write(int id, const std::string& data) {
        std::lock_guard<std::mutex> lock(dbMutex);
        std::ofstream file(dbPath, std::ios::app);
        if (file.is_open()) {
            file << id << "," << data << "\n";
            file.close();
            writeCount++;
        }
    }
    
    std::string Read(int id) {
        std::lock_guard<std::mutex> lock(dbMutex);
        std::ifstream file(dbPath);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (line.find(std::to_string(id)) != std::string::npos) {
                    readCount++;
                    return line;
                }
            }
        }
        return "";
    }
};

// 使用全局数据库
SimDb g_db("/tmp/hccl_sim_test.db");

void WorkerProcess(int processId, int iterations) {
    for (int i = 0; i < iterations; i++) {
        g_db.Write(processId * 1000 + i, "data_" + std::to_string(i));
        g_db.Read(processId * 1000 + i);
        usleep(100); // 模拟延迟
    }
}

int main() {
    const int numProcesses = 4;
    const int iterations = 100;
    
    std::cout << "=== 多进程数据库并发访问测试 ===" << std::endl;
    std::cout << "进程数：" << numProcesses << std::endl;
    std::cout << "每个进程写入次数：" << iterations << std::endl;
    std::cout << "数据库路径：" << g_db.dbPath << std::endl;
    std::cout << std::endl;
    
    // 清空旧数据
    std::remove(g_db.dbPath.c_str());
    
    std::vector<pid_t> pids;
    std::atomic<int> successCount{0};
    
    // 创建子进程
    for (int i = 0; i < numProcesses; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            // 子进程
            WorkerProcess(i, iterations);
            _exit(0);
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
    
    std::ifstream file(g_db.dbPath);
    if (file.is_open()) {
        int lineCount = 0;
        std::string line;
        while (std::getline(file, line)) {
            lineCount++;
        }
        file.close();
        
        std::cout << "数据库总记录数：" << lineCount << std::endl;
        std::cout << "预期记录数：" << (numProcesses * iterations) << std::endl;
        
        if (lineCount == numProcesses * iterations) {
            std::cout << "✓ 测试通过：所有记录都已写入" << std::endl;
        } else {
            std::cout << "✗ 测试失败：记录数不匹配" << std::endl;
        }
    }
    
    // 清理
    std::remove(g_db.dbPath.c_str());
    
    return 0;
}
