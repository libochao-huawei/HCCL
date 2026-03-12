#ifndef SIM_SQLITE_TABLE_H
#define SIM_SQLITE_TABLE_H

#include <sqlite3.h>
#include <optional>
#include <string>
#include <functional>
#include <optional>
#include <vector>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <unordered_map>
#include <typeindex>
#include <cstring>
#include <memory>

namespace sim {

class TableBase {
public:
    virtual ~TableBase() = default;
    virtual std::string GetTableName() const = 0;
};

template <typename T>
class SqliteTable : public TableBase {
public:
    using KeyType = uint64_t;
    using ValueType = T;

    SqliteTable(sqlite3* db, const std::string& tableName)
        : m_db(db), m_tableName(tableName) {
        CreateTableIfNotExists();
    }

    ~SqliteTable() override = default;

    SqliteTable(const SqliteTable&) = delete;
    SqliteTable& operator=(const SqliteTable&) = delete;

    KeyType Add(T& rec) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        char sql[512];
        snprintf(sql, sizeof(sql), "INSERT INTO %s (data) VALUES (?)",
            m_tableName.c_str());
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("SQL prepare failed: ") + sqlite3_errmsg(m_db));
        }
        
        sqlite3_bind_blob(stmt, 1, &rec, sizeof(T), SQLITE_STATIC);
        
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            throw std::runtime_error(std::string("SQL insert failed: ") + sqlite3_errmsg(m_db));
        }
        
        // 获取自增 ID，保证多进程环境下的唯一性
        rec.id = static_cast<KeyType>(sqlite3_last_insert_rowid(m_db));
        
        sqlite3_finalize(stmt);
        return rec.id;
    }

    bool Update(KeyType id, std::function<void(T&)> updater) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        T rec;
        if (!FindInternal(id, rec)) {
            return false;
        }
        
        updater(rec);
        
        char sql[512];
        snprintf(sql, sizeof(sql),
            "UPDATE %s SET data = ? WHERE id = ?",
            m_tableName.c_str());
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("SQL prepare failed: ") + sqlite3_errmsg(m_db));
        }
        
        sqlite3_bind_blob(stmt, 1, &rec, sizeof(T), SQLITE_STATIC);
        sqlite3_bind_int64(stmt, 2, id);
        
        int result = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        
        return result == SQLITE_DONE;
    }

    bool Delete(KeyType id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        char sql[512];
        snprintf(sql, sizeof(sql),
            "DELETE FROM %s WHERE id = ?",
            m_tableName.c_str());
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("SQL prepare failed: ") + sqlite3_errmsg(m_db));
        }
        
        sqlite3_bind_int64(stmt, 1, id);
        int result = sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        
        return result == SQLITE_DONE;
    }

    bool DeleteAll() {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        char sql[512];
        snprintf(sql, sizeof(sql),
            "DELETE FROM %s",
            m_tableName.c_str());
        
        if (sqlite3_exec(m_db, sql, nullptr, nullptr, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("SQL delete all failed: ") + sqlite3_errmsg(m_db));
        }
        
        // 重置 AUTOINCREMENT 计数器
        snprintf(sql, sizeof(sql), "DELETE FROM sqlite_sequence WHERE name='%s'",
            m_tableName.c_str());
        sqlite3_exec(m_db, sql, nullptr, nullptr, nullptr);
        
        return true;
    }

    std::optional<ValueType> Find(KeyType id) const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_mutex));
        
        T rec;
        if (FindInternal(id, rec)) {
            return rec;
        }
        return std::nullopt;
    }

    std::vector<ValueType> QueryList(std::function<bool(const T&)> pred) const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_mutex));
        
        std::vector<ValueType> result;
        
        char sql[512];
        snprintf(sql, sizeof(sql),
            "SELECT id, data FROM %s",
            m_tableName.c_str());
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("SQL prepare failed: ") + sqlite3_errmsg(m_db));
        }
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            T rec;
            const void* blob = sqlite3_column_blob(stmt, 1);
            int blobSize = sqlite3_column_bytes(stmt, 1);
            
            if (blobSize == static_cast<int>(sizeof(T))) {
                std::memcpy(&rec, blob, sizeof(T));
                // 从数据库读取真实的 id，覆盖 BLOB 中的 id
                rec.id = sqlite3_column_int64(stmt, 0);
                if (pred(rec)) {
                    result.push_back(rec);
                }
            }
        }
        
        sqlite3_finalize(stmt);
        return result;
    }

    std::pair<ValueType, bool> Query(std::function<bool(const T&)> pred) const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(m_mutex));
        
        char sql[512];
        snprintf(sql, sizeof(sql),
            "SELECT id, data FROM %s",
            m_tableName.c_str());
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("SQL prepare failed: ") + sqlite3_errmsg(m_db));
        }
        
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            T rec;
            const void* blob = sqlite3_column_blob(stmt, 1);
            int blobSize = sqlite3_column_bytes(stmt, 1);
            
            if (blobSize == static_cast<int>(sizeof(T))) {
                std::memcpy(&rec, blob, sizeof(T));
                // 从数据库读取真实的 id，覆盖 BLOB 中的 id
                rec.id = sqlite3_column_int64(stmt, 0);
                if (pred(rec)) {
                    sqlite3_finalize(stmt);
                    return {rec, true};
                }
            }
        }
        
        sqlite3_finalize(stmt);
        return {ValueType{}, false};
    }

    std::string GetTableName() const override {
        return m_tableName;
    }

private:
    void CreateTableIfNotExists() {
        char sql[512];
        // 使用 AUTOINCREMENT 确保多进程环境下 ID 全局唯一且单调递增
        snprintf(sql, sizeof(sql),
            "CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY AUTOINCREMENT, data BLOB NOT NULL)",
            m_tableName.c_str());
        
        if (sqlite3_exec(m_db, sql, nullptr, nullptr, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("SQL create table failed: ") + sqlite3_errmsg(m_db));
        }
    }

    bool FindInternal(KeyType id, T& rec) const {
        char sql[512];
        snprintf(sql, sizeof(sql),
            "SELECT id, data FROM %s WHERE id = ?",
            m_tableName.c_str());
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("SQL prepare failed: ") + sqlite3_errmsg(m_db));
        }
        
        sqlite3_bind_int64(stmt, 1, id);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const void* blob = sqlite3_column_blob(stmt, 1);
            int blobSize = sqlite3_column_bytes(stmt, 1);
            
            if (blobSize == static_cast<int>(sizeof(T))) {
                std::memcpy(&rec, blob, sizeof(T));
                // 从数据库读取真实的 id，覆盖 BLOB 中的 id
                rec.id = sqlite3_column_int64(stmt, 0);
                sqlite3_finalize(stmt);
                return true;
            }
        }
        
        sqlite3_finalize(stmt);
        return false;
    }

    sqlite3* m_db;
    std::string m_tableName;
    mutable std::mutex m_mutex;
};

class SqliteDatabase {
public:
    static SqliteDatabase& Instance() {
        static std::mutex s_mtx;
        static SqliteDatabase* s_instance = nullptr;
        
        std::lock_guard<std::mutex> lock(s_mtx);
        if (s_instance == nullptr) {
            s_instance = new SqliteDatabase();
        }
        return *s_instance;
    }

    static void SetDbPath(const std::string& path) {
        s_dbPath = path;
    }

    SqliteDatabase(const SqliteDatabase&) = delete;
    SqliteDatabase& operator=(const SqliteDatabase&) = delete;

    template <typename T>
    SqliteTable<T>& GetTable(const std::string& tableName) {
        std::lock_guard<std::mutex> lock(m_tablesMutex);
        
        auto it = m_tables.find(tableName);
        if (it == m_tables.end()) {
            auto table = std::make_unique<SqliteTable<T>>(m_db, tableName);
            auto* rawPtr = table.get();
            m_tables[tableName] = std::move(table);
            m_tableTypes[tableName] = typeid(T).name();
            return *rawPtr;
        }
        
        return *static_cast<SqliteTable<T>*>(m_tables[tableName].get());
    }

    void Close() {
        std::lock_guard<std::mutex> lock(m_tablesMutex);
        
        m_tables.clear();
        
        if (m_db) {
            sqlite3_close(m_db);
            m_db = nullptr;
        }
    }

    sqlite3* GetDb() {
        return m_db;
    }

    void ClearAllTables() {
        std::lock_guard<std::mutex> lock(m_tablesMutex);
        
        for (auto& pair : m_tables) {
            char sql[512];
            snprintf(sql, sizeof(sql), "DELETE FROM %s", pair.first.c_str());
            sqlite3_exec(m_db, sql, nullptr, nullptr, nullptr);
        }
    }

    static void RemoveDbFile() {
        if (!s_dbPath.empty() && s_dbPath != ":memory:") {
            std::remove(s_dbPath.c_str());
            std::string walPath = s_dbPath + "-wal";
            std::remove(walPath.c_str());
            std::string shmPath = s_dbPath + "-shm";
            std::remove(shmPath.c_str());
        }
    }

public:
    static std::string s_dbPath;

SqliteDatabase() : m_db(nullptr) {
        std::string actualPath = s_dbPath.empty() ? "/tmp/hccl_sim.db" : s_dbPath;
        
        if (sqlite3_open_v2(actualPath.c_str(), &m_db, 
                            SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI,
                            nullptr) != SQLITE_OK) {
            throw std::runtime_error(std::string("Failed to open SQLite database: ") + sqlite3_errmsg(m_db));
        }
        
        // 启用 WAL 模式以支持多进程并发
        sqlite3_exec(m_db, "PRAGMA journal_mode = WAL", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA synchronous = NORMAL", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA cache_size = -10000", nullptr, nullptr, nullptr);
        sqlite3_exec(m_db, "PRAGMA temp_store = MEMORY", nullptr, nullptr, nullptr);
        // 设置锁模式为 NORMAL，允许其他进程访问
        sqlite3_exec(m_db, "PRAGMA locking_mode = NORMAL", nullptr, nullptr, nullptr);
        // 设置忙超时，避免立即失败
        sqlite3_exec(m_db, "PRAGMA busy_timeout = 5000", nullptr, nullptr, nullptr);
    }

    ~SqliteDatabase() {
        Close();
    }

    sqlite3* m_db;
    mutable std::mutex m_tablesMutex;
    std::unordered_map<std::string, std::unique_ptr<TableBase>> m_tables;
    std::unordered_map<std::string, std::string> m_tableTypes;
};

}

#endif
