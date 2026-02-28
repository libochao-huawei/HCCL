#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <cstdint>

using u32 = uint32_t;

struct OxcLinkNode {
    std::string src_server_ip;
    u32 src_npu_index;
    std::string dst_server_ip;
    u32 dst_npu_index;
};

struct OxcCommKey {
    std::string task_id;
    std::string comm_type;
    u32 comm_id;

    bool operator<(const OxcCommKey& other) const {
        if (task_id != other.task_id) return task_id < other.task_id;
        if (comm_type != other.comm_type) return comm_type < other.comm_type;
        return comm_id < other.comm_id;
    }
};

struct OxcCommInfo {
    u32 pp_stage = 0;
    uint64_t data_size = 0;
    u32 npu_num = 0;
    u32 link_num_per_npu = 0;
};

struct OxcCommMatrixItem {
    OxcCommKey key;
    OxcCommInfo info;
    std::vector<OxcLinkNode> link_table;
};

class OxcGlobalRecorder {
public:
    static OxcGlobalRecorder& GetInstance();

    OxcGlobalRecorder(const OxcGlobalRecorder&) = delete;
    void operator=(const OxcGlobalRecorder&) = delete;

    void UpdateCommInfo(const OxcCommKey& key, const OxcCommInfo& info);
    void UpdateLinkTable(const OxcCommKey& key, const std::vector<OxcLinkNode>& links);
    void FlushToJSON(const std::string& filepath);

private:
    OxcGlobalRecorder() = default; 
    ~OxcGlobalRecorder() = default;

    std::map<OxcCommKey, OxcCommMatrixItem> data_map_; 
    std::mutex mutex_;
    bool is_flushed_ = false;
};