#include "oxcglobalrecorder.h"
#include <fstream>
#include <iostream>
#include </srv/workspace/br_dev_hccl_canneco_0128_5117041116700491777/work_code/hccl/output/third_party/json/single_include/nlohmann/json.hpp>

using json = nlohmann::json;
using u32 = uint32_t;

void to_json(json& j, const OxcLinkNode& p) {
    j = json{
        {"src_server_ip", p.src_server_ip},
        {"src_npu_index", p.src_npu_index},
        {"dst_server_ip", p.dst_server_ip},
        {"dst_npu_index", p.dst_npu_index}
    };
}

void to_json(json& j, const OxcCommKey& p) {
    j = json{
        {"task_id", p.task_id},
        {"comm_type", p.comm_type},
        {"comm_id", p.comm_id}
    };
}

void to_json(json& j, const OxcCommInfo& p) {
    j = json{
        {"pp_stage", p.pp_stage},
        {"data_size", p.data_size},
        {"npu_num", p.npu_num},
        {"link_num_per_npu", p.link_num_per_npu}
    };
}

void to_json(json& j, const OxcCommMatrixItem& p) {
    j = json{
        {"comm_domain_key", p.key},
        {"comm_domain_info", p.info},
        {"link_table", p.link_table}
    };
}

OxcGlobalRecorder& OxcGlobalRecorder::GetInstance() {
    static OxcGlobalRecorder instance;
    return instance;
}

void OxcGlobalRecorder::UpdateCommInfo(const OxcCommKey& key, const OxcCommInfo& info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_flushed_) return;

    OxcCommMatrixItem& item = data_map_[key]; 
    item.key = key; 
    item.info.data_size += info.data_size;
    item.info.pp_stage = info.pp_stage;
    item.info.npu_num = info.npu_num;
    item.info.link_num_per_npu = info.link_num_per_npu;
}

void OxcGlobalRecorder::UpdateLinkTable(const OxcCommKey& key, const std::vector<OxcLinkNode>& links) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_flushed_) return;

    OxcCommMatrixItem& item = data_map_[key];
    item.key = key;
    item.link_table = links;
}

void OxcGlobalRecorder::FlushToJSON(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (is_flushed_ || data_map_.empty()) return;

    try {
        json root;
        root["version"] = "1.0";
        root["version_time"] = "2026.02.06";
        root["version_tag"] = "coding";

        std::vector<OxcCommMatrixItem> output_list;
        output_list.reserve(data_map_.size());
        
        for (const auto& kv : data_map_) {
            output_list.push_back(kv.second);
        }

        root["comm_matrix"] = output_list;

        std::ofstream o(filepath);
        if (o.is_open()) {
            o << root.dump(4);
            o.close();
            std::cout << "[OxcGlobalRecorder] Written to " << filepath << std::endl;
        }

        is_flushed_ = true;
        data_map_.clear();

    } catch (const std::exception& e) {
        std::cerr << "[OxcGlobalRecorder] Error: " << e.what() << std::endl;
    }
}