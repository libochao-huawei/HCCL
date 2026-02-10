#ifndef HCCL_VM_SUBCMD_H
#define HCCL_VM_SUBCMD_H

#include <store/hccl_sim_shm_manager.h>
#include <store/hccl_shm_pub.h>

#include "hccl_common_defs.h"
#include "hccl_plugin_manager.h"

#include <cstring>
#include <string>
#include <unistd.h>
#include <stdlib.h>
#include <libgen.h>
#include <fstream>
#include <algorithm>
#include <exception>

using namespace HcclSim;

class CstyleCmd {
public:
    explicit CstyleCmd(const std::vector<std::string>& args) {
        storage_ = args;
        rebuild();
    }
    int argc() const {
        return static_cast<int>(pointers_.size());
    }
    char** argv() {
        return pointers_.data();
    }
    std::string cmd() {
        return cmd_;
    }
private:
    void rebuild() {
        std::stringstream ss;
        pointers_.clear();
        pointers_.reserve(storage_.size() + 1);
        for (auto& s : storage_) {
            pointers_.push_back(const_cast<char*>(s.c_str()));
            ss << s << " "; 
        }
        cmd_ = ss.str();
    }
    std::vector<std::string> storage_;
    std::vector<char*> pointers_;
    std::string cmd_;
};
// 通用工具方法
std::string GetBinLocation();
std::string ArgvToString(int argc, char *argv[]);
std::string FileInModelDir(const std::string& fileName);
bool ParseYamlTopo(std::string& fileName, TopoMeta& topo);
void RemoveFromLDPreload(const std::string& targetValue);
void ShowModel();

// init hvm env
HcclVmResult HvmInitSHM(TopoMeta topoMeta);
HcclVmResult HvmInitIPC(TopoMeta topoMeta);
HcclVmResult InitHvmEnv(TopoMeta topoMeta, uint32_t level);
HcclVmResult HcclVmExit();

// plugin manager
HcclVmResult InstallUserPlugin(std::string argStr);
HcclVmResult RunUserPlugin(std::string argStr);
HcclVmResult UninstallUserPlugin(std::string argStr);
HcclVmResult ShowUserPlugin();

#endif