#include <string>
#include "cmd_base_utils.h"
#include "hccl_vm_log.h"
#include "cmd_base.h"

int main(int argc, char *argv[])
{
    LogConfig config;
    InitLogger(config);

    const char* envCheck = std::getenv(HVM_BASH_ENV_KEY.c_str());

    if (envCheck != nullptr) {
        StartHostClient(argc, argv);
    } else {
        std::string cmd = ArgvToString(argc, argv);
        if (argc == 1) {
            cmd += " --help";
        }
        ParseCommand(cmd);
    }
    return 0;
}