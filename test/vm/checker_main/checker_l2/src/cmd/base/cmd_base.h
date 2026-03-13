#ifndef HCCL_VM_COMMAND_BASE_H
#define HCCL_VM_COMMAND_BASE_H

#include <CLI11.hpp>
#include <functional>
#include <vector>
#include <string>
#include <memory>

namespace HcclSim {

class CommandBase {
public:
    virtual ~CommandBase() = default;
    virtual void Setup(CLI::App& app) = 0;
};

// 命令注册器
class CommandRegistry {
public:
    using CommandCreator = std::function<std::unique_ptr<CommandBase>()>;
    
    static void RegisteCommand(const std::string& name, CommandCreator creator);
    static std::vector<std::unique_ptr<CommandBase>> CreateAll();
    
private:
    static auto& GetCreators();
};

// 命令自动注册
template<typename CommandType>
class CommandAutoRegister {
public:
    CommandAutoRegister() {
        CommandRegistry::RegisteCommand(
            CommandType::StaticName(),
            []() -> std::unique_ptr<CommandBase> {
                return std::make_unique<CommandType>();
            }
        );
    }
};

}

#endif