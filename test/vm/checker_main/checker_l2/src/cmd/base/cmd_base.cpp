#include "cmd_base.h"

namespace HcclSim {
auto& CommandRegistry::GetCreators() {
    static std::vector<std::pair<std::string, CommandCreator>> creators;
    return creators;
}

void CommandRegistry::RegisteCommand(const std::string& name, CommandCreator creator) {
    GetCreators().emplace_back(name, std::move(creator));
}

std::vector<std::unique_ptr<CommandBase>> CommandRegistry::CreateAll() {
    std::vector<std::unique_ptr<CommandBase>> commands;
    commands.reserve(GetCreators().size());
    for (const auto& [name, creator] : GetCreators()) {
        commands.push_back(creator());
    }
    return commands;
}

}
