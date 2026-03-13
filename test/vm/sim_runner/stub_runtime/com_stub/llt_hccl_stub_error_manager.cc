#include <iostream>
#include <string>
#include "error_manager.h"

namespace error_message {
void ReportInnerError(
    const char *file_name, const char *func, uint32_t line, const std::string error_code, const char *format, ...)
{
    return;
}
}  // namespace error_message

ErrorManager &ErrorManager::GetInstance()
{
    static ErrorManager instance;
    return instance;
}

thread_local error_message::Context ErrorManager::error_context_ = {0, " ", " ", " "};
error_message::Context &ErrorManager::GetErrorManagerContext()
{
    return error_context_;
}

void ErrorManager::SetErrorContext(error_message::Context error_context)
{
    return;
}

std::string Stage = "[Stage0][Stage1]";
const std::string &ErrorManager::GetLogHeader()
{
    return Stage;
}

void ErrorManager::ATCReportErrMessage(
    std::string error_code, const std::vector<std::string> &key, const std::vector<std::string> &value)
{
    std::cout << "ErrorMessage is:" << error_code << std::endl;
    for (int errMsgIdx = 0; errMsgIdx < key.size(); errMsgIdx++) {
        std::cout << key[errMsgIdx] << " is:" << value[errMsgIdx] << std::endl;
    }
}