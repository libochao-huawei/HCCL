#include "sim_runner_db.h"

namespace RunnerDB {
    std::vector<std::string> GetAllTableName()
    {
        return SimRunnerMemDB::Instance().GetAllTableName();
    }
}
