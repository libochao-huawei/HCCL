#include "sim_runner_db.h"
#include "sim_sqlite_db.h"

namespace RunnerDB {
    std::vector<std::string> GetAllTableName()
    {
        return SimRunnerSqliteDB::Instance().GetAllTableName();
    }
}
