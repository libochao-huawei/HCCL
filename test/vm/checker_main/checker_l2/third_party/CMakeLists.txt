# --- configure doctest ---
add_library(doctest_lib INTERFACE) 
target_include_directories(doctest_lib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/doctest
)
add_library(doctest::doctest ALIAS doctest_lib)

# --- configure CLI11 ---
add_library(cli11_lib INTERFACE)
target_include_directories(cli11_lib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/CLI11
)
add_library(CLI11::CLI11 ALIAS cli11_lib)

# --- configure nlohmann_json ---
add_library(nlohmann_json_lib INTERFACE)
target_include_directories(nlohmann_json_lib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann_json
)
add_library(JSON::JSON ALIAS nlohmann_json_lib)

# --- configure spdlog ---
add_library(spdlog_lib INTERFACE)
target_include_directories(spdlog_lib INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/spdlog
)
add_library(spdlog::spdlog ALIAS spdlog_lib)

# --- configure sqlite3 ---
find_library(SQLITE3_LIBRARY sqlite3 PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
find_path(SQLITE3_INCLUDE_DIR sqlite3.h PATHS /usr/include /usr/local/include)

if(SQLITE3_LIBRARY AND SQLITE3_INCLUDE_DIR)
    if(NOT TARGET SQLite::SQLite3)
        add_library(SQLite::SQLite3 UNKNOWN IMPORTED)
    endif()
    set_target_properties(SQLite::SQLite3 PROPERTIES
        IMPORTED_LOCATION ${SQLITE3_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${SQLITE3_INCLUDE_DIR}
    )
    message(STATUS "SQLite3 found: ${SQLITE3_LIBRARY}")
else()
    message(WARNING "SQLite3 not found, runnerdb may not compile correctly")
endif()