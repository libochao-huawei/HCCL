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