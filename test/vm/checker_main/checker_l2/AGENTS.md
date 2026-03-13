# HCCL VM - Agent Guidelines

## Build & Test Commands

### Environment Setup
```bash
source set_env.sh  # Set ASCEND_HOME_PATH and ASCEND_TOOLKIT_HOME before building
```

### Build
```bash
./build.sh                    # Configure and build
./build.sh -r                 # Reset build directory and rebuild
./build.sh -i                 # Build and install to hccl_vm_install/
./build.sh -p                 # Build and create CPack package
```

### Run Tests
```bash
cd build && cmake .. && make -j$(nproc)
cd build && ctest             # Run all tests
cd build && ctest -R hccl_host_test --verbose  # Run specific test
cd build/output/bin/hccl_host_test  # Run test binary directly
```

### Test Targets
- `hccl_host_test` - Host plugin manager tests
- `hccl_ipc_test` - IPC communication tests  
- `hccl_shm_test` - Shared memory tests
- `hccl_virtual_runtime_test` - Virtual runtime tests
- `hccl_proxy_test` - Proxy level tests

## Code Style Guidelines

### Imports
- Standard library headers first (alphabetical order)
- Third-party headers next
- Local project headers last
- Use angle brackets for system headers, quotes for local headers

```cpp
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "hccl_common_defs.h"
#include "hccl_vm_log.h"
```

### Formatting
- 4 spaces for indentation (no tabs)
- Opening brace on same line for functions/classes
- Closing brace on its own line
- Maximum line length: 120 characters
- Blank line between logical sections

```cpp
namespace HcclSim {

class HcclPluginManager {
 public:
  HcclVmResult RegisterPlugin(const HcclPlugin& plugin, const std::string& tag);
  
  std::vector<std::string> GetRegisteredPluginTags();

 private:
  HcclPluginManager() = default;
  ~HcclPluginManager() = default;
};

}  // namespace HcclSim
```

### Naming Conventions
- **Namespaces**: `HcclSim` for main namespace
- **Classes**: PascalCase (`HcclPluginManager`, `SHMManager`)
- **Functions**: PascalCase (`RegisterPlugin`, `InitShm`)
- **Variables**: camelCase (`pluginPath`, `m_manifest`)
- **Member variables**: Prefix with `m_` (`m_pluginPath`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_SCAN_DEPTH`, `BYTE_NUM_4K`)
- **Test classes**: PascalCase with `Test` suffix or `Suite` suffix
- **Test functions**: PascalCase with descriptive names

```cpp
namespace HcclSim {
constexpr uint32_t MAX_RETRY_COUNT = 3;

class SHMManager {
 public:
  static void* AllocateShmMemory(size_t size, const std::string& name);
  
 private:
  std::mutex m_mutex;
};
}
```

### Type Usage
- Use `std::uint32_t`, `std::int32_t` for fixed-width integers
- Use `size_t` for sizes and indices
- Prefer `std::string` over C-style strings
- Use `std::vector`, `std::array` over raw arrays
- Use `constexpr` for compile-time constants
- Use `nullptr` instead of `NULL`

### Error Handling
- Return `HcclVmResult` enum values for errors
- Use `HCCLVM_CHK_RET()` macro for function return checking
- Use `HCCLVM_CHK_PTR()` macro for null pointer validation
- Log errors using `HCCL_VM_ERROR()` macro before returning

```cpp
HcclVmResult SomeFunction(const void* ptr, int param) {
  HCCLVM_CHK_PTR(ptr);
  
  HcclVmResult ret = SomeOtherFunction(param);
  if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
    HCCL_VM_ERROR("[{}] Failed with code {}", __func__, ret);
    return ret;
  }
  
  return HcclVmResult::HCCL_SIM_SUCCESS;
}
```

### Error Code Categories
- `0-4095`: HCCL business errors
- `4096-5119`: Host errors
- `5120-6143`: Proxy errors
- `6144-7167`: SHM errors
- `7168-8191`: IPC errors

### Logging
- Use spdlog through `hccl_vm_log.h`
- Log format: `[LEVEL][PID:xxx][TID:xxx][file][line] message`
- Use `HCCL_VM_ERROR()`, `HCCL_VM_WARN()`, `HCCL_VM_INFO()` macros

```cpp
#include "hccl_vm_log.h"

HCCL_VM_ERROR("[{}] Operation failed: {}", __func__, errorCode);
HCCL_VM_INFO("[{}] Processing task {}", __func__, taskId);
```

### Memory Management
- Prefer RAII and smart pointers over raw `new`/`delete`
- Use `std::shared_ptr` for shared ownership
- Use `std::unique_ptr` for exclusive ownership
- Clean up shared memory in test teardown

### Thread Safety
- Use `std::mutex` for synchronization
- Use `std::atomic` for simple atomic operations
- Mark thread-safe methods appropriately
- Use `std::thread` for background tasks

### Test Writing (GoogleTest)
```cpp
#include "gtest/gtest.h"

class MyTestSuite : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup code
  }
  void TearDown() override {
    // Cleanup code
  }
};

TEST_F(MyTestSuite, TestFunctionName) {
  EXPECT_EQ(result, expected);
  EXPECT_NO_THROW(operation);
  EXPECT_THROW(failingOperation, std::runtime_error);
}
```

### Header Guards
```cpp
#ifndef HCCL_MODULE_H
#define HCCL_MODULE_H

// Content

#endif  // HCCL_MODULE_H
```

### Key Directories
- `src/` - Main source code
- `include/` - Public headers
- `test/` - Unit tests (Googletest-based)
- `third_party/` - External dependencies
- `build/` - Build output (generated)
- `hccl_vm_install/` - Installation output

### Common Dependencies
- spdlog (logging)
- Boost.Interprocess (shared memory, IPC)
- nlohmann::json (JSON parsing)
- CLI11 (command-line parsing)
- yaml-cpp (YAML parsing)
- zlib (compression)
