# HCCL Project Guide

## Quick Start

See [HCCLQuickStart.md](../hccl.wiki/HCCLQuickStart.md) for the full end-to-end tutorial covering environment setup, CANN Toolkit installation, HCCL compilation, and test execution.

## Key Points for Non-root Users

- CANN Toolkit must be installed with `--install-path=$HOME/Ascend`
- Always `source $HOME/Ascend/cann/set_env.sh` before building or testing
- HCCL 9.x requires CANN 9.x (incompatible with CANN 8.x)
- ST testing requires HCCL package installed into the CANN directory first

## Build & Test Commands

```bash
source $HOME/Ascend/cann/set_env.sh

# Build
bash build.sh --pkg

# Install HCCL into CANN
bash ./build_out/cann-hccl_9.1.0_linux-aarch64.run --full --install-path=$HOME/Ascend

# UT test
bash build.sh --ut

# ST test (all)
bash build.sh --st

# ST test (specific test cases on feature branches)
# Build first, then run targeted tests with gtest filter
bash build.sh --st  # builds the test binary
/home/l00907184/work_code/hccl/test/st/algorithm/build/testcase/hccl_checker_ops_stest --gtest_filter="*<feature_keyword>*"
# Example: on the allgatherv-multilevel-mesh1d branch:
#   --gtest_filter="*all_gather_v*:*reduce_scatter_v*"
```

## ST Testing on Feature Branches

When running ST tests on a feature branch, only run test cases related to that feature using gtest filter, instead of running all ST tests. This avoids long timeouts and ensures relevant tests are validated quickly.

1. Build the test binary: `bash build.sh --st`
2. Run targeted tests: `hccl_checker_ops_stest --gtest_filter="*<feature_keyword>*"`
3. Multiple keywords: `--gtest_filter="*keyword1*:*keyword2*"`

## Documented Solutions

`docs/solutions/` — documented solutions to past problems (bugs, best practices, workflow patterns), organized by category with YAML frontmatter (`module`, `tags`, `problem_type`). Relevant when implementing or debugging in documented areas.