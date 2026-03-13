如何编译LLT-V2

1）删除历史目录，从头开始执行（命令执行路径为代码根目录）
rm -rf tmp && mkdir -p tmp && cd tmp && cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_v2_alg -DFULL_COVERAGE=false -DCOVERAGE_RC_CONFIG=false -DFEATURE_LIST=USE_LEGACY_PROTOBUF=TRUE && TARGETS=legacy_alg_testcase  make -j64

rm -rf tmp && mkdir -p tmp && cd tmp && cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_v2_alg -DFULL_COVERAGE=false -DCOVERAGE_RC_CONFIG=false -DFEATURE_LIST=USE_LEGACY_PROTOBUF=TRUE && TARGETS=legacy_alg_ccu_1d_testcase  make -j64

rm -rf tmp && mkdir -p tmp && cd tmp && cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_v2_alg -DFULL_COVERAGE=false -DCOVERAGE_RC_CONFIG=false -DFEATURE_LIST=USE_LEGACY_PROTOBUF=TRUE && TARGETS=legacy_alg_ccu_1d_hf16p_testcase  make -j64

rm -rf tmp && mkdir -p tmp && cd tmp && cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_v2_alg -DFULL_COVERAGE=false -DCOVERAGE_RC_CONFIG=false -DFEATURE_LIST=USE_LEGACY_PROTOBUF=TRUE && TARGETS=legacy_alg_ccu_2d_testcase  make -j64

rm -rf tmp && mkdir -p tmp && cd tmp && cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_v2_alg -DFULL_COVERAGE=false -DCOVERAGE_RC_CONFIG=false -DFEATURE_LIST=USE_LEGACY_PROTOBUF=TRUE && TARGETS=legacy_alg_aicpu_2d_testcase  make -j64

rm -rf tmp && mkdir -p tmp && cd tmp && cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_v2_alg -DFULL_COVERAGE=false -DCOVERAGE_RC_CONFIG=false -DFEATURE_LIST=USE_LEGACY_PROTOBUF=TRUE && TARGETS=orion_alg_aiv_1d_testcase  make -j64

rm -rf tmp && mkdir -p tmp && cd tmp && cmake ../cmake/superbuild/ -DCUSTOM_PYTHON=python3 -DHOST_PACKAGE=ut -DBUILD_MOD=hccl_v2_alg -DFULL_COVERAGE=false -DCOVERAGE_RC_CONFIG=false -DFEATURE_LIST=USE_LEGACY_PROTOBUF=TRUE && TARGETS=legacy_alg_function_ut_testcase  make -j64

2）不删除历史目录，再次执行（适用于调试代码的场景）(执行目录为tmp目录)
TARGETS=legacy_alg_testcase  make -j64

TARGETS=legacy_alg_ccu_1d_testcase  make -j64

TARGETS=legacy_alg_ccu_2d_testcase  make -j64

TARGETS=legacy_alg_ccu_1d_hf16p_testcase  make -j64

TARGETS=legacy_alg_aicpu_2d_testcase  make -j64

TARGETS=orion_alg_aiv_1d_testcase  make -j64

TARGETS=legacy_alg_function_ut_testcase  make -j64

如何执行LLT
1）在tmp目录下执行
./llt_gcc4.9.3-prefix/src/llt_gcc4.9.3-build/llt/ace/comop/hccl/orion/algorithm/legacy_alg_testcase

./llt_gcc4.9.3-prefix/src/llt_gcc4.9.3-build/llt/ace/comop/hccl/orion/algorithm/legacy_alg_ccu_1d_testcase

./llt_gcc4.9.3-prefix/src/llt_gcc4.9.3-build/llt/ace/comop/hccl/orion/algorithm/legacy_alg_ccu_1d_hf16p_testcase

./llt_gcc4.9.3-prefix/src/llt_gcc4.9.3-build/llt/ace/comop/hccl/orion/algorithm/legacy_alg_ccu_2d_testcase

./llt_gcc4.9.3-prefix/src/llt_gcc4.9.3-build/llt/ace/comop/hccl/orion/algorithm/legacy_alg_aicpu_2d_testcase

./llt_gcc4.9.3-prefix/src/llt_gcc4.9.3-build/llt/ace/comop/hccl/orion/algorithm/orion_alg_aiv_1d_testcase

./llt_gcc4.9.3-prefix/src/llt_gcc4.9.3-build/llt/ace/comop/hccl/orion/algorithm/legacy_alg_function_ut_testcase