# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This testcase Makefile follows the same direct-build style as examples/04_custom_ops_p2p/testcase.
# It is intended for quick local bring-up when the custom op has already been installed to the CANN custom path.
# -----------------------------------------------------------------------------------------------------------

ifndef ASCEND_HOME_PATH
    $(error "ASCEND_HOME_PATH is not set, please source the Ascend environment first")
endif

CXXFLAGS := -std=c++14 \
        -Werror \
        -fstack-protector-strong \
        -fPIE -pie \
        -O2 \
        -Wl,-z,relro \
        -Wl,-z,now \
        -Wl,-z,noexecstack \
        -Wl,--copy-dt-needed-entries \
        -D_GLIBCXX_USE_CXX11_ABI=0

SOURCES = main.cc
ASCEND_INC_DIR = ${ASCEND_HOME_PATH}/include
ASCEND_LIB_DIR = ${ASCEND_HOME_PATH}/lib64
CUSTOM_VENDOR ?= cust
CUSTOM_INC_DIR = ${ASCEND_HOME_PATH}/opp/vendors/$(CUSTOM_VENDOR)/include
CUSTOM_LIB_DIR = ${ASCEND_HOME_PATH}/opp/vendors/$(CUSTOM_VENDOR)/lib64
TARGET = allgatherbatch_testcase

# Default run arguments can be overridden from command line, for example:
# make test TOKEN_BYTES=65536 SCALE_COUNT=0 DEVICES=4 WARMUP=2 ITERS=20
TOKEN_BYTES ?= 327680
SCALE_COUNT ?= 128
DEVICES ?= 8
PRINT_COUNT ?= 8
WARMUP ?= 1
ITERS ?= 1
EXTRA_ARGS ?=

FAST_TOKEN_BYTES ?= 65536
FAST_SCALE_COUNT ?= 0
FAST_DEVICES ?= 4
FAST_WARMUP ?= 1
FAST_ITERS ?= 5

SINGLE_TOKEN_BYTES ?= 327680
SINGLE_DEVICES ?= 8
SINGLE_WARMUP ?= 1
SINGLE_ITERS ?= 5

LIBS = -L$(ASCEND_LIB_DIR) -lascendcl -lhcomm -L${CUSTOM_LIB_DIR} -lhccl_allgatherbatch
INCS = -I$(ASCEND_INC_DIR) -I${CUSTOM_INC_DIR}
RUN_ARGS = --token-bytes $(TOKEN_BYTES) \
           --scale-count $(SCALE_COUNT) \
           --devices $(DEVICES) \
           --print-count $(PRINT_COUNT) \
           --warmup $(WARMUP) \
           --iters $(ITERS) \
           $(EXTRA_ARGS)

all:
	g++ $(CXXFLAGS) $(SOURCES) $(INCS) $(LIBS) -o ${TARGET}
	@echo "${TARGET} compile completed"

test: all
	export LD_LIBRARY_PATH=${CUSTOM_LIB_DIR}:${ASCEND_LIB_DIR}:$${LD_LIBRARY_PATH}; \
	./$(TARGET) $(RUN_ARGS)

# 默认双 item 场景，贴近设计文档的典型输入规模。
test-default: test

# 快速冒烟场景，适合先确认 runtime 和自定义算子链路是否通。
test-fast: all
	export LD_LIBRARY_PATH=${CUSTOM_LIB_DIR}:${ASCEND_LIB_DIR}:$${LD_LIBRARY_PATH}; \
	./$(TARGET) \
	    --token-bytes $(FAST_TOKEN_BYTES) \
	    --scale-count $(FAST_SCALE_COUNT) \
	    --devices $(FAST_DEVICES) \
	    --print-count $(PRINT_COUNT) \
	    --warmup $(FAST_WARMUP) \
	    --iters $(FAST_ITERS) \
	    $(EXTRA_ARGS)

# 单 item 场景，快速观察只有 token item 时的行为。
test-single-item: all
	export LD_LIBRARY_PATH=${CUSTOM_LIB_DIR}:${ASCEND_LIB_DIR}:$${LD_LIBRARY_PATH}; \
	./$(TARGET) \
	    --token-bytes $(SINGLE_TOKEN_BYTES) \
	    --scale-count 0 \
	    --devices $(SINGLE_DEVICES) \
	    --print-count $(PRINT_COUNT) \
	    --warmup $(SINGLE_WARMUP) \
	    --iters $(SINGLE_ITERS) \
	    $(EXTRA_ARGS)

clean:
	rm -f ${TARGET}

help:
	@echo ""
	@echo "AllGatherBatch Custom Op Testcase"
	@echo "================================="
	@echo ""
	@echo "Targets:"
	@echo "  all               - Build the testcase"
	@echo "  test              - Build and run with configurable variables"
	@echo "  test-default      - Alias of test, using default dual-item settings"
	@echo "  test-fast         - Run a smaller, faster smoke scenario"
	@echo "  test-single-item  - Run token-only scenario with scale disabled"
	@echo "  clean             - Remove build artifacts"
	@echo "  help              - Show this help"
	@echo ""
	@echo "Variables for test/test-default:"
	@echo "  CUSTOM_VENDOR installed custom-op vendor, default $(CUSTOM_VENDOR)"
	@echo "  TOKEN_BYTES  int8 token bytes per rank, default $(TOKEN_BYTES)"
	@echo "  SCALE_COUNT  fp32 scale count per rank, default $(SCALE_COUNT)"
	@echo "  DEVICES      device count to launch, default $(DEVICES)"
	@echo "  PRINT_COUNT  preview element count, default $(PRINT_COUNT)"
	@echo "  WARMUP       warmup iterations, default $(WARMUP)"
	@echo "  ITERS        measured iterations, default $(ITERS)"
	@echo "  EXTRA_ARGS   extra flags, for example EXTRA_ARGS=--no-verify"
	@echo ""
	@echo "Examples:"
	@echo "  make test CUSTOM_VENDOR=cust TOKEN_BYTES=65536 SCALE_COUNT=0 DEVICES=4 WARMUP=2 ITERS=20"
	@echo "  make test-fast"
	@echo "  make test-single-item EXTRA_ARGS=--no-verify"
	@echo ""

.PHONY: all test test-default test-fast test-single-item clean help