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
CUSTOM_INC_DIR = ${ASCEND_HOME_PATH}/opp/vendors/cust/include
CUSTOM_LIB_DIR = ${ASCEND_HOME_PATH}/opp/vendors/cust/lib64
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

clean:
	rm -f ${TARGET}

help:
	@echo ""
	@echo "AllGatherBatch Custom Op Testcase"
	@echo "================================="
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the testcase"
	@echo "  test      - Build and run the testcase"
	@echo "  clean     - Remove build artifacts"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  TOKEN_BYTES  int8 token bytes per rank, default $(TOKEN_BYTES)"
	@echo "  SCALE_COUNT  fp32 scale count per rank, default $(SCALE_COUNT)"
	@echo "  DEVICES      device count to launch, default $(DEVICES)"
	@echo "  PRINT_COUNT  preview element count, default $(PRINT_COUNT)"
	@echo "  WARMUP       warmup iterations, default $(WARMUP)"
	@echo "  ITERS        measured iterations, default $(ITERS)"
	@echo "  EXTRA_ARGS   extra flags, for example EXTRA_ARGS=--no-verify"
	@echo ""
	@echo "Example:"
	@echo "  make test TOKEN_BYTES=65536 SCALE_COUNT=0 DEVICES=4 WARMUP=2 ITERS=20"
	@echo ""

.PHONY: all test clean help
