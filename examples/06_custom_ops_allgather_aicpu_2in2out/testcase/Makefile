# Makefile for dual-route allgather testcase

CXX := mpic++
CXXFLAGS := -std=c++17 -O2 -g -Wall
ASCEND_HOME ?= /usr/local/Ascend/ascend-toolkit/latest
INCLUDES := -I../inc -I$(ASCEND_HOME)/include -I$(ASCEND_HOME)/include/hccl
LIB_DIR ?= ../build/op_host
LIBS := -L$(ASCEND_HOME)/lib64 -lascendcl -lhccl -L$(LIB_DIR) -lhccl_custom_allgather_2in2out -pthread

TARGET := main
NP ?= 2
COUNT0 ?= 1024
COUNT1 ?= 2048
WARMUP ?= 20
ITERS ?= 100

all: $(TARGET)

$(TARGET): main.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	export LD_LIBRARY_PATH=$(LIB_DIR):$(ASCEND_HOME)/lib64:$$LD_LIBRARY_PATH && \
	mpirun -n $(NP) ./$(TARGET) --count0=$(COUNT0) --count1=$(COUNT1) --warmup=$(WARMUP) --iters=$(ITERS)
