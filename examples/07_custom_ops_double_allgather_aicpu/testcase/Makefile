CXX := mpic++
CXXFLAGS := -std=c++17 -O2 -g -Wall
ASCEND_HOME ?= /home/kk/Ascend
INCLUDES := -I../inc -I$(ASCEND_HOME)/include -I$(ASCEND_HOME)/include/hccl
LIB_DIR := ../build
LIBS := -L$(ASCEND_HOME)/lib64 -lhccl -lascendcl -L$(LIB_DIR) -lhccl_custom_double_allgather -pthread

TARGET := main

all: $(TARGET)

$(TARGET): main.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	export HCCL_CUSTOM_KERNEL_LAUNCH_ASC=1 && export LD_LIBRARY_PATH=$(LIB_DIR):$(ASCEND_HOME)/lib64:$$LD_LIBRARY_PATH && mpirun -n 2 ./$(TARGET) --count0 1024 --count1 2048 --dtype0 fp32 --dtype1 int32 --warmup 5 --iters 20

