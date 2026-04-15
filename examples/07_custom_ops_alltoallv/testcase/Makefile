# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

CC = g++
CXX = g++
MPICXX = mpic++
MPIRUN = mpirun

ASCEND_INSTALL_PATH ?= /usr/local/Ascend
CUSTOM_OPS_OPP_LIB_PATH ?= $(ASCEND_INSTALL_PATH)/lib
CUSTOM_OPS_OPP_INC_PATH ?= $(ASCEND_INSTALL_PATH)/include

INC_PATH = -I$(CUSTOM_OPS_OPP_INC_PATH) \
           -I$(ASCEND_INSTALL_PATH)/ drivers/extensions \
           -I$(ASCEND_INSTALL_PATH)/toolkit/latest/compiler/include \
           -I$(ASCEND_INSTALL_PATH)/driver/../driver/utils \
           -I$(ASCEND_INSTALL_PATH)/driver/../driver/dvpp \
           -I$(ASCEND_INSTALL_PATH)/driver/../driver/include

LIB_PATH = -L$(CUSTOM_OPS_OPP_LIB_PATH) \
           -L$(ASCEND_INSTALL_PATH)/lib64

LIBS = -lhccl -lacl -lmpi -lstdc++

SRC = main.cc
TARGET = hccl_custom_alltoallv_test
KERNEL_OBJ = hccl_custom_alltoallv_kernels.o

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) $(KERNEL_OBJ)
	$(MPICXX) $(SRC) -o $(TARGET) $(INC_PATH) $(LIB_PATH) $(LIBS) -DCUSTOM_OPS_OPP_LIB_PATH=$(CUSTOM_OPS_OPP_LIB_PATH) -DCUSTOM_OPS_OPP_INC_PATH=$(CUSTOM_OPS_OPP_INC_PATH)

$(KERNEL_OBJ):
	@echo "Kernel object should be built from CMake"
	@exit 1

run: $(TARGET)
	export LD_LIBRARY_PATH=$(CUSTOM_OPS_OPP_LIB_PATH):$(ASCEND_INSTALL_PATH)/lib64:$(ASCEND_INSTALL_PATH)/driver/lib64:$$LD_LIBRARY_PATH
	$(MPIRUN) --allow-run-as-root -n 2 ./$(TARGET)

clean:
	rm -f $(TARGET)