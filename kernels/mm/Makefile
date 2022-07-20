LD=clang++
CC=nvcc
NVCC=nvcc

EXECUTABLE=mm
BUILD_DIR := ./_build

CFLAGS=
# CXXFLAGS=-L/usr/lib/cuda -lcudart
CXXFLAGS=-rpath ../../target/debug -L../../target/debug -lcudart
NVCFLAGS=
NVCFLAGS+=-g -lineinfo -O3 -Xcompiler -rdynamic
NVCFLAGS+=--use_fast_math --ptxas-options=-v --compiler-options -fPIC --compiler-options -Wall
NVCFLAGS+=-gencode arch=compute_30,code=sm_30
NVCFLAGS+=-gencode arch=compute_30,code=sm_53
NVCFLAGS+=-gencode arch=compute_53,code=sm_53
NVCFLAGS+=-gencode arch=compute_60,code=compute_60
NVCFLAGS+=-gencode arch=compute_75,code=compute_75

SRCS := mm.cc mm.cu

# hello.cpp becomes BUILD_DIR/hello.cpp.o
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

$(BUILD_DIR)/$(EXECUTABLE): $(OBJS)
		$(LD) $(CCFLAGS) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.cc.o: %.cc
		mkdir -p $(dir $@)
		$(CC) $(CCFLAGS) -c $< -o $@
		
$(BUILD_DIR)/%.cu.o: %.cu
		mkdir -p $(dir $@)
		$(NVCC) $(NVCFLAGS) -c $< -o $@

.PHONY: clean
clean:
		$(RM) -rf $(BUILD_DIR)
