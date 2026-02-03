CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

CONDA_PREFIX ?= $(shell conda info --base)/envs/master
MPI_INCLUDE = -I$(CONDA_PREFIX)/include
MPI_LIB = -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib -lmpi

SRC_DIR = src
BUILD_DIR = build

SEQUENTIAL = $(BUILD_DIR)/sequential
PARALLEL = $(BUILD_DIR)/parallel
PARALLEL_OPT = $(BUILD_DIR)/parallel_optimized

.PHONY: all clean run_seq run_par run_opt

all: $(BUILD_DIR) $(SEQUENTIAL) $(PARALLEL) $(PARALLEL_OPT)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(SEQUENTIAL): $(SRC_DIR)/sequential.cpp $(SRC_DIR)/matrix_utils.h
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $@ $(SRC_DIR)/sequential.cpp -lm

$(PARALLEL): $(SRC_DIR)/parallel.cpp $(SRC_DIR)/matrix_utils.h
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $(MPI_INCLUDE) -o $@ $(SRC_DIR)/parallel.cpp $(MPI_LIB) -lm

$(PARALLEL_OPT): $(SRC_DIR)/parallel_optimized.cpp $(SRC_DIR)/matrix_utils.h
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $(MPI_INCLUDE) -o $@ $(SRC_DIR)/parallel_optimized.cpp $(MPI_LIB) -lm

run_seq: $(SEQUENTIAL)
	./$(SEQUENTIAL)

run_par: $(PARALLEL)
	mpirun -np 4 ./$(PARALLEL)

run_opt: $(PARALLEL_OPT)
	mpirun -np 4 ./$(PARALLEL_OPT)

clean:
	rm -rf $(BUILD_DIR)
