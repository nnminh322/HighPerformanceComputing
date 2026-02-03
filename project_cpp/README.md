# MNIST Neural Network - C++ Implementation

3-layer neural network for MNIST digit recognition implemented in C++ with MPI parallelization.

## Structure

```
project_cpp/
├── src/                    # Source code
│   ├── matrix_utils.h      # Matrix operations library
│   ├── sequential.cpp      # Sequential version
│   └── parallel.cpp        # MPI parallel version
├── data/                   # Dataset (CSV format)
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── scripts/                # Utility scripts
│   └── convert_data.py     # Convert .mat to CSV
├── build/                  # Compiled binaries
└── Makefile
```

## Network Architecture

- Input: 400 units (20x20 image)
- Hidden: 25 units (sigmoid)
- Output: 10 units (digits 0-9)

## Setup

### 1. Install MPI (conda)

```bash
conda activate master
conda install -c conda-forge openmpi openmpi-mpicc
```

### 2. Prepare Data

```bash
python scripts/convert_data.py /path/to/mnistdata.mat data/
```

### 3. Build

```bash
make all
```

## Run

**Sequential:**
```bash
make run_seq
# or
./build/sequential
```

**Parallel (4 processes):**
```bash
make run_par
# or
mpirun -np 4 ./build/parallel
```

## Results

| Version | Train Acc | Test Acc | Time |
|---------|-----------|----------|------|
| C++ Sequential | 93.60% | 91.30% | 19.70s |
| C++ Parallel (4 proc) | 93.60% | 91.30% | 5.82s |
