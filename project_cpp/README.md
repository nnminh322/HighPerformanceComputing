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

### 1. Install MPI (conda) (remember create env before running)

```bash
conda activate env 
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

**Parallel (2 processes):**
```bash
make run_par_2
# or
mpirun -np 2 ./build/parallel
```

**Parallel (4 processes):**
```bash
make run_par_4
# or
mpirun -np 4 ./build/parallel
```

**Parallel (8 processes):**
```bash
make run_par_8
# or
mpirun -np 8 ./build/parallel
```

## Performance Benchmark

**CPU Info:**
- 10 cores / 10 threads (Apple Silicon)
- 4000 training samples, 1000 test samples
- 300 iterations with learning rate 1.5

| Version | Processes | Training Time | Speedup | Efficiency |
|---------|-----------|---------------|---------|-----------|
| Sequential | 1 | 19.58s | 1.0x | - |
| Parallel | 2 | 10.74s | 1.82x | 91% |
| Parallel | 4 | 5.99s | 3.27x | 82% |
| Parallel | 8 | 5.18s | 3.78x | 47% |

**Key Results:**
- Accuracy consistent across all versions: Train 93.60%, Test 91.30%
- Sweet spot: 4 processes (best balance of speedup vs efficiency)
- Diminishing returns after 4 processes due to overhead and small dataset per process
