# MNIST Neural Network - Python Implementation

Mạng nơ-ron 3 lớp (400→25→10) huấn luyện trên tập MNIST, bao gồm phiên bản tuần tự và song song (MPI).

## Cấu trúc

```
project/
├── Mnist_sequential.py   # Phiên bản tuần tự
├── Mnist_parallel.py     # Phiên bản song song (MPI)
├── data/
│   └── mnistdata.mat     # Dữ liệu MNIST (5000 samples, 400 features)
└── README.md
```

## Yêu cầu

- Python 3.x
- numpy, scipy, scikit-learn, numba
- mpi4py (cho phiên bản song song)

## Chạy

### Sequential
```bash
python Mnist_sequential.py
```

### Parallel (MPI)
```bash
mpirun -np 4 python Mnist_parallel.py
```

## Cấu hình

| Thông số | Giá trị |
|----------|---------|
| Input layer | 400 neurons |
| Hidden layer | 25 neurons |
| Output layer | 10 neurons |
| Learning rate | 1.5 |
| Iterations | 300 |
| Regularization | 0.01 |
| Train/Test split | 80%/20% |

## Kết quả

| Version | Procs | Train Acc | Test Acc | Thời gian | Speedup |
|---------|-------|-----------|----------|-----------|---------|
| Sequential | 1 | 94.05% | 91.90% | 15.67s | 1.0x |
| Parallel | 2 | 94.05% | 91.90% | 9.08s | 1.73x |
| **Parallel** | **4** | **94.05%** | **91.90%** | **5.73s** | **2.74x** |
| Parallel | 8 | 94.05% | 91.90% | 6.61s | 2.37x |
