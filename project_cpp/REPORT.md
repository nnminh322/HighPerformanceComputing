# MNIST Neural Network - System Report

## 1. System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING FLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │  Data   │───▶│ Forward Pass │───▶│ Cost Function (J)   │    │
│  │ (X, y)  │    │  a1→z2→a2→h  │    │ Cross-entropy + L2  │    │
│  └─────────┘    └──────────────┘    └──────────┬──────────┘    │
│                                                 │               │
│                                                 ▼               │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │ Updated │◀───│ Gradient     │◀───│ Backpropagation     │    │
│  │ Weights │    │ Descent      │    │ δ3→δ2→∇θ1,∇θ2       │    │
│  └─────────┘    └──────────────┘    └─────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Forward Pass:**
- a1 = [1, X] (add bias)
- z2 = a1 × θ1ᵀ
- a2 = sigmoid(z2)
- z3 = [1, a2] × θ2ᵀ  
- h = sigmoid(z3)

**Backpropagation:**
- δ3 = h - y
- δ2 = (δ3 × θ2[:,1:]) ⊙ σ'(z2)
- ∇θ2 = δ3ᵀ × a2 / m
- ∇θ1 = δ2ᵀ × a1 / m

## 2. Parallelization Strategy

### Mô tả chiến lược song song hóa

**Vấn đề gốc:** Trong phiên bản tuần tự, toàn bộ 4000 mẫu dữ liệu được xử lý trên một process duy nhất. Mỗi iteration phải tính forward pass và backpropagation cho tất cả samples → tốn thời gian.

**Giải pháp:** Áp dụng **Data Parallelism** - chia dữ liệu training cho nhiều processes xử lý đồng thời.

**Những gì đã được song song hóa:**

1. **Chia dữ liệu (Data Partitioning):**
   - 4000 samples được chia đều cho 4 processes
   - Mỗi process xử lý 1000 samples độc lập
   - Sử dụng `MPI_Scatterv` để phân phối dữ liệu

2. **Tính toán song song (Parallel Computation):**
   - Forward propagation: Mỗi process tính activation cho phần dữ liệu của mình
   - Backpropagation: Mỗi process tính local gradients từ local data
   - 4 processes chạy đồng thời → giảm thời gian tính toán ~4x

3. **Đồng bộ hóa Gradients (Gradient Synchronization):**
   - Sau mỗi iteration, các local gradients được tổng hợp bằng `MPI_Allreduce`
   - Tính trung bình gradients từ tất cả processes
   - Tất cả processes cập nhật weights giống nhau → đảm bảo model consistency

**Tại sao phương pháp này hiệu quả:**
- Computation được phân tán → mỗi process làm ít việc hơn
- Memory được phân tán → mỗi process chỉ load 1/4 dữ liệu
- Communication overhead nhỏ (chỉ sync gradients, không sync activations)

### Data Parallelism với MPI

```
┌────────────────────────────────────────────────────────────────┐
│                    MPI DATA PARALLELISM                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Process 0 (Root)          Process 1,2,3                       │
│  ┌────────────────┐        ┌────────────────┐                  │
│  │ Load Data      │        │                │                  │
│  │ Init Weights   │        │                │                  │
│  └───────┬────────┘        └────────────────┘                  │
│          │                                                     │
│          ▼ Broadcast(θ1, θ2)                                   │
│  ┌───────────────────────────────────────────────────┐        │
│  │     All processes receive initial weights          │        │
│  └───────────────────────────────────────────────────┘        │
│          │                                                     │
│          ▼ Scatter(X, y)                                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ Proc 0  │  │ Proc 1  │  │ Proc 2  │  │ Proc 3  │           │
│  │ 1000    │  │ 1000    │  │ 1000    │  │ 1000    │           │
│  │ samples │  │ samples │  │ samples │  │ samples │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │            │                 │
│       ▼            ▼            ▼            ▼                 │
│  ┌─────────────────────────────────────────────────┐          │
│  │  Local Forward + Backprop → Local Gradients     │          │
│  └─────────────────────────────────────────────────┘          │
│       │            │            │            │                 │
│       └────────────┴────────────┴────────────┘                 │
│                        │                                       │
│                        ▼ AllReduce(∇θ) / numProcs              │
│              ┌─────────────────────┐                           │
│              │  Average Gradients  │                           │
│              │  Update Weights     │                           │
│              └─────────────────────┘                           │
│                        │                                       │
│                        ▼ (repeat 300 iterations)               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### MPI Operations Used

| Operation | Purpose |
|-----------|---------|
| `MPI_Bcast` | Broadcast weights từ root đến tất cả processes |
| `MPI_Scatterv` | Chia dữ liệu training cho các processes |
| `MPI_Allreduce` | Tổng hợp gradients từ tất cả processes |

### Configuration

- **Processes:** 4
- **Data split:** 4000 samples ÷ 4 = 1000 samples/process
- **Learning rate:** 1.5
- **Iterations:** 300
- **Regularization:** 0.01

## 3. Performance Results

| Version | Language | Train Acc | Test Acc | Time | Speedup |
|---------|----------|-----------|----------|------|---------|
| Sequential | Python | 94.05% | 91.90% | 156.45s | 1.0x |
| Parallel (4 proc) | Python | 94.20% | 92.10% | ~40s | ~4x |
| Sequential | C++ | 93.60% | 91.30% | 19.70s | 7.9x |
| **Parallel (4 proc)** | **C++** | **93.60%** | **91.30%** | **5.82s** | **26.9x** |

### Key Observations

1. **C++ vs Python:** C++ sequential nhanh hơn Python sequential ~8x
2. **Parallel speedup:** C++ parallel (4 proc) đạt ~3.4x so với C++ sequential
3. **Overall:** C++ parallel nhanh hơn Python sequential ~27x
4. **Accuracy:** Tương đương giữa các phiên bản (~91-94%)

### Bottlenecks

- Matrix operations trong `matrix_utils.h` chưa tối ưu (có thể dùng BLAS/LAPACK)
- MPI communication overhead với small batch sizes
- Memory allocation/deallocation trong mỗi iteration

### Potential Improvements

1. Sử dụng BLAS (OpenBLAS, Intel MKL) cho matrix multiplication
2. Overlap computation và communication
3. GPU acceleration (CUDA/OpenCL)
