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


### Configuration

- **Processes:** 4
- **Data split:** 4000 samples ÷ 4 = 1000 samples/process
- **Learning rate:** 1.5
- **Iterations:** 300
- **Regularization:** 0.01

## 3. Performance Results

### System Info
- **CPU:** Apple Silicon, 10 cores / 10 threads
- **Max threads:** 10

### C++ Scaling Results

| Version | Procs | Samples/Proc | Train Acc | Test Acc | Training Time | Speedup (vs Seq) |
|---------|-------|-------------|-----------|----------|--------------|-------------------|
| Sequential | 1 | 4000 | 93.60% | 91.30% | 19.58s | 1.0x |
| Parallel | 2 | 2000 | 93.60% | 91.30% | 10.74s | 1.82x |
| **Parallel** | **4** | **1000** | **93.60%** | **91.30%** | **5.99s** | **3.27x** |
| Parallel | 8 | 500 | 93.60% | 91.30% | 5.18s | 3.78x |

### Cross-Language Comparison

| Version | Language | Train Acc | Test Acc | Time | Speedup (vs Python Seq) |
|---------|----------|-----------|----------|------|-------------------------|
| Sequential | Python | 94.05% | 91.90% | 156.45s | 1.0x |
| Parallel (4 proc) | Python | 94.20% | 92.10% | ~40s | ~4x |
| Sequential | C++ | 93.60% | 91.30% | 19.58s | 8.0x |
| Parallel (2 proc) | C++ | 93.60% | 91.30% | 10.74s | 14.6x |
| **Parallel (4 proc)** | **C++** | **93.60%** | **91.30%** | **5.99s** | **26.1x** |
| Parallel (8 proc) | C++ | 93.60% | 91.30% | 5.18s | 30.2x |

### Key Observations

1. **C++ vs Python:** C++ sequential nhanh hơn Python sequential ~8x
2. **Parallel scaling (C++):**
   - 2 procs: 1.82x speedup (efficiency 91%)
   - 4 procs: 3.27x speedup (efficiency 82%)
   - 8 procs: 3.78x speedup (efficiency 47%)
3. **Diminishing returns:** Từ 4→8 procs chỉ cải thiện thêm ~15% (5.99s→5.18s), do:
   - Dataset nhỏ (4000 samples): mỗi process chỉ xử lý 500 samples với 8 procs
   - MPI communication overhead tăng khi số processes tăng
   - Amdahl's Law: phần sequential (AllReduce, weight update) giới hạn speedup
4. **Sweet spot:** 4 processes là lựa chọn tối ưu cho dataset này (cân bằng speedup vs resource)
5. **Accuracy:** Tương đương giữa tất cả các phiên bản (~93.60% train, ~91.30% test)

### Bottlenecks & Đánh giá khả năng tối ưu MPI

**Thử nghiệm tối ưu: Parallel Evaluation**

Ý tưởng: Thay vì chỉ process 0 predict toàn bộ train/test set, ta phân tán việc evaluation:
- Scatter data cho tất cả processes
- Mỗi process predict phần của mình  
- `MPI_Reduce` để tổng hợp số predictions đúng về root

File: `src/parallel_optimized.cpp`

**Kết quả so sánh:**

| Version | Training Time | Eval Time | Total |
|---------|---------------|-----------|-------|
| Parallel (gốc) | 5.56s | ~0.02s | 5.58s |
| Parallel (optimized) | 5.58s | 0.014s | 5.59s |

**Kết luận:** Với dataset nhỏ (4000 train, 1000 test), việc tối ưu evaluation **không mang lại cải thiện đáng kể** vì:
- Evaluation time chỉ chiếm ~0.3% tổng thời gian
- Overhead của MPI communication (scatter/reduce) có thể lớn hơn benefit

