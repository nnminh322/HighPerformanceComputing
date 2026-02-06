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

### Cross-Language Comparison (Fair — cùng naive matmul)

> **Phương pháp:** Cả Python và C++ đều dùng naive triple-loop matmul, element-wise sigmoid. Python sử dụng `numba @njit` (JIT compile thành machine code), thời gian bao gồm cả JIT compilation. Cùng config: 300 iterations, lr=1.5, reg=0.01, seed=42.

| Procs | Lang | Samples/Proc | Train Acc | Test Acc | Time | Speedup | Efficiency |
|-------|------|-------------|-----------|----------|------|---------|------------|
| 1 (seq) | C++ | 4000 | 93.60% | 91.30% | 19.58s | 1.0x | — |
| 1 (seq) | Python | 4000 | 94.05% | 91.90% | 15.67s | 1.0x | — |
| 2 | C++ | 2000 | 93.60% | 91.30% | 10.74s | 1.82x | 91% |
| 2 | Python | 2000 | 94.05% | 91.90% | 9.08s | 1.73x | 86% |
| **4** | **C++** | **1000** | **93.60%** | **91.30%** | **5.99s** | **3.27x** | **82%** |
| **4** | **Python** | **1000** | **94.05%** | **91.90%** | **5.73s** | **2.74x** | **68%** |
| 8 | C++ | 500 | 93.60% | 91.30% | 5.18s | 3.78x | 47% |
| 8 | Python | 500 | 94.05% | 91.90% | 6.61s | 2.37x | 30% |

**Nhận xét:**
1. **Sequential:** Python (15.67s) và C++ (19.58s) ở mức tương đương — numba LLVM tối ưu loop tốt hơn một chút so với clang++ trên `std::vector<std::vector<double>>` (non-contiguous memory)
2. **C++ scaling tốt hơn:** C++ đạt 82% efficiency ở 4 procs, Python chỉ 68%. Ở 8 procs, C++ vẫn cải thiện (3.78x) trong khi Python bắt đầu chậm lại (2.37x)
3. **8 procs:** Python (6.61s) chậm hơn C++ (5.18s) — MPI overhead trong Python (pickle serialization, GIL) lớn hơn C++ (raw memory copy)
4. **Sweet spot: 4 procs** cho cả hai ngôn ngữ với dataset 4000 samples

### Phân tích sự khác biệt Accuracy

**Python:** Train 94.05%, Test 91.90% | **C++:** Train 93.60%, Test 91.30%

Chênh lệch ~0.5% do: (1) RNG khác nhau (`numpy.random.randn` vs `std::mt19937`) → initial weights khác → convergence path khác, (2) Data loading: Python từ `.mat`, C++ từ CSV convert.

### Phân tích Diminishing Returns ở 8 Procs

**Phân tích chi tiết bằng Amdahl's Law:**

Mỗi iteration bao gồm:
- **Computation** (parallelizable): Forward + Backprop trên local data
- **Communication** (sequential): `MPI_Allreduce` cho theta1_grad (25×401) + theta2_grad (10×26) + cost → ~82KB/iteration × 300 iterations

```
Speedup lý thuyết (Amdahl's Law): S = 1 / (f + (1-f)/p)
  - f = tỷ lệ sequential (communication)
  - p = số processes

Từ kết quả C++ 4 procs (3.27x):
  3.27 = 1 / (f + (1-f)/4) → f ≈ 0.056 (5.6% communication)

Dự đoán cho 8 procs:
  S = 1 / (0.056 + 0.944/8) = 5.75x
  Thực tế C++: 3.78x | Python: 3.22x — thấp hơn dự đoán do overhead tăng phi tuyến
```

**Nguyên nhân cụ thể:**
1. **Dataset quá nhỏ:** 500 samples/proc với 8 procs → computation ~8ms/iter, gần bằng communication overhead
2. **MPI_Allreduce latency tăng:** log₂(8) = 3 bước reduce vs log₂(4) = 2
3. **Cache contention:** 8 processes trên 10 cores → chia sẻ L2/L3 cache
4. **Memory bandwidth saturation:** 8 processes đồng thời truy cập memory

### Key Observations

1. **Fair comparison:** Cùng naive matmul, C++ và Python cho thời gian tương đương (19.58s vs 15.67s sequential). C++ scaling tốt hơn khi tăng processes
2. **Sweet spot = 4 procs:** Cả hai ngôn ngữ đều đạt hiệu quả tốt nhất ở 4 procs với dataset 4000 samples
3. **8 procs không hiệu quả:** Diminishing returns do dataset nhỏ (500 samples/proc), MPI overhead tăng, cache contention
4. **Accuracy ổn định:** Tất cả parallel versions cho cùng accuracy với sequential (cùng ngôn ngữ) → data parallelism không ảnh hưởng model quality

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

