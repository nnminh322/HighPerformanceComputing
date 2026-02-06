"""
MNIST Neural Network - Sequential (Naive Matmul)
Sử dụng numba @njit naive triple-loop matmul, giống hệt C++ implementation.
Mục đích: So sánh công bằng với C++ (cùng thuật toán, khác ngôn ngữ).
"""
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from numba import njit, prange
import time

# Cấu trúc mạng nơ-ron
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10

SEED = 42
np.random.seed(SEED)

# ==========================================
# Naive math operations (matching C++ style)
# ==========================================

@njit
def naive_matmul(A, B):
    """Triple-loop matrix multiply - giống hệt C++ matmul trong matrix_utils.h"""
    m, n = A.shape
    n2, p = B.shape
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

@njit
def naive_sigmoid(z):
    """Element-wise sigmoid - giống C++ sigmoidMatrix"""
    rows, cols = z.shape
    result = np.zeros_like(z)
    for i in range(rows):
        for j in range(cols):
            result[i, j] = 1.0 / (1.0 + np.exp(-z[i, j]))
    return result

@njit
def naive_sigmoid_gradient(z):
    """Element-wise sigmoid gradient - giống C++ sigmoidGradient"""
    s = naive_sigmoid(z)
    return s * (1.0 - s)

@njit
def add_bias_column(m):
    """Thêm cột bias = 1 vào đầu ma trận - giống C++ addBiasColumn"""
    rows, cols = m.shape
    result = np.ones((rows, cols + 1))
    result[:, 1:] = m
    return result

@njit
def argmax_rows(m):
    """Argmax theo hàng - giống C++ argmaxRows"""
    rows = m.shape[0]
    result = np.zeros(rows, dtype=np.int64)
    for i in range(rows):
        max_val = m[i, 0]
        max_idx = 0
        for j in range(1, m.shape[1]):
            if m[i, j] > max_val:
                max_val = m[i, j]
                max_idx = j
        result[i] = max_idx
    return result

# ==========================================
# Hàm tiện ích
# ==========================================

def convert_memory_ordering_f2c(array):
    return np.ascontiguousarray(array) if np.isfortran(array) else array

def load_and_split_data(training_file='data/mnistdata.mat', test_size=0.2, random_state=42):
    training_data = sio.loadmat(training_file)
    inputs = convert_memory_ordering_f2c(training_data['X'].astype('float64'))  # float64 matching C++
    labels = convert_memory_ordering_f2c(training_data['y'].ravel())
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def rand_init_weights(size_in, size_out):
    epsilon_init = np.sqrt(6) / np.sqrt(size_in + size_out)
    return np.random.randn(size_out, size_in + 1) * epsilon_init

def cost_function(theta1, theta2, inputs, labels, regular=0.01):
    m = len(inputs)
    
    # Forward propagation - dùng naive_matmul thay vì np.dot
    a1 = add_bias_column(inputs)                          # (m, 401)
    z2 = naive_matmul(a1, theta1.T.copy())                # (m, 25)
    a2 = naive_sigmoid(z2)
    a2 = add_bias_column(a2)                              # (m, 26)
    z3 = naive_matmul(a2, theta2.T.copy())                # (m, 10)
    h = naive_sigmoid(z3)
    
    # One-hot encoding
    y_matrix = np.eye(Output_layer_size)[labels.astype(np.int64) - 1]
    
    # Cost + regularization
    cost = (-1/m) * np.sum(y_matrix * np.log(h + 1e-10) + (1 - y_matrix) * np.log(1 - h + 1e-10))
    reg = (regular/(2*m)) * (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))
    total_cost = cost + reg
    
    # Backpropagation - dùng naive_matmul
    delta3 = h - y_matrix                                  # (m, 10)
    delta2 = naive_matmul(delta3, theta2[:, 1:].copy()) * naive_sigmoid_gradient(z2)  # (m, 25)
    
    theta2_grad = (1/m) * naive_matmul(delta3.T.copy(), a2) + (regular/m) * np.hstack([np.zeros((Output_layer_size, 1)), theta2[:, 1:]])
    theta1_grad = (1/m) * naive_matmul(delta2.T.copy(), a1) + (regular/m) * np.hstack([np.zeros((Hidden_layer_size, 1)), theta1[:, 1:]])
    
    return total_cost, (theta1_grad, theta2_grad)

def gradient_descent(inputs, labels, learning_rate=1, iterations=50):
    theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
    theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
    
    for i in range(iterations):
        cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2, inputs, labels)
        theta1 -= learning_rate * theta1_grad
        theta2 -= learning_rate * theta2_grad
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Iteration {i+1}/{iterations}, Cost: {cost:.4f}")
    
    return cost, (theta1, theta2)

def predict(model, inputs):
    theta1, theta2 = model
    a1 = add_bias_column(inputs)
    z2 = naive_matmul(a1, theta1.T.copy())
    a2 = naive_sigmoid(z2)
    a2 = add_bias_column(a2)
    z3 = naive_matmul(a2, theta2.T.copy())
    h = naive_sigmoid(z3)
    return argmax_rows(h) + 1

if __name__ == "__main__":
    print("=" * 50)
    print("  MNIST Neural Network - Sequential (Naive)")
    print("  Math backend: numba @njit triple-loop matmul")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Training (bao gồm JIT compilation time ở iteration đầu)
    start_time = time.time()
    cost, model = gradient_descent(X_train, y_train, learning_rate=1.5, iterations=300)
    training_time = time.time() - start_time
    
    # Evaluation
    train_pred = predict(model, X_train)
    train_accuracy = np.mean(train_pred == y_train)
    
    test_pred = predict(model, X_test)
    test_accuracy = np.mean(test_pred == y_test)
    
    print(f"\nTraining accuracy: {train_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    print(f"Total training time: {training_time:.2f} seconds")
