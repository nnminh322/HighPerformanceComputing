"""
MNIST Neural Network - Parallel MPI (Naive Matmul)
Sử dụng numba @njit naive triple-loop matmul, giống hệt C++ implementation.
Mục đích: So sánh công bằng với C++ (cùng thuật toán, khác ngôn ngữ).
"""
import numpy as np
import os
import scipy.io as sio
import time
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from numba import njit
import numba

os.environ['MNISTNN_PARALLEL'] = 'yes'

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
    s = naive_sigmoid(z)
    return s * (1.0 - s)

@njit
def add_bias_column(m):
    rows, cols = m.shape
    result = np.ones((rows, cols + 1))
    result[:, 1:] = m
    return result

@njit
def argmax_rows(m):
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
    inputs = convert_memory_ordering_f2c(training_data['X'].astype('float64'))
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
    
    a1 = add_bias_column(inputs)
    z2 = naive_matmul(a1, theta1.T.copy())
    a2 = naive_sigmoid(z2)
    a2 = add_bias_column(a2)
    z3 = naive_matmul(a2, theta2.T.copy())
    h = naive_sigmoid(z3)
    
    y_matrix = np.eye(Output_layer_size)[labels.astype(np.int64) - 1]
    cost = (-1/m) * np.sum(y_matrix * np.log(h + 1e-10) + (1 - y_matrix) * np.log(1 - h + 1e-10))
    reg = (regular/(2*m)) * (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))
    total_cost = cost + reg
    
    delta3 = h - y_matrix
    delta2 = naive_matmul(delta3, theta2[:, 1:].copy()) * naive_sigmoid_gradient(z2)
    
    theta2_grad = (1/m) * naive_matmul(delta3.T.copy(), a2) + (regular/m) * np.hstack([np.zeros((Output_layer_size, 1)), theta2[:, 1:]])
    theta1_grad = (1/m) * naive_matmul(delta2.T.copy(), a1) + (regular/m) * np.hstack([np.zeros((Hidden_layer_size, 1)), theta1[:, 1:]])
    
    return total_cost, (theta1_grad, theta2_grad)

def gradient_descent(inputs, labels, learning_rate=1, iterations=100):
    total_start_time = time.time()
    
    if rank == 0:
        theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
        theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
    else:
        theta1 = np.zeros((Hidden_layer_size, Input_layer_size + 1))
        theta2 = np.zeros((Output_layer_size, Hidden_layer_size + 1))
    
    comm.Bcast(theta1, root=0)
    comm.Bcast(theta2, root=0)
    
    if rank == 0:
        local_inputs = np.array_split(inputs, size)
        local_labels = np.array_split(labels, size)
    else:
        local_inputs = None
        local_labels = None

    local_inputs = comm.scatter(local_inputs, root=0)
    local_labels = comm.scatter(local_labels, root=0)
    
    if rank == 0:
        print(f"Each process handling ~{len(local_inputs)} samples")
    
    for i in range(iterations):
        iter_start = time.time()
        
        cost, (local_theta1_grad, local_theta2_grad) = cost_function(
            theta1, theta2, local_inputs, local_labels
        )
        local_cost = cost
        
        global_theta1_grad = np.zeros_like(theta1)
        global_theta2_grad = np.zeros_like(theta2)
        comm.Allreduce(local_theta1_grad, global_theta1_grad, op=MPI.SUM)
        comm.Allreduce(local_theta2_grad, global_theta2_grad, op=MPI.SUM)
        global_theta1_grad /= size
        global_theta2_grad /= size
        
        theta1 -= learning_rate * global_theta1_grad
        theta2 -= learning_rate * global_theta2_grad
        
        global_cost = comm.allreduce(local_cost, op=MPI.SUM) / size
        
        if rank == 0 and ((i + 1) % 10 == 0 or i == 0):
            iter_time = time.time() - iter_start
            print(f"Iteration {i+1}/{iterations}, Cost: {global_cost:.4f}, "
                  f"Iter Time: {iter_time:.2f}s")
    
    total_time = time.time() - total_start_time
    if rank == 0:
        print(f"Total training time: {total_time:.2f}s")
    
    return global_cost, (theta1, theta2), total_time

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
    if rank == 0:
        print("=" * 50)
        print("  MNIST Neural Network - Parallel MPI (Naive)")
        print(f"  Processes: {size}")
        print("  Math backend: numba @njit triple-loop matmul")
        print("=" * 50)
    
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    cost, model, training_time = gradient_descent(
        X_train, y_train,
        learning_rate=1.5,
        iterations=300
    )
    
    if rank == 0:
        train_pred = predict(model, X_train)
        train_accuracy = np.mean(train_pred == y_train)
        
        test_pred = predict(model, X_test)
        test_accuracy = np.mean(test_pred == y_test)
        
        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")
        print(f"Total training time: {training_time:.2f}s")
