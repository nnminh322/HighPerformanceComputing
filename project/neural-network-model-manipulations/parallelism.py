# import functools
# import numpy as np
# import math
# import os
# import scipy.io as sio
# import time
# from mpi4py import MPI

# # Thiết lập biến môi trường
# os.environ['MNISTNN_PARALLEL'] = 'yes'

# # Khởi tạo MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

# # Cấu trúc mạng nơ-ron
# Input_layer_size = 400
# Hidden_layer_size = 25
# Output_layer_size = 10

# # Hàm tiện ích
# def convert_memory_ordering_f2c(array):
#     return np.ascontiguousarray(array) if np.isfortran(array) else array

# # Vector hóa sigmoid để tăng hiệu suất
# sigmoid = np.vectorize(lambda z: 1.0 / (1 + np.exp(-z)))
# sigmoid_gradient = np.vectorize(lambda z: sigmoid(z) * (1 - sigmoid(z)))

# def load_training_data(training_file='mnistdata.mat'):
#     training_data = sio.loadmat(training_file)
#     inputs = convert_memory_ordering_f2c(training_data['X'].astype('float32'))
#     labels = convert_memory_ordering_f2c(training_data['y'].ravel())
#     return inputs, labels

# def rand_init_weights(size_in, size_out):
#     epsilon_init = np.sqrt(6) / np.sqrt(size_in + size_out)  # Khởi tạo Xavier
#     return np.random.randn(size_out, size_in + 1) * epsilon_init

# def cost_function(theta1, theta2, inputs, labels, regular=0.01):
#     m = len(inputs)
    
#     # Thêm bias và forward propagation
#     a1 = np.hstack([np.ones((m, 1)), inputs])  # (m, 401)
#     z2 = np.dot(a1, theta1.T)                  # (m, 25)
#     a2 = sigmoid(z2)
#     a2 = np.hstack([np.ones((m, 1)), a2])     # (m, 26)
#     z3 = np.dot(a2, theta2.T)                  # (m, 10)
#     h = sigmoid(z3)
    
#     # Chuyển labels thành ma trận one-hot
#     y_matrix = np.eye(Output_layer_size)[labels-1]
    
#     # Tính cost với regularization
#     cost = (-1/m) * np.sum(y_matrix * np.log(h) + (1 - y_matrix) * np.log(1 - h))
#     reg = (regular/(2*m)) * (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))
#     total_cost = cost + reg
    
#     # Backpropagation
#     delta3 = h - y_matrix                      # (m, 10)
#     delta2 = np.dot(delta3, theta2[:, 1:]) * sigmoid_gradient(z2)  # (m, 25)
    
#     theta2_grad = (1/m) * np.dot(delta3.T, a2) + (regular/m) * np.hstack([np.zeros((Output_layer_size, 1)), theta2[:, 1:]])
#     theta1_grad = (1/m) * np.dot(delta2.T, a1) + (regular/m) * np.hstack([np.zeros((Hidden_layer_size, 1)), theta1[:, 1:]])
    
#     return total_cost, (theta1_grad, theta2_grad)

# def gradient_descent(inputs, labels, learning_rate=0.8, iterations=100, batch_size=100):
#     # Khởi tạo weights
#     if rank == 0:
#         theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
#         theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
#     else:
#         theta1 = np.zeros((Hidden_layer_size, Input_layer_size + 1))
#         theta2 = np.zeros((Output_layer_size, Hidden_layer_size + 1))
#     print(f"Process {rank} of {size} is running")
#     # Đồng bộ weights ban đầu
#     comm.Bcast(theta1, root=0)
#     comm.Bcast(theta2, root=0)
    
#     # Chia dữ liệu cho các process
#     local_m = len(inputs) // size
#     start_idx = rank * local_m
#     end_idx = start_idx + local_m if rank != size-1 else len(inputs)
#     local_inputs = inputs[start_idx:end_idx]
#     local_labels = labels[start_idx:end_idx]
    
#     print(f"Process {rank} handling {len(local_inputs)} samples")
    
    
#     # Mini-batch gradient descent
#     for i in range(iterations):
#         t_start = time.time()
        
#         # Chia mini-batches
#         num_batches = max(1, len(local_inputs) // batch_size)
#         batch_cost = 0
#         theta1_grad_sum = np.zeros_like(theta1)
#         theta2_grad_sum = np.zeros_like(theta2)
        
#         # Xử lý từng batch
#         for batch in range(num_batches):
#             batch_start = batch * batch_size
#             batch_end = min(batch_start + batch_size, len(local_inputs))
#             batch_inputs = local_inputs[batch_start:batch_end]
#             batch_labels = local_labels[batch_start:batch_end]
            
#             # Tính cost và gradient cho batch
#             cost, (t1_grad, t2_grad) = cost_function(theta1, theta2, batch_inputs, batch_labels)
#             batch_cost += cost
#             theta1_grad_sum += t1_grad
#             theta2_grad_sum += t2_grad
        
#         # Giảm trung bình gradients từ tất cả batches
#         local_theta1_grad = theta1_grad_sum / num_batches
#         local_theta2_grad = theta2_grad_sum / num_batches
        
#         # Thu thập và trung bình gradients từ tất cả process
#         global_theta1_grad = np.zeros_like(theta1)
#         global_theta2_grad = np.zeros_like(theta2)
#         comm.Allreduce(local_theta1_grad, global_theta1_grad, op=MPI.SUM)
#         comm.Allreduce(local_theta2_grad, global_theta2_grad, op=MPI.SUM)
#         global_theta1_grad /= size
#         global_theta2_grad /= size
        
#         # Cập nhật weights
#         theta1 -= learning_rate * global_theta1_grad
#         theta2 -= learning_rate * global_theta2_grad
        
#         # Tính total cost
#         local_cost = batch_cost / num_batches
#         global_cost = comm.allreduce(local_cost, op=MPI.SUM) / size
        
#         if rank == 0:
#             print(f"Iteration {i+1}/{iterations}, Cost: {global_cost:.4f}, Time: {time.time()-t_start:.2f}s")
    
#     return global_cost, (theta1, theta2)

# def predict(model, inputs):
#     theta1, theta2 = model
#     a1 = np.hstack([np.ones((len(inputs), 1)), inputs])
#     a2 = sigmoid(np.dot(a1, theta1.T))
#     a2 = np.hstack([np.ones((len(a2), 1)), a2])
#     h = sigmoid(np.dot(a2, theta2.T))
#     return np.argmax(h, axis=1) + 1

# if __name__ == "__main__":
#     print(f"Running with {size} processes")
    
#     # Load dữ liệu
#     inputs, labels = load_training_data()
    
#     # Train model
#     cost, model = gradient_descent(inputs, labels, learning_rate=2, iterations=60, batch_size=100)
    
#     # Dự đoán và tính độ chính xác
#     if rank == 0:
#         predictions = predict(model, inputs)
#         accuracy = np.mean(predictions == labels)
#         print(f"Training accuracy: {accuracy*100:.2f}%")
        
        
import functools
import numpy as np
import math
import os
import scipy.io as sio
import time
from mpi4py import MPI
from sklearn.model_selection import train_test_split

# Thiết lập biến môi trường
os.environ['MNISTNN_PARALLEL'] = 'yes'

# Khởi tạo MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Cấu trúc mạng nơ-ron
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10

# Hàm tiện ích
def convert_memory_ordering_f2c(array):
    return np.ascontiguousarray(array) if np.isfortran(array) else array

sigmoid = np.vectorize(lambda z: 1.0 / (1 + np.exp(-z)))
sigmoid_gradient = np.vectorize(lambda z: sigmoid(z) * (1 - sigmoid(z)))

def load_and_split_data(training_file='mnistdata.mat', test_size=0.2, random_state=42):
    training_data = sio.loadmat(training_file)
    inputs = convert_memory_ordering_f2c(training_data['X'].astype('float32'))
    labels = convert_memory_ordering_f2c(training_data['y'].ravel())
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def rand_init_weights(size_in, size_out):
    epsilon_init = np.sqrt(6) / np.sqrt(size_in + size_out)
    return np.random.randn(size_out, size_in + 1) * epsilon_init

def cost_function(theta1, theta2, inputs, labels, regular=0.01):
    m = len(inputs)
    a1 = np.hstack([np.ones((m, 1)), inputs])
    z2 = np.dot(a1, theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2])
    z3 = np.dot(a2, theta2.T)
    h = sigmoid(z3)
    
    y_matrix = np.eye(Output_layer_size)[labels-1]
    cost = (-1/m) * np.sum(y_matrix * np.log(h) + (1 - y_matrix) * np.log(1 - h))
    reg = (regular/(2*m)) * (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))
    total_cost = cost + reg
    
    delta3 = h - y_matrix
    delta2 = np.dot(delta3, theta2[:, 1:]) * sigmoid_gradient(z2)
    
    theta2_grad = (1/m) * np.dot(delta3.T, a2) + (regular/m) * np.hstack([np.zeros((Output_layer_size, 1)), theta2[:, 1:]])
    theta1_grad = (1/m) * np.dot(delta2.T, a1) + (regular/m) * np.hstack([np.zeros((Hidden_layer_size, 1)), theta1[:, 1:]])
    
    return total_cost, (theta1_grad, theta2_grad)

def gradient_descent(inputs, labels, learning_rate=0.8, iterations=100, batch_size=None, use_minibatch=True):
    # Đo thời gian tổng cộng
    total_start_time = time.time()
    
    # Khởi tạo weights
    if rank == 0:
        theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
        theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
    else:
        theta1 = np.zeros((Hidden_layer_size, Input_layer_size + 1))
        theta2 = np.zeros((Output_layer_size, Hidden_layer_size + 1))
    print(f"Process {rank} of {size} is running")
    # Đồng bộ weights
    comm.Bcast(theta1, root=0)
    comm.Bcast(theta2, root=0)
    
    # Chia dữ liệu cho các process
    # local_m = len(inputs) // size
    # start_idx = rank * local_m
    # end_idx = start_idx + local_m if rank != size-1 else len(inputs)
    # local_inputs = inputs[start_idx:end_idx]
    # local_labels = labels[start_idx:end_idx]
    # Thay đoạn chia dữ liệu trong gradient_descent
    if rank == 0:
        local_inputs = np.array_split(inputs, size)
        local_labels = np.array_split(labels, size)
    else:
        local_inputs = None
        local_labels = None

    local_inputs = comm.scatter(local_inputs, root=0)
    local_labels = comm.scatter(local_labels, root=0)
    # if rank == 0:
    print(f"Process {rank} handling {len(local_inputs)} samples")
    
    # Gradient descent
    for i in range(iterations):
        iter_start = time.time()
        
        if use_minibatch and batch_size is not None:
            # Mini-batch gradient descent
            num_batches = max(1, len(local_inputs) // batch_size)
            batch_cost = 0
            theta1_grad_sum = np.zeros_like(theta1)
            theta2_grad_sum = np.zeros_like(theta2)
            
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, len(local_inputs))
                batch_inputs = local_inputs[batch_start:batch_end]
                batch_labels = local_labels[batch_start:batch_end]
                
                cost, (t1_grad, t2_grad) = cost_function(theta1, theta2, batch_inputs, batch_labels)
                batch_cost += cost
                theta1_grad_sum += t1_grad
                theta2_grad_sum += t2_grad
            
            local_theta1_grad = theta1_grad_sum / num_batches
            local_theta2_grad = theta2_grad_sum / num_batches
            local_cost = batch_cost / num_batches
        else:
            # Standard gradient descent
            cost, (local_theta1_grad, local_theta2_grad) = cost_function(
                theta1, theta2, local_inputs, local_labels
            )
            local_cost = cost
        
        # Thu thập gradients
        global_theta1_grad = np.zeros_like(theta1)
        global_theta2_grad = np.zeros_like(theta2)
        comm.Allreduce(local_theta1_grad, global_theta1_grad, op=MPI.SUM)
        comm.Allreduce(local_theta2_grad, global_theta2_grad, op=MPI.SUM)
        global_theta1_grad /= size
        global_theta2_grad /= size
        
        # Cập nhật weights
        theta1 -= learning_rate * global_theta1_grad
        theta2 -= learning_rate * global_theta2_grad
        
        # Tính total cost
        global_cost = comm.allreduce(local_cost, op=MPI.SUM) / size
        
        if rank == 0:
            iter_time = time.time() - iter_start
            print(f"Iteration {i+1}/{iterations}, Cost: {global_cost:.4f}, "
                  f"Iteration Time: {iter_time:.2f}s")
    
    total_time = time.time() - total_start_time
    if rank == 0:
        print(f"Total training time: {total_time:.2f}s")
    
    return global_cost, (theta1, theta2), total_time

def predict(model, inputs):
    theta1, theta2 = model
    a1 = np.hstack([np.ones((len(inputs), 1)), inputs])
    a2 = sigmoid(np.dot(a1, theta1.T))
    a2 = np.hstack([np.ones((len(a2), 1)), a2])
    h = sigmoid(np.dot(a2, theta2.T))
    return np.argmax(h, axis=1) + 1

if __name__ == "__main__":
    print(f"Running with {size} processes")
    
    # Load và chia dữ liệu
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Train model (có thể chọn mini-batch hoặc standard GD)
    cost, model, training_time = gradient_descent(
        X_train, y_train, 
        learning_rate=2, 
        iterations=60, 
        batch_size=100,  # Set None để bỏ mini-batch
        use_minibatch=True  # False để dùng standard GD
    )
    
    # Đánh giá model
    if rank == 0:
        # Độ chính xác trên tập train
        train_pred = predict(model, X_train)
        train_accuracy = np.mean(train_pred == y_train)
        
        # Độ chính xác trên tập test
        test_pred = predict(model, X_test)
        test_accuracy = np.mean(test_pred == y_test)
        
        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")
        print(f"Total training time: {training_time:.2f}s") 