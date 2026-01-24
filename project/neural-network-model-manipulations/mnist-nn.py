# #!/usr/bin/python
# #
# # MNIST digit recognizer
# #
# # The simple recognizer is implememted purely in Python.  The purpose of
# # this program is to present the details how to constructa simple neural
# # network for prediction from scratch:
# #
# # 1. To build a 3-layer neural network (only one hidden layer).
# # 2. To train a model with self-implemented SGD (stochastic gradient descent).
# # 3. To predict data with the trained model.
# #
# # This program is based on the exercise of Andrew Ng's machine learning
# # course on Coursera: https://www.coursera.org/learn/machine-learning
# #
# #
# # This program is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program.  If not, see <http://www.gnu.org/licenses/>.


# import numpy as np
# import math
# import scipy.io as sio


# # Structure of the 3-layer neural network.
# Input_layer_size = 400
# Hidden_layer_size = 25
# Output_layer_size = 10


# def convert_memory_ordering_f2c(array):
#     if np.isfortran(array) is True:
#         return np.ascontiguousarray(array)
#     else:
#         return array


# def load_training_data(training_file='mnistdata.mat'):
#     '''Load training data (mnistdata.mat) and return (inputs, labels).

#     inputs: numpy array with size (5000, 400).
#     labels: numpy array with size (5000, 1).

#     The training data is from Andrew Ng's exercise of the Coursera
#     machine learning course (ex4data1.mat).
#     '''
#     training_data = sio.loadmat(training_file)
#     inputs = training_data['X'].astype('f8')
#     inputs = convert_memory_ordering_f2c(inputs)
#     labels = training_data['y']
#     labels = convert_memory_ordering_f2c(labels)
#     return (inputs, labels)


# def load_weights(weight_file='mnistweights.mat'):
#     '''Load training data (mnistdata.mat) and return (inputs, labels).

#     The weights file is from Andrew Ng's exercise of the Coursera
#     machine learning course (ex4weights.mat).
#     '''
#     weights = sio.loadmat(weight_file)
#     theta1 = convert_memory_ordering_f2c(weights['Theta1'].astype('f8'))  # size: 25 entries, each has 401 numbers
#     theta2 = convert_memory_ordering_f2c(weights['Theta2'].astype('f8'))  # size: 10 entries, each has  26 numbers
#     return (theta1, theta2)


# def rand_init_weights(size_in, size_out):
#     epsilon_init = 0.12
#     return np.random.rand(size_out, 1 + size_in) * 2 * epsilon_init - epsilon_init


# def sigmoid(z):
#     return 1.0 / (1 + pow(math.e, -z))


# def sigmoid_gradient(z):
#     return sigmoid(z) * (1 - sigmoid(z))


# def cost_function(theta1, theta2, input_layer_size, hidden_layer_size, output_layer_size, inputs, labels, regular=0):
#     '''
#     Note: theta1, theta2, inputs, labels are numpy arrays:

#         theta1: (25, 401)
#         theta2: (10, 26)
#         inputs: (5000, 400)
#         labels: (5000, 1)
#     '''
#     # construct neural network
#     input_layer = np.insert(inputs, 0, 1, axis=1)  # add bias, 5000x401

#     hidden_layer = np.dot(input_layer, np.transpose(theta1))
#     hidden_layer = sigmoid(hidden_layer)
#     hidden_layer = np.insert(hidden_layer, 0, 1, axis=1)  # add bias, 5000x26

#     output_layer = np.dot(hidden_layer, np.transpose(theta2))  # 5000x10
#     output_layer = sigmoid(output_layer)
#     #print('input  layer shape: {}'.format(input_layer.shape))
#     #print('hidden layer shape: {}'.format(hidden_layer.shape))
#     #print('output layer shape: {}'.format(output_layer.shape))

#     # forward propagation: calculate cost
#     cost = 0.0
#     for training_index in xrange(len(inputs)):
#         # transform label y[i] from a number to a vector.
#         #
#         # Note:
#         #   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         #    1  2  3  4  5  6  7  8  9 10
#         #
#         #   if y[i] is 0 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
#         #   if y[i] is 1 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#         outputs = [0] * output_layer_size
#         outputs[labels[training_index]-1] = 1

#         for k in xrange(output_layer_size):
#             cost += -outputs[k] * math.log(output_layer[training_index][k]) - (1 - outputs[k]) * math.log(1 - output_layer[training_index][k])
#     cost /= len(inputs)

#     # back propagation: calculate gradiants
#     theta1_grad = np.zeros_like(theta1)  # 25x401
#     theta2_grad = np.zeros_like(theta2)  # 10x26
#     for index in xrange(len(inputs)):
#         # transform label y[i] from a number to a vector.
#         outputs = np.zeros((1, output_layer_size))  # (1,10)
#         outputs[0][labels[index]-1] = 1

#         # calculate delta3
#         delta3 = (output_layer[index] - outputs).T  # (10,1)

#         # calculate delta2
#         z2 = np.dot(theta1, input_layer[index:index+1].T)  # (25,401) x (401,1)
#         z2 = np.insert(z2, 0, 1, axis=0)  # add bias, (26,1)
#         delta2 = np.multiply(
#             np.dot(theta2.T, delta3),  # (26,10) x (10,1)
#             sigmoid_gradient(z2)       # (26,1)
#         )
#         delta2 = delta2[1:]  # (25,1)

#         # calculate gradients of theta1 and theta2
#         # (25,401) = (25,1) x (1,401)
#         theta1_grad += np.dot(delta2, input_layer[index:index+1])
#         # (10,26) = (10,1) x (1,26)
#         theta2_grad += np.dot(delta3, hidden_layer[index:index+1])
#     theta1_grad /= len(inputs)
#     theta2_grad /= len(inputs)

#     return cost, (theta1_grad, theta2_grad)


# def gradient_descent(inputs, labels, learningrate=0.8, iteration=50):
#     '''
#     @return cost and trained model (weights).
#     '''
#     rand_theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
#     rand_theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
#     theta1 = rand_theta1
#     theta2 = rand_theta2
#     cost = 0.0
#     for i in xrange(iteration):
#         cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
#             Input_layer_size, Hidden_layer_size, Output_layer_size,
#             inputs, labels, regular=0)
#         theta1 -= learningrate * theta1_grad
#         theta2 -= learningrate * theta2_grad
#         print('Iteration {0} (learning rate {2}, iteration {3}), cost: {1}'.format(i+1, cost, learningrate, iteration))
#     return cost, (theta1, theta2)


# def train(inputs, labels, learningrate=0.8, iteration=50):
#     cost, model = gradient_descent(inputs, labels, learningrate, iteration)
#     return model


# def predict(model, inputs):
#     theta1, theta2 = model
#     a1 = np.insert(inputs, 0, 1, axis=1)  # add bias, (5000,401)
#     a2 = np.dot(a1, theta1.T)  # (5000,401) x (401,25)
#     a2 = sigmoid(a2)
#     a2 = np.insert(a2, 0, 1, axis=1)  # add bias, (5000,26)
#     a3 = np.dot(a2, theta2.T)  # (5000,26) x (26,10)
#     a3 = sigmoid(a3)  # (5000,10)
#     return [i.argmax()+1 for i in a3]


# if __name__ == '__main__':
#     # Note: There are 10 units which present the digits [1-9, 0]
#     # (in order) in the output layer.
#     inputs, labels = load_training_data()

#     # train the model from scratch and predict based on it
#     # learning rate 0.10, iteration  60: 36% (cost: 3.217)
#     # learning rate 1.75, iteration  50: 77%
#     # learning rate 1.90, iteration  50: 75%
#     # learning rate 2.00, iteration  50: 82%
#     # learning rate 2.00, iteration 100: 87%
#     # learning rate 2.00, iteration 200: 93% (cost: 0.572)
#     # learning rate 2.00, iteration 300: 94% (cost: 0.485)
#     # learning rate 2.05, iteration  50: 79%
#     # learning rate 2.20, iteration  50: 64%
#     model = train(inputs, labels, learningrate=0.1, iteration=60)

#     # Load pretrained weights for debugging precision.
#     # The precision will be around 97% (0.9756).
#     #weights = load_weights()
#     #theta1 = weights[0]  # size: 25 entries, each has 401 numbers
#     #theta2 = weights[1]  # size: 10 entries, each has  26 numbers
#     #model = (theta1, theta2)
#     #cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2, Input_layer_size, Hidden_layer_size, Output_layer_size, inputs, labels, regular=0)
#     #print('cost:', cost)

#     outputs = predict(model, inputs)

#     correct_prediction = 0
#     for i, predict in enumerate(outputs):
#         if predict == labels[i][0]:
#             correct_prediction += 1
#     precision = float(correct_prediction) / len(labels)
#     print('precision: {}'.format(precision))
import functools
import numpy as np
import math
import os
import scipy.io as sio
import time
from mpi4py import MPI

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

# Vector hóa sigmoid để tăng hiệu suất
sigmoid = np.vectorize(lambda z: 1.0 / (1 + np.exp(-z)))
sigmoid_gradient = np.vectorize(lambda z: sigmoid(z) * (1 - sigmoid(z)))

def load_training_data(training_file='mnistdata.mat'):
    training_data = sio.loadmat(training_file)
    inputs = convert_memory_ordering_f2c(training_data['X'].astype('float32'))
    labels = convert_memory_ordering_f2c(training_data['y'].ravel())
    return inputs, labels

def rand_init_weights(size_in, size_out):
    epsilon_init = np.sqrt(6) / np.sqrt(size_in + size_out)  # Khởi tạo Xavier
    return np.random.randn(size_out, size_in + 1) * epsilon_init

def cost_function(theta1, theta2, inputs, labels, regular=0.01):
    m = len(inputs)
    
    # Thêm bias và forward propagation
    a1 = np.hstack([np.ones((m, 1)), inputs])  # (m, 401)
    z2 = np.dot(a1, theta1.T)                  # (m, 25)
    a2 = sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2])     # (m, 26)
    z3 = np.dot(a2, theta2.T)                  # (m, 10)
    h = sigmoid(z3)
    
    # Chuyển labels thành ma trận one-hot
    y_matrix = np.eye(Output_layer_size)[labels-1]
    
    # Tính cost với regularization
    cost = (-1/m) * np.sum(y_matrix * np.log(h) + (1 - y_matrix) * np.log(1 - h))
    reg = (regular/(2*m)) * (np.sum(np.square(theta1[:, 1:])) + np.sum(np.square(theta2[:, 1:])))
    total_cost = cost + reg
    
    # Backpropagation
    delta3 = h - y_matrix                      # (m, 10)
    delta2 = np.dot(delta3, theta2[:, 1:]) * sigmoid_gradient(z2)  # (m, 25)
    
    theta2_grad = (1/m) * np.dot(delta3.T, a2) + (regular/m) * np.hstack([np.zeros((Output_layer_size, 1)), theta2[:, 1:]])
    theta1_grad = (1/m) * np.dot(delta2.T, a1) + (regular/m) * np.hstack([np.zeros((Hidden_layer_size, 1)), theta1[:, 1:]])
    
    return total_cost, (theta1_grad, theta2_grad)

def gradient_descent(inputs, labels, learning_rate=0.8, iterations=100, batch_size=100):
    # Khởi tạo weights
    if rank == 0:
        theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
        theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
    else:
        theta1 = np.zeros((Hidden_layer_size, Input_layer_size + 1))
        theta2 = np.zeros((Output_layer_size, Hidden_layer_size + 1))
    print(f"Process {rank} of {size} is running")
    # Đồng bộ weights ban đầu
    comm.Bcast(theta1, root=0)
    comm.Bcast(theta2, root=0)
    
    # Chia dữ liệu cho các process
    local_m = len(inputs) // size
    start_idx = rank * local_m
    end_idx = start_idx + local_m if rank != size-1 else len(inputs)
    local_inputs = inputs[start_idx:end_idx]
    local_labels = labels[start_idx:end_idx]
    
    print(f"Process {rank} handling {len(local_inputs)} samples")
    
    
    # Mini-batch gradient descent
    for i in range(iterations):
        t_start = time.time()
        
        # Chia mini-batches
        num_batches = max(1, len(local_inputs) // batch_size)
        batch_cost = 0
        theta1_grad_sum = np.zeros_like(theta1)
        theta2_grad_sum = np.zeros_like(theta2)
        
        # Xử lý từng batch
        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = min(batch_start + batch_size, len(local_inputs))
            batch_inputs = local_inputs[batch_start:batch_end]
            batch_labels = local_labels[batch_start:batch_end]
            
            # Tính cost và gradient cho batch
            cost, (t1_grad, t2_grad) = cost_function(theta1, theta2, batch_inputs, batch_labels)
            batch_cost += cost
            theta1_grad_sum += t1_grad
            theta2_grad_sum += t2_grad
        
        # Giảm trung bình gradients từ tất cả batches
        local_theta1_grad = theta1_grad_sum / num_batches
        local_theta2_grad = theta2_grad_sum / num_batches
        
        # Thu thập và trung bình gradients từ tất cả process
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
        local_cost = batch_cost / num_batches
        global_cost = comm.allreduce(local_cost, op=MPI.SUM) / size
        
        if rank == 0:
            print(f"Iteration {i+1}/{iterations}, Cost: {global_cost:.4f}, Time: {time.time()-t_start:.2f}s")
    
    return global_cost, (theta1, theta2)

def predict(model, inputs):
    theta1, theta2 = model
    a1 = np.hstack([np.ones((len(inputs), 1)), inputs])
    a2 = sigmoid(np.dot(a1, theta1.T))
    a2 = np.hstack([np.ones((len(a2), 1)), a2])
    h = sigmoid(np.dot(a2, theta2.T))
    return np.argmax(h, axis=1) + 1

if __name__ == "__main__":
    print(f"Running with {size} processes")
    
    # Load dữ liệu
    inputs, labels = load_training_data()
    
    # Train model
    cost, model = gradient_descent(inputs, labels, learning_rate=2, iterations=60, batch_size=100)
    
    # Dự đoán và tính độ chính xác
    if rank == 0:
        predictions = predict(model, inputs)
        accuracy = np.mean(predictions == labels)
        print(f"Training accuracy: {accuracy*100:.2f}%")