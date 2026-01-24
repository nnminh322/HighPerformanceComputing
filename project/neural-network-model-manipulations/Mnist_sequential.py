import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import time

# Cấu trúc mạng nơ-ron
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10

SEED = 42  
np.random.seed(SEED)

# Hàm tiện ích
def convert_memory_ordering_f2c(array):
    return np.ascontiguousarray(array) if np.isfortran(array) else array

# Vector hóa sigmoid để tăng hiệu suất
sigmoid = np.vectorize(lambda z: 1.0 / (1 + np.exp(-z)))
sigmoid_gradient = np.vectorize(lambda z: sigmoid(z) * (1 - sigmoid(z)))

def load_and_split_data(training_file='mnistdata.mat', test_size=0.2, random_state=42):
    """Tải và chia dữ liệu MNIST thành tập train/test."""
    training_data = sio.loadmat(training_file)
    inputs = convert_memory_ordering_f2c(training_data['X'].astype('float32'))
    labels = convert_memory_ordering_f2c(training_data['y'].ravel())
    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def rand_init_weights(size_in, size_out):
    """Khởi tạo weights ngẫu nhiên bằng phương pháp Xavier."""
    epsilon_init = np.sqrt(6) / np.sqrt(size_in + size_out)
    return np.random.randn(size_out, size_in + 1) * epsilon_init

def cost_function(theta1, theta2, inputs, labels, regular=0.01):
    """Tính hàm mất mát và gradient."""
    m = len(inputs)
    
    # Forward propagation
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

def gradient_descent(inputs, labels, learning_rate=1, iterations=50):
    """Huấn luyện mạng nơ-ron bằng gradient descent."""
    # Khởi tạo weights
    theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
    theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
    
    # Gradient descent
    for i in range(iterations):
        cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2, inputs, labels)
        theta1 -= learning_rate * theta1_grad
        theta2 -= learning_rate * theta2_grad
        print(f"Iteration {i+1}/{iterations}, Cost: {cost:.4f}")
    
    return cost, (theta1, theta2)

def predict(model, inputs):
    """Dự đoán nhãn từ inputs."""
    theta1, theta2 = model
    a1 = np.hstack([np.ones((len(inputs), 1)), inputs])
    a2 = sigmoid(np.dot(a1, theta1.T))
    a2 = np.hstack([np.ones((len(a2), 1)), a2])
    h = sigmoid(np.dot(a2, theta2.T))
    return np.argmax(h, axis=1) + 1

if __name__ == "__main__":
    # Load và chia dữ liệu
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Đo thời gian huấn luyện
    start_time = time.time()
    
    # Huấn luyện mô hình
    cost, model = gradient_descent(X_train, y_train, learning_rate=1.5, iterations=300)
    
    # Tính thời gian huấn luyện
    training_time = time.time() - start_time
    
    # Đánh giá mô hình
    train_pred = predict(model, X_train)
    train_accuracy = np.mean(train_pred == y_train)
    
    test_pred = predict(model, X_test)
    test_accuracy = np.mean(test_pred == y_test)
    
    # In kết quả
    print(f"\nTraining accuracy: {train_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    print(f"Total training time: {training_time:.2f} seconds")