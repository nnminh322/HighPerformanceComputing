#!/usr/bin/env python3
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import os
import sys

def convert_mat_to_csv(mat_file, output_dir='.'):
    print(f"Loading {mat_file}...")
    data = sio.loadmat(mat_file)
    inputs = np.ascontiguousarray(data['X'].astype('float64'))
    labels = np.ascontiguousarray(data['y'].ravel())
    
    print(f"Data shape: inputs={inputs.shape}, labels={labels.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    train_file = os.path.join(output_dir, 'mnist_train.csv')
    with open(train_file, 'w') as f:
        for i in range(len(X_train)):
            f.write(str(int(y_train[i])) + ',' + ','.join(map(str, X_train[i])) + '\n')
    
    test_file = os.path.join(output_dir, 'mnist_test.csv')
    with open(test_file, 'w') as f:
        for i in range(len(X_test)):
            f.write(str(int(y_test[i])) + ',' + ','.join(map(str, X_test[i])) + '\n')
    
    print(f"Created: {train_file} ({len(X_train)} samples), {test_file} ({len(X_test)} samples)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert_data.py <mat_file> [output_dir]")
        sys.exit(1)
    mat_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'
    convert_mat_to_csv(mat_file, output_dir)
