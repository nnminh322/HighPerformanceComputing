#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cassert>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;
using VectorInt = std::vector<int>;

inline void printMatrix(const Matrix& m, const std::string& name = "") {
    if (!name.empty()) std::cout << name << " (" << m.size() << "x" << (m.empty() ? 0 : m[0].size()) << "):" << std::endl;
    for (const auto& row : m) {
        for (const auto& val : row) std::cout << val << " ";
        std::cout << std::endl;
    }
}

inline Matrix zeros(int rows, int cols) { return Matrix(rows, Vector(cols, 0.0)); }
inline Matrix ones(int rows, int cols) { return Matrix(rows, Vector(cols, 1.0)); }

inline Matrix eye(int n) {
    Matrix result = zeros(n, n);
    for (int i = 0; i < n; i++) result[i][i] = 1.0;
    return result;
}

inline int rows(const Matrix& m) { return m.size(); }
inline int cols(const Matrix& m) { return m.empty() ? 0 : m[0].size(); }

inline Matrix transpose(const Matrix& m) {
    if (m.empty()) return Matrix();
    int r = m.size(), c = m[0].size();
    Matrix result(c, Vector(r));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            result[j][i] = m[i][j];
    return result;
}

inline Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty()) return Matrix();
    int m = A.size(), n = A[0].size(), p = B[0].size();
    if (n != (int)B.size()) {
        std::cerr << "Matrix multiplication error: A(" << m << "x" << n << ") * B(" << B.size() << "x" << p << ")" << std::endl;
        assert(false && "Matrix dimensions mismatch");
    }
    Matrix C(m, Vector(p, 0.0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

inline Matrix add(const Matrix& A, const Matrix& B) {
    int r = A.size(), c = A[0].size();
    Matrix C(r, Vector(c));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

inline Matrix subtract(const Matrix& A, const Matrix& B) {
    int r = A.size(), c = A[0].size();
    Matrix C(r, Vector(c));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

inline Matrix elementMul(const Matrix& A, const Matrix& B) {
    int r = A.size(), c = A[0].size();
    Matrix C(r, Vector(c));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            C[i][j] = A[i][j] * B[i][j];
    return C;
}

inline Matrix scalarMul(const Matrix& A, double scalar) {
    int r = A.size(), c = A[0].size();
    Matrix C(r, Vector(c));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            C[i][j] = A[i][j] * scalar;
    return C;
}

inline Matrix addBiasColumn(const Matrix& m) {
    int r = m.size(), c = m[0].size();
    Matrix result(r, Vector(c + 1));
    for (int i = 0; i < r; i++) {
        result[i][0] = 1.0;
        for (int j = 0; j < c; j++) result[i][j + 1] = m[i][j];
    }
    return result;
}

inline Matrix sliceCols(const Matrix& m, int start) {
    int r = m.size(), c = m[0].size() - start;
    Matrix result(r, Vector(c));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            result[i][j] = m[i][j + start];
    return result;
}

inline Matrix hstack(const Matrix& A, const Matrix& B) {
    int r = A.size(), ca = A[0].size(), cb = B[0].size();
    Matrix result(r, Vector(ca + cb));
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < ca; j++) result[i][j] = A[i][j];
        for (int j = 0; j < cb; j++) result[i][ca + j] = B[i][j];
    }
    return result;
}

inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

inline Matrix sigmoidMatrix(const Matrix& m) {
    int r = m.size(), c = m[0].size();
    Matrix result(r, Vector(c));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            result[i][j] = sigmoid(m[i][j]);
    return result;
}

inline double sigmoidGradient(double z) {
    double s = sigmoid(z);
    return s * (1.0 - s);
}

inline Matrix sigmoidGradientMatrix(const Matrix& m) {
    int r = m.size(), c = m[0].size();
    Matrix result(r, Vector(c));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            result[i][j] = sigmoidGradient(m[i][j]);
    return result;
}

inline Matrix logMatrix(const Matrix& m) {
    int r = m.size(), c = m[0].size();
    Matrix result(r, Vector(c));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            result[i][j] = std::log(std::max(m[i][j], 1e-15));
    return result;
}

inline Matrix oneMinusMatrix(const Matrix& m) {
    int r = m.size(), c = m[0].size();
    Matrix result(r, Vector(c));
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            result[i][j] = 1.0 - m[i][j];
    return result;
}

inline double sum(const Matrix& m) {
    double total = 0.0;
    for (const auto& row : m)
        for (const auto& val : row)
            total += val;
    return total;
}

inline double sumSquare(const Matrix& m) {
    double total = 0.0;
    for (const auto& row : m)
        for (const auto& val : row)
            total += val * val;
    return total;
}

inline VectorInt argmaxRows(const Matrix& m) {
    VectorInt result(m.size());
    for (size_t i = 0; i < m.size(); i++) {
        int maxIdx = 0;
        double maxVal = m[i][0];
        for (size_t j = 1; j < m[i].size(); j++) {
            if (m[i][j] > maxVal) { maxVal = m[i][j]; maxIdx = j; }
        }
        result[i] = maxIdx;
    }
    return result;
}

inline Matrix randInitWeights(int sizeIn, int sizeOut, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    double epsilonInit = std::sqrt(6.0) / std::sqrt(sizeIn + sizeOut);
    std::normal_distribution<> dis(0.0, epsilonInit);
    Matrix weights(sizeOut, Vector(sizeIn + 1));
    for (int i = 0; i < sizeOut; i++)
        for (int j = 0; j < sizeIn + 1; j++)
            weights[i][j] = dis(gen);
    return weights;
}

inline Vector flatten(const Matrix& m) {
    Vector result;
    for (const auto& row : m) result.insert(result.end(), row.begin(), row.end());
    return result;
}

inline Matrix reshape(const Vector& v, int rows, int cols) {
    Matrix result(rows, Vector(cols));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i][j] = v[i * cols + j];
    return result;
}

inline bool loadMNISTData(const std::string& filename, Matrix& inputs, VectorInt& labels, bool normalize = false) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }
    inputs.clear();
    labels.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        Vector row;
        std::getline(ss, value, ',');
        labels.push_back(std::stoi(value));
        while (std::getline(ss, value, ',')) {
            double val = std::stod(value);
            if (normalize) val /= 255.0;
            row.push_back(val);
        }
        inputs.push_back(row);
    }
    file.close();
    return true;
}

inline void generateSyntheticData(Matrix& inputs, VectorInt& labels, int numSamples = 5000, int inputSize = 400, int numClasses = 10, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> disDat(0.0, 1.0);
    std::uniform_int_distribution<> disLabel(1, numClasses);
    inputs.clear();
    labels.clear();
    for (int i = 0; i < numSamples; i++) {
        Vector row(inputSize);
        for (int j = 0; j < inputSize; j++) row[j] = disDat(gen);
        inputs.push_back(row);
        labels.push_back(disLabel(gen));
    }
}

inline void trainTestSplit(const Matrix& inputs, const VectorInt& labels,
                           Matrix& X_train, Matrix& X_test, VectorInt& y_train, VectorInt& y_test,
                           double testSize = 0.2, unsigned int seed = 42) {
    int n = inputs.size();
    int testCount = static_cast<int>(n * testSize);
    int trainCount = n - testCount;
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);
    X_train.clear(); X_test.clear(); y_train.clear(); y_test.clear();
    for (int i = 0; i < trainCount; i++) {
        X_train.push_back(inputs[indices[i]]);
        y_train.push_back(labels[indices[i]]);
    }
    for (int i = trainCount; i < n; i++) {
        X_test.push_back(inputs[indices[i]]);
        y_test.push_back(labels[indices[i]]);
    }
}

#endif
