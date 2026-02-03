#include <mpi.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstring>
#include "matrix_utils.h"

const int INPUT_LAYER_SIZE = 400;
const int HIDDEN_LAYER_SIZE = 25;
const int OUTPUT_LAYER_SIZE = 10;
const unsigned int SEED = 42;

int rank, numProcs;

struct Gradients {
    Matrix theta1_grad;
    Matrix theta2_grad;
};

struct Model {
    Matrix theta1;
    Matrix theta2;
};

void broadcastMatrix(Matrix& mat, int root, MPI_Comm comm) {
    int r, c;
    if (rank == root) {
        r = mat.size();
        c = mat.empty() ? 0 : mat[0].size();
    }
    MPI_Bcast(&r, 1, MPI_INT, root, comm);
    MPI_Bcast(&c, 1, MPI_INT, root, comm);
    if (rank != root) mat = zeros(r, c);
    std::vector<double> flat(r * c);
    if (rank == root) flat = flatten(mat);
    MPI_Bcast(flat.data(), r * c, MPI_DOUBLE, root, comm);
    if (rank != root) mat = reshape(flat, r, c);
}

Matrix allreduceMatrix(const Matrix& localMat, MPI_Comm comm) {
    int r = localMat.size();
    int c = localMat.empty() ? 0 : localMat[0].size();
    std::vector<double> localFlat = flatten(localMat);
    std::vector<double> globalFlat(r * c, 0.0);
    MPI_Allreduce(localFlat.data(), globalFlat.data(), r * c, MPI_DOUBLE, MPI_SUM, comm);
    return reshape(globalFlat, r, c);
}

std::pair<double, Gradients> costFunction(const Matrix& theta1, const Matrix& theta2,
                                          const Matrix& inputs, const VectorInt& labels, double regular = 0.01) {
    int m = inputs.size();
    if (m == 0) {
        Gradients grads;
        grads.theta1_grad = zeros(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE + 1);
        grads.theta2_grad = zeros(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE + 1);
        return {0.0, grads};
    }
    
    Matrix a1 = addBiasColumn(inputs);
    Matrix z2 = matmul(a1, transpose(theta1));
    Matrix a2_nobias = sigmoidMatrix(z2);
    Matrix a2 = addBiasColumn(a2_nobias);
    Matrix z3 = matmul(a2, transpose(theta2));
    Matrix h = sigmoidMatrix(z3);
    
    Matrix y_matrix = zeros(m, OUTPUT_LAYER_SIZE);
    for (int i = 0; i < m; i++) {
        int label = labels[i];
        if (label >= 1 && label <= OUTPUT_LAYER_SIZE) y_matrix[i][label - 1] = 1.0;
    }
    
    Matrix log_h = logMatrix(h);
    Matrix log_1_minus_h = logMatrix(oneMinusMatrix(h));
    Matrix one_minus_y = oneMinusMatrix(y_matrix);
    
    double cost = 0.0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < OUTPUT_LAYER_SIZE; j++)
            cost += -y_matrix[i][j] * log_h[i][j] - one_minus_y[i][j] * log_1_minus_h[i][j];
    cost /= m;
    
    double reg = 0.0;
    for (size_t i = 0; i < theta1.size(); i++)
        for (size_t j = 1; j < theta1[i].size(); j++)
            reg += theta1[i][j] * theta1[i][j];
    for (size_t i = 0; i < theta2.size(); i++)
        for (size_t j = 1; j < theta2[i].size(); j++)
            reg += theta2[i][j] * theta2[i][j];
    reg = (regular / (2.0 * m)) * reg;
    double totalCost = cost + reg;
    
    Matrix delta3 = subtract(h, y_matrix);
    Matrix theta2_nobias = sliceCols(theta2, 1);
    Matrix delta2_temp = matmul(delta3, theta2_nobias);
    Matrix sig_grad_z2 = sigmoidGradientMatrix(z2);
    Matrix delta2 = elementMul(delta2_temp, sig_grad_z2);
    
    Matrix theta2_grad = scalarMul(matmul(transpose(delta3), a2), 1.0 / m);
    Matrix theta1_grad = scalarMul(matmul(transpose(delta2), a1), 1.0 / m);
    
    for (size_t i = 0; i < theta1_grad.size(); i++)
        for (size_t j = 1; j < theta1_grad[i].size(); j++)
            theta1_grad[i][j] += (regular / m) * theta1[i][j];
    for (size_t i = 0; i < theta2_grad.size(); i++)
        for (size_t j = 1; j < theta2_grad[i].size(); j++)
            theta2_grad[i][j] += (regular / m) * theta2[i][j];
    
    Gradients grads;
    grads.theta1_grad = theta1_grad;
    grads.theta2_grad = theta2_grad;
    return {totalCost, grads};
}

void scatterData(const Matrix& inputs, const VectorInt& labels, Matrix& localInputs, VectorInt& localLabels, int root) {
    int totalSamples, inputSize;
    if (rank == root) {
        totalSamples = inputs.size();
        inputSize = inputs.empty() ? INPUT_LAYER_SIZE : inputs[0].size();
    }
    MPI_Bcast(&totalSamples, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&inputSize, 1, MPI_INT, root, MPI_COMM_WORLD);
    
    std::vector<int> sendCounts(numProcs), displs(numProcs);
    int baseCount = totalSamples / numProcs;
    int remainder = totalSamples % numProcs;
    int offset = 0;
    for (int i = 0; i < numProcs; i++) {
        sendCounts[i] = baseCount + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendCounts[i];
    }
    int localSize = sendCounts[rank];
    
    std::vector<double> flatInputs;
    if (rank == root) flatInputs = flatten(inputs);
    
    std::vector<int> inputSendCounts(numProcs), inputDispls(numProcs);
    for (int i = 0; i < numProcs; i++) {
        inputSendCounts[i] = sendCounts[i] * inputSize;
        inputDispls[i] = displs[i] * inputSize;
    }
    
    std::vector<double> localFlatInputs(localSize * inputSize);
    MPI_Scatterv(flatInputs.data(), inputSendCounts.data(), inputDispls.data(), MPI_DOUBLE,
                 localFlatInputs.data(), localSize * inputSize, MPI_DOUBLE, root, MPI_COMM_WORLD);
    localInputs = reshape(localFlatInputs, localSize, inputSize);
    
    std::vector<int> allLabels;
    if (rank == root) allLabels = labels;
    localLabels.resize(localSize);
    MPI_Scatterv(allLabels.data(), sendCounts.data(), displs.data(), MPI_INT,
                 localLabels.data(), localSize, MPI_INT, root, MPI_COMM_WORLD);
}

std::tuple<double, Model, double> gradientDescentParallel(const Matrix& inputs, const VectorInt& labels,
                                                          double learningRate = 1.0, int iterations = 100) {
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    
    Matrix theta1, theta2;
    if (rank == 0) {
        theta1 = randInitWeights(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, SEED);
        theta2 = randInitWeights(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, SEED + 1);
    } else {
        theta1 = zeros(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE + 1);
        theta2 = zeros(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE + 1);
    }
    
    if (rank == 0) std::cout << "Process " << rank << " of " << numProcs << " is running (root)\n";
    
    broadcastMatrix(theta1, 0, MPI_COMM_WORLD);
    broadcastMatrix(theta2, 0, MPI_COMM_WORLD);
    
    Matrix localInputs;
    VectorInt localLabels;
    scatterData(inputs, labels, localInputs, localLabels, 0);
    
    std::cout << "Process " << rank << " handling " << localInputs.size() << " samples\n";
    
    double globalCost = 0.0;
    
    for (int i = 0; i < iterations; i++) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        
        auto [localCost, grads] = costFunction(theta1, theta2, localInputs, localLabels);
        
        Matrix globalTheta1Grad = allreduceMatrix(grads.theta1_grad, MPI_COMM_WORLD);
        Matrix globalTheta2Grad = allreduceMatrix(grads.theta2_grad, MPI_COMM_WORLD);
        
        globalTheta1Grad = scalarMul(globalTheta1Grad, 1.0 / numProcs);
        globalTheta2Grad = scalarMul(globalTheta2Grad, 1.0 / numProcs);
        
        for (size_t r = 0; r < theta1.size(); r++)
            for (size_t c = 0; c < theta1[r].size(); c++)
                theta1[r][c] -= learningRate * globalTheta1Grad[r][c];
        for (size_t r = 0; r < theta2.size(); r++)
            for (size_t c = 0; c < theta2[r].size(); c++)
                theta2[r][c] -= learningRate * globalTheta2Grad[r][c];
        
        MPI_Allreduce(&localCost, &globalCost, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        globalCost /= numProcs;
        
        if (rank == 0) {
            auto iterEnd = std::chrono::high_resolution_clock::now();
            auto iterDuration = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);
            double iterTime = iterDuration.count() / 1000.0;
            std::cout << "Iteration " << (i + 1) << "/" << iterations
                      << ", Cost: " << std::fixed << std::setprecision(4) << globalCost
                      << ", Time: " << std::fixed << std::setprecision(2) << iterTime << "s\n";
        }
    }
    
    auto totalEndTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(totalEndTime - totalStartTime);
    double totalTime = totalDuration.count() / 1000.0;
    
    if (rank == 0) std::cout << "Total training time: " << std::fixed << std::setprecision(2) << totalTime << "s\n";
    
    Model model;
    model.theta1 = theta1;
    model.theta2 = theta2;
    return {globalCost, model, totalTime};
}

VectorInt predict(const Model& model, const Matrix& inputs) {
    Matrix a1 = addBiasColumn(inputs);
    Matrix a2_nobias = sigmoidMatrix(matmul(a1, transpose(model.theta1)));
    Matrix a2 = addBiasColumn(a2_nobias);
    Matrix h = sigmoidMatrix(matmul(a2, transpose(model.theta2)));
    VectorInt predictions = argmaxRows(h);
    for (auto& p : predictions) p += 1;
    return predictions;
}

double accuracy(const VectorInt& predictions, const VectorInt& labels) {
    int correct = 0;
    for (size_t i = 0; i < predictions.size(); i++)
        if (predictions[i] == labels[i]) correct++;
    return static_cast<double>(correct) / labels.size();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    if (rank == 0) {
        std::cout << "========================================\n";
        std::cout << "  MNIST Neural Network - MPI Parallel  \n";
        std::cout << "========================================\n";
        std::cout << "Running with " << numProcs << " processes\n";
    }
    
    Matrix X_train, X_test;
    VectorInt y_train, y_test;
    
    if (rank == 0) {
        std::string trainFile = "data/mnist_train.csv";
        std::string testFile = "data/mnist_test.csv";
        
        bool dataLoaded = false;
        if (loadMNISTData(trainFile, X_train, y_train, false) && 
            loadMNISTData(testFile, X_test, y_test, false)) {
            std::cout << "\nLoaded MNIST data from CSV files\n";
            dataLoaded = true;
        }
        
        if (!dataLoaded) {
            std::cout << "\nCSV files not found, generating synthetic data...\n";
            Matrix inputs;
            VectorInt labels;
            generateSyntheticData(inputs, labels, 5000, INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE, SEED);
            trainTestSplit(inputs, labels, X_train, X_test, y_train, y_test, 0.2, SEED);
        }
        
        std::cout << "Training samples: " << X_train.size() << "\n";
        std::cout << "Test samples: " << X_test.size() << "\n";
        std::cout << "Input features: " << INPUT_LAYER_SIZE << "\n";
        std::cout << "Hidden units: " << HIDDEN_LAYER_SIZE << "\n";
        std::cout << "Output classes: " << OUTPUT_LAYER_SIZE << "\n";
    }
    
    int testSize;
    if (rank == 0) testSize = X_test.size();
    MPI_Bcast(&testSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) std::cout << "\n--- Training ---\n";
    
    auto [finalCost, model, trainingTime] = gradientDescentParallel(X_train, y_train, 1.5, 300);
    
    if (rank == 0) {
        std::cout << "\n--- Evaluation ---\n";
        
        VectorInt trainPred = predict(model, X_train);
        double trainAcc = accuracy(trainPred, y_train);
        
        VectorInt testPred = predict(model, X_test);
        double testAcc = accuracy(testPred, y_test);
        
        std::cout << "Training accuracy: " << std::fixed << std::setprecision(2) << (trainAcc * 100) << "%\n";
        std::cout << "Test accuracy: " << std::fixed << std::setprecision(2) << (testAcc * 100) << "%\n";
        std::cout << "Training time: " << std::fixed << std::setprecision(2) << trainingTime << "s\n";
        
        std::cout << "\n========================================\n";
        std::cout << "  Training Complete!                   \n";
        std::cout << "========================================\n";
    }
    
    MPI_Finalize();
    return 0;
}
