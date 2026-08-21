#include "matrix.h"
#include <random>

NNMatrix::NNMatrix(size_t r, size_t c, float initial_val) : rows(r), cols(c), data(r, std::vector<float>(c, initial_val)) {}

NNMatrix::NNMatrix(std::vector<std::vector<float>>& input_data) {
    if (input_data.empty()) {
        rows = 0;
        cols = 0;
        return;
    }

    rows = input_data.size();
    cols = input_data[0].size();

    data = input_data;
}

NNMatrix NNMatrix::transpose() const {
    NNMatrix result(cols, rows, 0.0f);

    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            result.data[j][i] = data[i][j];
        }
    }
    
    return result;
}

NNMatrix NNMatrix::create_random(size_t r, size_t c, float min_val, float max_val) {
    NNMatrix mat(r, c, 0.0f);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);

    // TODO: Populate mat.data with uniform random floats between min_val and max_val
    for(size_t i = 0; i < r; i++) {
        for(size_t j = 0; j < c; j++) {
            mat.data[i][j] = dist(rng);
        }
    }
    
    return mat;
}

NNMatrix NNMatrix::operator*(const NNMatrix& rhs) const {
    if (this->cols != rhs.rows) {
        throw std::invalid_argument("Matrix dimension mismatch for multiplication!");
    }

    NNMatrix result(rows, rhs.cols, 0.0f);

    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < rhs.cols; j++) {
            for(size_t k = 0; k < cols; k++) {
                result.data[i][j] += data[i][k] * rhs.data[k][j];
            }
        }
    }

    return result;
}

std::vector<float> NNMatrix::operator*(const std::vector<float>& vec) const {
    if (this->cols != vec.size()) {
        throw std::invalid_argument("Matrix-Vector dimension mismatch!");
    }

    std::vector<float> result(rows, 0.0f);

    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            result[i] += data[i][j] * vec[j];
        }
    }

    return result;
}

NNMatrix NNMatrix::operator+(const std::vector<float>& bias) const {
    if (rows != bias.size()) {
        throw std::invalid_argument("Bias vector size must match NNMatrix rows!");
    }

    NNMatrix result(rows, cols, 0.0f);

    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            result.data[i][j] += data[i][j] + bias[i];
        }
    }

    return result;
}

NNMatrix NNMatrix::operator+(const NNMatrix& rhs) const {
    if (rows != rhs.rows || cols != rhs.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition!");
    }

    NNMatrix result(rows, cols, 0.0f);

    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + rhs.data[i][j];
        }
    }

    return result;
}

void NNMatrix::print() const {
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[ ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << data[i][j] << " ";
        }
        std::cout << "]\n";
    }
}
