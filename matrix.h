#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>

class NNMatrix {
    public:
        std::vector<std::vector<float>> data;

        NNMatrix() = default;
        NNMatrix(size_t r, size_t c, float initial_val = 0.0f);
        NNMatrix(std::vector<std::vector<float>>& input_data);
        
        size_t get_rows() {return rows;}
        size_t get_cols() {return cols;}

        NNMatrix transpose() const;
        static NNMatrix create_random(size_t r, size_t c, float min_val, float max_val);

        NNMatrix operator*(const NNMatrix& rhs) const;
        std::vector<float> operator*(const std::vector<float>& vec) const;
        NNMatrix operator+(const std::vector<float>& bias) const;
        NNMatrix operator+(const NNMatrix& rhs) const;

        void print() const;

    private:
        size_t rows, cols;
};

inline std::vector<float> operator+(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    if(lhs.size() != rhs.size()) {
        throw std::invalid_argument("Vector dimensions must match for addition.");
    }

    std::vector<float> result(lhs.size());

    for(size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    
    return result;
}
