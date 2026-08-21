#include "neural_network.h"
#include <cmath>
#include <stdexcept>
#include "matrix.h"

NeuralNetwork::NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size, float weight_init_range)
    : num_inputs(input_size), num_hidden(hidden_size), num_output(output_size) {

    weights_ih = NNMatrix::create_random(num_hidden, num_inputs, -weight_init_range, +weight_init_range);
    weights_ho = NNMatrix::create_random(num_output, num_hidden, -weight_init_range, +weight_init_range);
    bias_h = std::vector<float>(num_hidden, 0.0f); 
    bias_o = std::vector<float>(num_output, 0.0f);
}


float NeuralNetwork::fast_sigmoid(float raw_z) {
    return 0.5f * (raw_z / (1.0f + std::abs(raw_z)) + 1.0f);
}

float NeuralNetwork::fast_sigmoid_derivative(float raw_z) {
    float denom = 1.0f + std::abs(raw_z);
    return 1.0f / (denom * denom);
}

std::vector<float> NeuralNetwork::fast_sigmoid(std::vector<float> raw_z) {
    std::vector<float> output = {};

    for(size_t i = 0; i < raw_z.size(); i++) {
        output.push_back(fast_sigmoid(raw_z[i]));
    }

    return output;
}

std::vector<float> NeuralNetwork::fast_sigmoid_derivative(std::vector<float> raw_z) {
    std::vector<float> output = {};

    for(size_t i = 0; i < raw_z.size(); i++) {
        output.push_back(fast_sigmoid_derivative(raw_z[i]));
    }

    return output;
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& inputs) {
    if(inputs.size() != num_inputs) {
        throw std::invalid_argument("Input size does not match the number of inputs!");
    }

    last_inputs = inputs;

    z_hidden = (weights_ih * last_inputs) + bias_h;
    a_hidden = fast_sigmoid(z_hidden);

    z_output = (weights_ho * a_hidden) + bias_o;
    a_output = fast_sigmoid(z_output);

    return a_output;
}

void NeuralNetwork::backward(const std::vector<float>& delta_o, float learning_rate) {
    std::vector<float> delta_h(num_hidden, 0.0f);

    for (size_t i = 0; i < num_hidden; i++) {
        auto error_sum = 0.0f;

        for (size_t j = 0; j < num_output; j++) {
            error_sum += weights_ho.data[j][i] * delta_o[j];
        }

        delta_h[i] = error_sum * fast_sigmoid_derivative(z_hidden[i]);
    }

    for (size_t i = 0; i < num_output; i++) {
        for (size_t j = 0; j < num_hidden; j++) {
            weights_ho.data[i][j] -= learning_rate * delta_o[i] * a_hidden[j];
        }

        bias_o[i] -= learning_rate * delta_o[i];
    }

    for (size_t j = 0; j < num_hidden; j++) {
        for (size_t k = 0; k < num_inputs; k++) { // Fixed: num_inputs limit
            weights_ih.data[j][k] -= learning_rate * delta_h[j] * last_inputs[k];
        }
        
        bias_h[j] -= learning_rate * delta_h[j];
    }
}

std::vector<float> NeuralNetwork::get_flat_weights() const {
    std::vector<float> flat_weights;

    size_t total_params = (num_hidden * num_inputs) + num_hidden + 
                         (num_output * num_hidden) + num_output;
    flat_weights.reserve(total_params);

    for(size_t i = 0; i < num_hidden; i++) {
        for(size_t j = 0; j < num_inputs; j++) {
            flat_weights.push_back(weights_ih.data[i][j]);
        }
    }

    for (size_t i = 0; i < num_hidden; ++i) {
        flat_weights.push_back(bias_h[i]);
    }

    for(size_t i = 0; i < num_output; i++) {
        for(size_t j = 0; j < num_hidden; j++) {
            flat_weights.push_back(weights_ho.data[i][j]);
        }
    }

    for (size_t i = 0; i < num_output; ++i) {
        flat_weights.push_back(bias_o[i]);
    }

    return flat_weights;
}

void NeuralNetwork::set_flat_weights(const std::vector<float>& flat_weights) {
    size_t expected_size = (num_hidden * num_inputs) + num_hidden + 
                          (num_output * num_hidden) + num_output;
    
    if(flat_weights.size() != expected_size) {
        throw std::invalid_argument("Flat weight vector size does not match network topology.");
    }

    size_t index = 0;

    for(size_t i = 0; i < num_hidden; i++) {
        for(size_t j = 0; j < num_inputs; j++) {
            weights_ih.data[i][j] = flat_weights[index++];
        }
    }

    for (size_t i = 0; i < num_hidden; ++i) {
        bias_h[i] = flat_weights[index++];
    }

    for (size_t i = 0; i < num_output; ++i) {
        for (size_t j = 0; j < num_hidden; ++j) {
            weights_ho.data[i][j] = flat_weights[index++];
        }
    }

    for (size_t i = 0; i < num_output; ++i) {
        bias_o[i] = flat_weights[index++];
    }
}


void PolicyGradient::train(const std::vector<bool>& actions_taken, float reward, float learning_rate) {
    if (actions_taken.size() != num_output) {
        throw std::invalid_argument("Actions vector size mismatch in PolicyGradient::train.");
    }

    std::vector<float> delta_o(num_output, 0.0f);
    
    for(size_t i = 0; i < num_output; i++) {
        float action_error = 0.0f;

        if(actions_taken[i]) {
            float prob = std::max(a_output[i], 1e-7f);
            action_error = -reward/prob;
        } else {
            float prob = std::max(1.0f - a_output[i], 1e-7f);
            action_error = -reward/prob;
        }

        delta_o[i] = action_error * fast_sigmoid_derivative(z_output[i]);
    }
    backward(delta_o, learning_rate);
}
