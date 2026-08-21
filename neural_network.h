#pragma once

#include <vector>
#include "matrix.h"

class NeuralNetwork {
    public:
        NeuralNetwork() = default;
        NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size, float weight_init_range);
        virtual ~NeuralNetwork() = default;

        static float fast_sigmoid(float raw_z);
        static float fast_sigmoid_derivative(float raw_z);

        static std::vector<float> fast_sigmoid(std::vector<float> raw_z);
        static std::vector<float> fast_sigmoid_derivative(std::vector<float> raw_z);

        std::vector<float> forward(const std::vector<float>& inputs);
        virtual void backward(const std::vector<float>& delta_o, float learning_rate);

        std::vector<float> get_flat_weights() const;
        void set_flat_weights(const std::vector<float>& flat_weights);

    protected:
        int num_inputs;
        int num_hidden;
        int num_output;

        NNMatrix weights_ih;
        NNMatrix weights_ho;

        std::vector<float> bias_h;
        std::vector<float> bias_o;
        std::vector<float> last_inputs;   // Cached raw inputs (12)
        std::vector<float> z_hidden;      // Cached raw pre-activation hidden sums (16)
        std::vector<float> a_hidden;      // Cached activated hidden values (16)
        std::vector<float> z_output;      // Cached raw pre-activation output sums (4)
        std::vector<float> a_output;      // Cached final thruster probabilities (4)
};

class PolicyGradient : public NeuralNetwork {
    public:
        using NeuralNetwork::NeuralNetwork;

        void train(const std::vector<bool>& actions_taken, float reward, float learning_rate);
        void save_weights(const std::string& filename) const;
        void load_weights(const std::string& filename);
};