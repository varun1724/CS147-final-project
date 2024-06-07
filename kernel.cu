#include "kernel.h"


// Initializes the weights to a random value before training
void initialize_weights(float *weights, int size) {
    for (int i = 0; i < size; ++i) {
        weights[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}


__global__ void conv_forward(float *input, float *filter, float *bias, float *output, int input_size, int output_size, int filter_size, int num_filters) {

}


__global__ void relu_forward(float *input, float *output, int size) {

}


__global__ void pool_forward(float *input, float *output, int input_size, int output_size, int pool_size) {

}


__global__ void fc_forward(float *input, float *weights, float *bias, float *output, int input_size, int output_size) {

}


__global__ void softmax_forward(float *input, float *output, int size) {

}


__global__ void compute_loss(float *output, float *target, float *loss, int size) {

}


__global__ void conv_backward(float *d_output, float *input, float *filter, float *d_filter, float *d_bias, int input_size, int output_size, int filter_size, int num_filters) {

}


__global__ void pool_backward(float *d_output, float *input, float *d_input, int input_size, int output_size, int pool_size) {

}


__global__ void fc_backward(float *d_output, float *input, float *weights, float *d_weights, float *d_bias, int input_size, int output_size) {
    
}


// Updates weights after gradient descent is computed
__global__ void update_weights(float *weights, float *gradients, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}