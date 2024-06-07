#ifndef KERNEL_H
#define KERNEL_H

// Define the sizes of layers and other parameters
#define INPUT_SIZE 784 // Example input size (28x28 image)
#define CONV1_OUTPUT_SIZE 32*26*26 // Example output size after convolution layer
#define POOL1_OUTPUT_SIZE 32*13*13 // Example output size after pooling layer
#define FC1_OUTPUT_SIZE 128 // Example output size of fully connected layer
#define OUTPUT_SIZE 10 // Example output size for 10 classes (classification)

// Declare device pointers
float *d_input, *d_conv1_output, *d_pool1_output, *d_fc1_output, *d_output;
float *d_conv1_filter, *d_conv1_bias, *d_fc1_weights, *d_fc1_bias, *d_output_weights, *d_output_bias;
float *d_target;
float *d_loss;

void initalize_weights(float* weights, int size);
__global__ void conv_forward(float *input, float *filter, float *bias, float *output, int input_size, int output_size, int filter_size, int num_filters);
__global__ void relu_forward(float *input, float *output, int size);
__global__ void pool_forward(float *input, float *output, int input_size, int output_size, int pool_size);
__global__ void fc_forward(float *input, float *weights, float *bias, float *output, int input_size, int output_size);
__global__ void softmax_forward(float *input, float *output, int size);
__global__ void compute_loss(float *output, float *target, float *loss, int size);
__global__ void conv_backward(float *d_output, float *input, float *filter, float *d_filter, float *d_bias, int input_size, int output_size, int filter_size, int num_filters);
__global__ void pool_backward(float *d_output, float *input, float *d_input, int input_size, int output_size, int pool_size);
__global__ void fc_backward(float *d_output, float *input, float *weights, float *d_weights, float *d_bias, int input_size, int output_size);
__global__ void update_weights(float *weights, float *gradients, float learning_rate, int size);

#endif