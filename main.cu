#include <iostream>
#include <cuda_runtime.h>
#include "kernel.h"

using namespace std;

void save_weights(float* d_weights, int size, const char* filename);
void save_model();
void load_weights(float* d_weights, int size, const char* filename);
void load_model();
void allocate_memory();
void free_memory();

////////////////////////////////////////
////////////////////////////////////////
// - TRAINING + PREDICTION FUNCTIONS
////////////////////////////////////////
////////////////////////////////////////

// Training function for the model
void train(float* input, float* target, int epochs, float learningRate) {
    allocate_memory();

    // Allocate memory for gradients
    float *d_conv1_filter_grad, *d_conv1_bias_grad, *d_fc1_weights_grad, *d_fc1_bias_grad, *d_output_weights_grad, *d_output_bias_grad;

    // Allocate space for host variables 
    float *h_conv1_filter = (float *)malloc(CONV1_OUTPUT_SIZE * sizeof(float));
    float *h_conv1_bias = (float *)malloc(CONV1_OUTPUT_SIZE * sizeof(float));
    float *h_fc1_weights = (float *)malloc(FC1_OUTPUT_SIZE * sizeof(float));
    float *h_fc1_bias = (float *)malloc(FC1_OUTPUT_SIZE * sizeof(float));
    float *h_output_weights = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    float *h_output_bias = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // Assign initial random weights
    initalize_weights(h_conv1_filter, CONV1_OUTPUT_SIZE);
    initalize_weights(h_conv1_bias, CONV1_OUTPUT_SIZE);
    initalize_weights(h_fc1_weights, FC1_OUTPUT_SIZE);
    initalize_weights(h_fc1_bias, FC1_OUTPUT_SIZE);
    initalize_weights(h_output_weights, OUTPUT_SIZE);
    initalize_weights(h_output_bias, OUTPUT_SIZE);

    // Copy weights to device pointers
    cudaMemcpy(d_conv1_filter, h_conv1_filter, CONV1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv1_bias, h_conv1_bias, CONV1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_weights, h_fc1_weights, FC1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_bias, h_fc1_bias, FC1_OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_weights, h_output_weights, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_bias, h_output_bias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Free host variables as they are not needed anymore
    free(h_conv1_filter);
    free(h_conv1_bias);
    free(h_fc1_weights);
    free(h_fc1_bias);
    free(h_output_weights);
    free(h_output_bias);

    // Run epochs
    for (int i = 0; i < epochs; ++i) {
        // recopy to device for each epoch
        cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_target, target, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        int block_dim = 256;

        // Convolutional layer
        int grid_dim = (CONV1_OUTPUT_SIZE + block_dim - 1) / block_dim;
        conv_forward<<<grid_dim, block_dim>>>(d_input, d_conv1_filter, d_conv1_bias, d_conv1_output, INPUT_SIZE, 26, 3, 32);

        // Pooling layer
        grid_dim = (POOL1_OUTPUT_SIZE + block_dim - 1) / block_dim;
        pool_forward<<<grid_dim, block_dim>>>(d_conv1_output, d_pool1_output, 26, 13, 2);

        // First fully connected layer
        grid_dim = (FC1_OUTPUT_SIZE + block_dim - 1) / block_dim;
        fc_forward<<<grid_dim, block_dim>>>(d_pool1_output, d_fc1_weights, d_fc1_bias, d_fc1_output, POOL1_OUTPUT_SIZE, FC1_OUTPUT_SIZE);

        // Second fully connected layer (output layer)
        grid_dim = (OUTPUT_SIZE + block_dim - 1) / block_dim;
        fc_forward<<<grid_dim, block_dim>>>(d_fc1_output, d_output_weights, d_output_bias, d_output, FC1_OUTPUT_SIZE, OUTPUT_SIZE);

        // Takes the calculated weights in d_output and performs a softmax activation on them. Returns the answer back in d_output
        // TODO - Fix this so it only takes one input for this problem, change name to softmax_activation
        softmax_forward<<<grid_dim, block_dim>>>(d_output, d_output, OUTPUT_SIZE);

        // Compute loss
        float h_loss;
        
        // initializes d_loss to 0s
        cudaMemset(d_loss, 0, sizeof(float));
        compute_loss<<<1, block_dim>>>(d_output, d_target, d_loss, OUTPUT_SIZE);
        cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

        // output loss
        cout << "Epoch: " << i << ", Loss: " << h_loss << endl;

        // Start with backwards pass now
        // Computing gradients to adjust weights

        // Gradients for output layer
        fc_backward<<<grid_dim, block_dim>>>(d_output, d_fc1_output, d_output_weights, d_output_weights_grad, d_output_bias_grad, FC1_OUTPUT_SIZE, OUTPUT_SIZE);

        // Gradients for the first fully connected layer
        grid_dim = (FC1_OUTPUT_SIZE + block_dim - 1) / block_dim;
        fc_backward<<<grid_dim, block_dim>>>(d_fc1_output, d_pool1_output, d_fc1_weights, d_fc1_weights_grad, d_fc1_bias_grad, POOL1_OUTPUT_SIZE, FC1_OUTPUT_SIZE);

        // Gradients for the pooling layer
        grid_dim = (POOL1_OUTPUT_SIZE + block_dim - 1) / block_dim;
        pool_backward<<<grid_dim, block_dim>>>(d_pool1_output, d_conv1_output, d_conv1_output, 26, 13, 2);

        // Gradients for the convolutional layer
        grid_dim = (CONV1_OUTPUT_SIZE + block_dim - 1) / block_dim;
        conv_backward<<<grid_dim, block_dim>>>(d_conv1_output, d_input, d_conv1_filter, d_conv1_filter_grad, d_conv1_bias_grad, INPUT_SIZE, 26, 3, 32);

        // Update weights for conv layer
        grid_dim = (CONV1_OUTPUT_SIZE + block_dim - 1) / block_dim;
        update_weights<<<grid_dim, block_dim>>>(d_conv1_filter, d_conv1_filter_grad, learningRate, CONV1_OUTPUT_SIZE);
        update_weights<<<grid_dim, block_dim>>>(d_conv1_bias, d_conv1_bias_grad, learningRate, CONV1_OUTPUT_SIZE);

        // Update weights for connected layer
        grid_dim = (FC1_OUTPUT_SIZE + block_dim - 1) / block_dim;
        update_weights<<<grid_dim, block_dim>>>(d_fc1_weights, d_fc1_weights_grad, learningRate, FC1_OUTPUT_SIZE);
        update_weights<<<grid_dim, block_dim>>>(d_fc1_bias, d_fc1_bias_grad, learningRate, FC1_OUTPUT_SIZE);

        // Update weights for output later
        grid_dim = (grid_dim + block_dim - 1) / block_dim;
        update_weights<<<grid_dim, block_dim>>>(d_output_weights, d_output_weights_grad, learningRate, OUTPUT_SIZE);
        update_weights<<<grid_dim, block_dim>>>(d_output_bias, d_output_bias_grad, learningRate, OUTPUT_SIZE);

    }

    // Save model and weights for prediction
    save_model();

    free_memory();

    // Free gradient variables
    free(d_conv1_filter_grad);
    free(d_conv1_bias_grad);
    free(d_fc1_weights_grad);
    free(d_fc1_bias_grad);
    free(d_output_weights_grad);
    free(d_output_bias_grad);

}


// Prediction function
void predict(float* input, float* output, int input_size) {
    allocate_memory();

    // Copy host input to device
    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Load saved model
    load_model();

     // Forward pass, no need for gradient descent here as model is already trained
    int block_dim = 256;

    // Convolutional layer
    int grid_dim = (CONV1_OUTPUT_SIZE + block_dim - 1) / block_dim;
    conv_forward<<<grid_dim, block_dim>>>(d_input, d_conv1_filter, d_conv1_bias, d_conv1_output, INPUT_SIZE, 26, 3, 32);
    
    // Pooling layer
    grid_dim = (POOL1_OUTPUT_SIZE + block_dim - 1) / block_dim;
    pool_forward<<<grid_dim, block_dim>>>(d_conv1_output, d_pool1_output, 26, 13, 2);

    // First fully connected layer aks dense layer
    grid_dim = (FC1_OUTPUT_SIZE + block_dim - 1) / block_dim;
    fc_forward<<<grid_dim, block_dim>>>(d_pool1_output, d_fc1_weights, d_fc1_bias, d_fc1_output, POOL1_OUTPUT_SIZE, FC1_OUTPUT_SIZE);

    // Output layer
    grid_dim = (OUTPUT_SIZE + block_dim - 1) / block_dim;
    fc_forward<<<grid_dim, block_dim>>>(d_fc1_output, d_output_weights, d_output_bias, d_output, FC1_OUTPUT_SIZE, OUTPUT_SIZE);

    // Softmax activation
    softmax_forward<<<grid_dim, block_dim>>>(d_output, d_output, OUTPUT_SIZE);

    // Copy the results back to the host
    cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    free_memory();

}

////////////////////////////////////////
////////////////////////////////////////
// - MAIN FUNCTION
////////////////////////////////////////
////////////////////////////////////////

int main() {
    // input is images, output is model predictions, target is the correct label of the image
    float *h_input = (float *)malloc(INPUT_SIZE * sizeof(float));
    float *h_target = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    float *h_output = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // TODO - Input training data in here with actual images
    for (int i = 0; i < INPUT_SIZE; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // TODO - Input correct labels corresponding to the input data here
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        h_target[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocating device variables
    float *d_input, *d_target;
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(float));

    // Copying host variables to device
    cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel to train model
    int epochs = 1000;
    float learning_rate = 0.01;
    train(d_input, d_target, epochs, learning_rate);

    // Launch kernel to predict from model
    // TODO - Add functionality that allows to predict on different datsets without having to retrain the model over again
    predict(h_input, h_output, INPUT_SIZE);

    // Output results 
    // TODO - Output results in a better format
    cout << "Predicted output: ";
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        cout << h_output[i] << " ";
    }
    cout << endl;

    // Free memory
    free(h_input);
    free(h_target);
    free(h_output);


    return 0;
}

////////////////////////////////////////
////////////////////////////////////////
// - HELPER FUNCTIONS
////////////////////////////////////////
////////////////////////////////////////

// Allocate memory for all device pointers
void allocate_memory() {
    // Allocate memory for device inputs and outputs
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_conv1_output, CONV1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_pool1_output, POOL1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc1_output, FC1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float));

    // Allocate memory for device weights and biases
    cudaMalloc(&d_conv1_filter, CONV1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_conv1_bias, CONV1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc1_weights, FC1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc1_bias, FC1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output_weights, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output_bias, OUTPUT_SIZE * sizeof(float));

    // Allocate memory for device target and loss
    cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
}


// Saves weights to a file
void save_weights(float* d_weights, int size, const char* filename) {
    float* h_weights = (float* )malloc(size * sizeof(float));

    // copies device weights to host pointer
    cudaMemcpy(h_weights, d_weights, size * sizeof(float), cudaMemcpyDeviceToHost);

    FILE *f = fopen(filename, "wb");
    fwrite(h_weights, sizeof(float), size, f);
    fclose(f);

    free(h_weights);
}


// Saves the model by saving the weights of each part of the model
void save_model() {
    save_weights(d_conv1_filter, CONV1_OUTPUT_SIZE, "conv1_filter.bin");
    save_weights(d_conv1_bias, CONV1_OUTPUT_SIZE, "conv1_bias.bin");
    save_weights(d_fc1_weights, FC1_OUTPUT_SIZE, "fc1_weights.bin");
    save_weights(d_fc1_bias, FC1_OUTPUT_SIZE, "fc1_bias.bin");
    save_weights(d_output_weights, OUTPUT_SIZE, "output_weights.bin");
    save_weights(d_output_bias, OUTPUT_SIZE, "output_bias.bin");
}

// Loads weights from a file and copies to device pointer
void load_weights(float* d_weights, int size, const char* filename) {
    float *h_weights = (float* )malloc(size * sizeof(float));

    FILE *f = fopen(filename, "rb");
    fread(h_weights, sizeof(float), size, f);
    fclose(f);

    // Copies read weights to device pointer
    cudaMemcpy(d_weights, h_weights, size * sizeof(float), cudaMemcpyHostToDevice);

    free(h_weights);
}

// Loads the pretrained model from the saved files
void load_model() {
    load_weights(d_conv1_filter, CONV1_OUTPUT_SIZE, "conv1_filter.bin");
    load_weights(d_conv1_bias, CONV1_OUTPUT_SIZE, "conv1_bias.bin");
    load_weights(d_fc1_weights, FC1_OUTPUT_SIZE, "fc1_weights.bin");
    load_weights(d_fc1_bias, FC1_OUTPUT_SIZE, "fc1_bias.bin");
    load_weights(d_output_weights, OUTPUT_SIZE, "output_weights.bin");
    load_weights(d_output_bias, OUTPUT_SIZE, "output_bias.bin");
}

// Free device memory
void free_memory() {
    cudaFree(d_input);
    cudaFree(d_conv1_output);
    cudaFree(d_pool1_output);
    cudaFree(d_fc1_output);
    cudaFree(d_output);
    cudaFree(d_conv1_filter);
    cudaFree(d_conv1_bias);
    cudaFree(d_fc1_weights);
    cudaFree(d_fc1_bias);
    cudaFree(d_output_weights);
    cudaFree(d_output_bias);
    cudaFree(d_target);
    cudaFree(d_loss);
}