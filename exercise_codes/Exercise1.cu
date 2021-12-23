#include <iostream>

__global__ void gpu_function(const int *input1, const int *input2, int *output, unsigned int final_index) {
    unsigned int global_index = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(global_index < final_index)
        output[global_index] = input1[global_index] + input2[global_index];
}

void random_ints(int *array, unsigned int num) {
    for(int k = 0; k < num; k++)
        array[k] = rand() % 100;
}

int main() {
    unsigned int inputSize = 17;
    unsigned int blockSize = 3;  // Number of threads in each block
    unsigned int memorySize = inputSize * sizeof(int);

    int *host_ptr1, *host_ptr2, *host_ptr3; // CPU pointers declared in host
    int *device_ptr1, *device_ptr2, *device_ptr3; // GPU pointers declared in host

    // Memory allocation on CPU
    host_ptr1 = (int *)malloc(memorySize);
    host_ptr2 = (int *)malloc(memorySize);
    host_ptr3 = (int *)malloc(memorySize);

    // Memory allocation on GPU
    cudaMalloc((void **)&device_ptr1, memorySize);
    cudaMalloc((void **)&device_ptr2, memorySize);
    cudaMalloc((void **)&device_ptr3, memorySize);

    // Initialize the values of the variables
    random_ints(host_ptr1, inputSize);
    random_ints(host_ptr2, inputSize);

    // Moving inputs to GPU
    cudaMemcpy(device_ptr1, host_ptr1, memorySize, cudaMemcpyHostToDevice);
    cudaMemcpy(device_ptr2, host_ptr2, memorySize, cudaMemcpyHostToDevice);

    // Evaluating on GPU
    unsigned int numBlocks = (inputSize + blockSize - 1) / blockSize;
    unsigned int numThreads = blockSize;
    std::cout << "#Blocks: " << numBlocks << "\t#Threads: " << numThreads << std::endl;
    gpu_function<<<numBlocks,numThreads>>>(device_ptr1, device_ptr2, device_ptr3, inputSize);

    // Moving result to CPU for printing
    cudaMemcpy(host_ptr3, device_ptr3, memorySize, cudaMemcpyDeviceToHost);
    for(int k = 0; k < inputSize; k++)
        std::cout << host_ptr1[k] << " + " << host_ptr2[k] <<" = " << host_ptr3[k] << std::endl;

    // Finalizing the code by freed all the pointers
    free(host_ptr1); free(host_ptr2); free(host_ptr3);
    cudaFree(device_ptr1); cudaFree(device_ptr2); cudaFree(device_ptr3);

    std::cout << "EOF" << std::endl;
    return 0;
}
