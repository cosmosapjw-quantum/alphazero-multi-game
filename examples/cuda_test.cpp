#include <alphazero/AlphaZero.h>
#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "Testing CUDA and LibTorch functionality" << std::endl;
    
    // Initialize AlphaZero (this will print CUDA availability)
    alphazero::initialize();
    
    // Create a simple tensor operation to test CUDA
    if (torch::cuda::is_available()) {
        // Create a tensor on GPU
        torch::Tensor tensor = torch::ones({3, 3}, torch::kCUDA);
        std::cout << "Created tensor on CUDA device" << std::endl;
        std::cout << tensor << std::endl;
        
        // Perform a simple operation
        torch::Tensor result = tensor * 2.0;
        std::cout << "Result of operation:" << std::endl;
        std::cout << result << std::endl;
    } else {
        std::cout << "Skipping CUDA tensor test since CUDA is not available" << std::endl;
    }
    
    std::cout << "Test completed successfully!" << std::endl;
    
    return 0;
} 