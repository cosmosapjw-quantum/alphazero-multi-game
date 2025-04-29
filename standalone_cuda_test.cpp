#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "Standalone CUDA Test" << std::endl;
    
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
        
        // Create tensors on GPU
        int device_id = 0; // Use first device by default
        auto device = torch::Device(torch::kCUDA, device_id);
        torch::Tensor a = torch::ones({3, 3}, device);
        torch::Tensor b = torch::ones({3, 3}, device);
        
        // Perform a CUDA operation
        torch::Tensor c = a + b;
        
        // Print results
        std::cout << "Result on device " << device_id << ": " << c << std::endl;
    } else {
        std::cout << "CUDA is not available!" << std::endl;
    }
    
    return 0;
}