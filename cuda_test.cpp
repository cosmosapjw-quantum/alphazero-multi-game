#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "YES" : "NO") << std::endl;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    
    if (torch::cuda::is_available()) {
        int device_id = 0;  // Use first device by default
        auto device = torch::Device(torch::kCUDA, device_id);
        torch::Tensor tensor = torch::ones({2, 3}, device);
        std::cout << "Tensor on CUDA device " << device_id << ": " << tensor << std::endl;
        
#ifdef USE_CUDNN
        std::cout << "CUDNN IS ENABLED IN BUILD" << std::endl;
#else
        std::cout << "CUDNN is not enabled in build" << std::endl;
#endif
        
        // Test a simple convolution (uses cuDNN if available)
        try {
            torch::nn::Conv2d conv(torch::nn::Conv2dOptions(3, 8, 3).padding(1));
            conv->to(device);
            
            torch::Tensor input = torch::ones({1, 3, 32, 32}).to(device);
            torch::Tensor output = conv->forward(input);
            
            std::cout << "Convolution completed successfully (uses cuDNN if available)" << std::endl;
            std::cout << "Output shape: [" << output.size(0) << ", " << output.size(1) 
                     << ", " << output.size(2) << ", " << output.size(3) << "]" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in convolution test: " << e.what() << std::endl;
        }
    }
    
    return 0;
}