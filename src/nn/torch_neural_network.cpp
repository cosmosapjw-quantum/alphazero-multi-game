// src/nn/torch_neural_network.cpp
#include "alphazero/nn/torch_neural_network.h"
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>

namespace alphazero {
namespace nn {

TorchNeuralNetwork::TorchNeuralNetwork(
    const std::string& modelPath, 
    core::GameType gameType,
    int boardSize,
    bool useGpu
) : gameType_(gameType),
    boardSize_(boardSize),
    debugMode_(false),
    avgInferenceTimeMs_(0.0f),
    batchSize_(16),
    stopBatchThread_(false) {
    
    // Set device
    isGpu_ = useGpu && torch::cuda::is_available();
    device_ = isGpu_ ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
    
    // Set default board size if not specified
    if (boardSize_ <= 0) {
        switch (gameType_) {
            case core::GameType::GOMOKU:
                boardSize_ = 15;
                break;
            case core::GameType::CHESS:
                boardSize_ = 8;
                break;
            case core::GameType::GO:
                boardSize_ = 19;
                break;
            default:
                boardSize_ = 15;
                break;
        }
    }
    
    // Set input channels based on game type
    switch (gameType_) {
        case core::GameType::GOMOKU:
            inputChannels_ = 8;  // Current player stones, opponent stones, some history, and auxiliary channels
            actionSpaceSize_ = boardSize_ * boardSize_;
            break;
        case core::GameType::CHESS:
            inputChannels_ = 14;  // 6 piece types x 2 colors + auxiliary channels
            actionSpaceSize_ = 64 * 73;  // 64 squares, 73 possible moves per square (max)
            break;
        case core::GameType::GO:
            inputChannels_ = 8;  // Current player stones, opponent stones, some history, and auxiliary channels
            actionSpaceSize_ = boardSize_ * boardSize_ + 1;  // +1 for pass move
            break;
        default:
            throw std::runtime_error("Unsupported game type");
    }
    
    // Load model if path is provided, otherwise create a new one
    if (!modelPath.empty()) {
        try {
            // Load model from file
            model_ = torch::jit::load(modelPath);
            model_.to(device_);
            model_.eval();
            
            if (debugMode_) {
                std::cout << "Loaded model from: " << modelPath << std::endl;
            }
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading model: " + std::string(e.what()));
        }
    } else {
        // Create a new model
        createModel();
    }
    
    // Start batch processing thread
    batchThread_ = std::thread(&TorchNeuralNetwork::batchProcessingLoop, this);
}

TorchNeuralNetwork::~TorchNeuralNetwork() {
    // Stop batch processing thread
    {
        std::unique_lock<std::mutex> lock(batchMutex_);
        stopBatchThread_ = true;
        batchCondVar_.notify_all();
    }
    
    if (batchThread_.joinable()) {
        batchThread_.join();
    }
}

std::pair<std::vector<float>, float> TorchNeuralNetwork::predict(const core::IGameState& state) {
    // Convert state to tensor
    torch::Tensor input = stateTensor(state);
    
    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Forward pass
    torch::NoGradGuard no_grad;
    torch::jit::IValue output;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        output = model_.forward({input});
    }
    
    // Measure inference time
    auto end = std::chrono::high_resolution_clock::now();
    float inferenceTime = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Update average inference time
    {
        std::lock_guard<std::mutex> lock(mutex_);
        avgInferenceTimeMs_ = (avgInferenceTimeMs_ * 0.95f) + (inferenceTime * 0.05f);
    }
    
    // Process output
    return processOutput(output, state.getActionSpaceSize());
}

void TorchNeuralNetwork::predictBatch(
    const std::vector<std::reference_wrapper<const core::IGameState>>& states,
    std::vector<std::vector<float>>& policies,
    std::vector<float>& values
) {
    if (states.empty()) {
        policies.clear();
        values.clear();
        return;
    }
    
    // Prepare input tensors
    std::vector<torch::Tensor> inputTensors;
    inputTensors.reserve(states.size());
    
    for (const auto& state : states) {
        inputTensors.push_back(stateTensor(state.get()));
    }
    
    // Stack tensors into a batch
    torch::Tensor batchInput = torch::stack(inputTensors).to(device_);
    
    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Forward pass
    torch::NoGradGuard no_grad;
    torch::jit::IValue output;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        output = model_.forward({batchInput});
    }
    
    // Measure inference time
    auto end = std::chrono::high_resolution_clock::now();
    float inferenceTime = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Update average inference time
    {
        std::lock_guard<std::mutex> lock(mutex_);
        avgInferenceTimeMs_ = (avgInferenceTimeMs_ * 0.95f) + (inferenceTime * 0.05f / states.size());
    }
    
    // Process output for each state
    auto outputTuple = output.toTuple();
    auto policyBatch = outputTuple->elements()[0].toTensor();
    auto valueBatch = outputTuple->elements()[1].toTensor();
    
    // Resize output vectors
    policies.resize(states.size());
    values.resize(states.size());
    
    // Convert to std::vector
    for (size_t i = 0; i < states.size(); ++i) {
        auto policyTensor = policyBatch[i];
        auto valueTensor = valueBatch[i];
        
        // Convert policy to vector
        int actionSize = states[i].get().getActionSpaceSize();
        policies[i].resize(actionSize);
        
        if (policyTensor.size(0) == actionSize) {
            // Direct copy
            std::memcpy(policies[i].data(), policyTensor.data_ptr<float>(), 
                       actionSize * sizeof(float));
        } else {
            // Resize or truncate if needed
            int copySize = std::min(actionSize, static_cast<int>(policyTensor.size(0)));
            std::memcpy(policies[i].data(), policyTensor.data_ptr<float>(), 
                       copySize * sizeof(float));
            
            // Fill the rest with zeros
            for (int j = copySize; j < actionSize; ++j) {
                policies[i][j] = 0.0f;
            }
        }
        
        // Convert value to float
        values[i] = valueTensor.item<float>();
    }
}

std::future<std::pair<std::vector<float>, float>> TorchNeuralNetwork::predictAsync(
    const core::IGameState& state
) {
    // Create promise
    std::promise<std::pair<std::vector<float>, float>> promise;
    std::future<std::pair<std::vector<float>, float>> future = promise.get_future();
    
    // Convert state to tensor
    torch::Tensor input = stateTensor(state);
    
    // Add to batch queue
    {
        std::unique_lock<std::mutex> lock(batchMutex_);
        batchQueue_.push({input, std::move(promise)});
        batchCondVar_.notify_one();
    }
    
    return future;
}

bool TorchNeuralNetwork::isGpuAvailable() const {
    return isGpu_;
}

std::string TorchNeuralNetwork::getDeviceInfo() const {
    std::ostringstream oss;
    
    if (isGpu_) {
        oss << "GPU: " << torch::cuda::get_device_name(device_.index());
    } else {
        oss << "CPU";
    }
    
    return oss.str();
}

float TorchNeuralNetwork::getInferenceTimeMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return avgInferenceTimeMs_;
}

int TorchNeuralNetwork::getBatchSize() const {
    return batchSize_;
}

std::string TorchNeuralNetwork::getModelInfo() const {
    std::ostringstream oss;
    
    oss << "Game: ";
    switch (gameType_) {
        case core::GameType::GOMOKU:
            oss << "Gomoku";
            break;
        case core::GameType::CHESS:
            oss << "Chess";
            break;
        case core::GameType::GO:
            oss << "Go";
            break;
        default:
            oss << "Unknown";
            break;
    }
    
    oss << ", Board size: " << boardSize_;
    oss << ", Input channels: " << inputChannels_;
    oss << ", Action space: " << actionSpaceSize_;
    
    return oss.str();
}

size_t TorchNeuralNetwork::getModelSizeBytes() const {
    // This is an approximation of model size
    size_t size = 0;
    
    // Iterate through model parameters
    for (const auto& param : model_.parameters()) {
        size += param.nbytes();
    }
    
    return size;
}

void TorchNeuralNetwork::benchmark(int numIterations, int batchSize) {
    // Create a dummy state
    std::unique_ptr<core::IGameState> state = core::createGameState(gameType_, boardSize_);
    
    // Single inference benchmark
    {
        std::cout << "Single inference benchmark:" << std::endl;
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            predict(*state);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < numIterations; ++i) {
            predict(*state);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        float totalTimeMs = std::chrono::duration<float, std::milli>(end - start).count();
        
        std::cout << "  Average time: " << (totalTimeMs / numIterations) << " ms" << std::endl;
    }
    
    // Batch inference benchmark
    {
        std::cout << "Batch inference benchmark (batch size = " << batchSize << "):" << std::endl;
        
        // Prepare states
        std::vector<std::reference_wrapper<const core::IGameState>> states;
        states.reserve(batchSize);
        
        for (int i = 0; i < batchSize; ++i) {
            states.push_back(std::cref(*state));
        }
        
        std::vector<std::vector<float>> policies;
        std::vector<float> values;
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            predictBatch(states, policies, values);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < numIterations; ++i) {
            predictBatch(states, policies, values);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        float totalTimeMs = std::chrono::duration<float, std::milli>(end - start).count();
        
        std::cout << "  Average time: " << (totalTimeMs / numIterations) << " ms" << std::endl;
        std::cout << "  Average time per state: " << (totalTimeMs / numIterations / batchSize) << " ms" << std::endl;
    }
}

void TorchNeuralNetwork::enableDebugMode(bool enable) {
    debugMode_ = enable;
}

void TorchNeuralNetwork::printModelSummary() const {
    // Print model parameters
    std::cout << "Model summary:" << std::endl;
    std::cout << "Game type: " << static_cast<int>(gameType_) << std::endl;
    std::cout << "Board size: " << boardSize_ << std::endl;
    std::cout << "Input channels: " << inputChannels_ << std::endl;
    std::cout << "Action space size: " << actionSpaceSize_ << std::endl;
    std::cout << "Device: " << (isGpu_ ? "GPU" : "CPU") << std::endl;
    
    // Print model size
    size_t modelSizeBytes = getModelSizeBytes();
    std::cout << "Model size: " << (modelSizeBytes / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // Print inference time
    std::cout << "Average inference time: " << getInferenceTimeMs() << " ms" << std::endl;
}

torch::Tensor TorchNeuralNetwork::stateTensor(const core::IGameState& state) const {
    // Get tensor representation from state
    auto tensorRep = state.getEnhancedTensorRepresentation();
    
    // Convert to torch tensor
    std::vector<float> flatTensor;
    int channels = tensorRep.size();
    int height = tensorRep[0].size();
    int width = tensorRep[0][0].size();
    
    flatTensor.reserve(channels * height * width);
    
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                flatTensor.push_back(tensorRep[c][i][j]);
            }
        }
    }
    
    // Create tensor of shape [1, C, H, W]
    torch::Tensor tensor = torch::from_blob(flatTensor.data(), {1, channels, height, width}, 
                                           torch::kFloat32).clone();
    
    return tensor.to(device_);
}

std::pair<std::vector<float>, float> TorchNeuralNetwork::processOutput(
    const torch::jit::IValue& output, int actionSize) const {
    
    auto outputTuple = output.toTuple();
    auto policyTensor = outputTuple->elements()[0].toTensor();
    auto valueTensor = outputTuple->elements()[1].toTensor();
    
    // Convert policy to vector
    std::vector<float> policy(actionSize);
    if (policyTensor.size(1) == actionSize) {
        // Direct copy
        std::memcpy(policy.data(), policyTensor.data_ptr<float>(), actionSize * sizeof(float));
    } else {
        // Resize policy if needed
        if (debugMode_) {
            std::cerr << "Warning: Policy size mismatch. Expected " << actionSize
                     << ", got " << policyTensor.size(1) << std::endl;
        }
        
        // Copy what we can
        int copySize = std::min(actionSize, static_cast<int>(policyTensor.size(1)));
        std::memcpy(policy.data(), policyTensor.data_ptr<float>(), copySize * sizeof(float));
        
        // Fill the rest with zeros
        for (int i = copySize; i < actionSize; ++i) {
            policy[i] = 0.0f;
        }
    }
    
    // Get value
    float value = valueTensor.item<float>();
    
    return {policy, value};
}

void TorchNeuralNetwork::batchProcessingLoop() {
    while (true) {
        std::vector<torch::Tensor> batch;
        std::vector<std::promise<std::pair<std::vector<float>, float>>> promises;
        
        // Wait for batch or stop signal
        {
            std::unique_lock<std::mutex> lock(batchMutex_);
            batchCondVar_.wait(lock, [this] {
                return stopBatchThread_ || !batchQueue_.empty();
            });
            
            if (stopBatchThread_ && batchQueue_.empty()) {
                break;
            }
            
            // Collect batch
            int currentBatchSize = 0;
            while (!batchQueue_.empty() && currentBatchSize < batchSize_) {
                auto [tensor, promise] = std::move(batchQueue_.front());
                batchQueue_.pop();
                
                batch.push_back(std::move(tensor));
                promises.push_back(std::move(promise));
                
                currentBatchSize++;
            }
        }
        
        if (batch.empty()) {
            continue;
        }
        
        // Stack tensors into a batch
        torch::Tensor batchInput = torch::stack(batch).to(device_);
        
        // Forward pass
        torch::NoGradGuard no_grad;
        torch::jit::IValue output;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            output = model_.forward({batchInput});
        }
        
        // Process output for each state
        auto outputTuple = output.toTuple();
        auto policyBatch = outputTuple->elements()[0].toTensor();
        auto valueBatch = outputTuple->elements()[1].toTensor();
        
        // Fulfill promises
        for (size_t i = 0; i < batch.size(); ++i) {
            auto policyTensor = policyBatch[i];
            auto valueTensor = valueBatch[i];
            
            // Convert policy to vector
            std::vector<float> policy(actionSpaceSize_);
            std::memcpy(policy.data(), policyTensor.data_ptr<float>(), 
                       policyTensor.size(0) * sizeof(float));
            
            // Convert value to float
            float value = valueTensor.item<float>();
            
            // Fulfill promise
            promises[i].set_value({policy, value});
        }
    }
}

// Create a simple model (This would normally be done in Python and loaded via TorchScript)
void TorchNeuralNetwork::createModel() {
    if (debugMode_) {
        std::cout << "Creating a simple ResNet model for testing..." << std::endl;
    }
    
    // We should actually use TorchScript to export our model from Python
    // For now, we'll just throw an error
    throw std::runtime_error("Creating models directly in C++ is not supported. Please provide a pre-trained model.");
}

} // namespace nn
} // namespace alphazero