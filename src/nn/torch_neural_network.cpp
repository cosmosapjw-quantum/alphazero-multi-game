// src/nn/torch_neural_network.cpp
#include "alphazero/nn/torch_neural_network.h"

// Explicitly include torch/script.h for torch::jit::load
#ifndef LIBTORCH_OFF
#include <torch/script.h>
#endif

#include <iostream>  // For std::cout/cerr
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <spdlog/spdlog.h>

namespace alphazero {
namespace nn {

TorchNeuralNetwork::TorchNeuralNetwork(
    const std::string& modelPath, 
    core::GameType gameType,
    int boardSize,
    bool useGpu,
    const TorchNeuralNetworkConfig& config
) : gameType_(gameType),
    boardSize_(boardSize),
    config_(config),
    debugMode_(false),
    avgInferenceTimeMs_(0.0f),
    batchSize_(config.batchSize)
{
    
#ifndef LIBTORCH_OFF
    // Set device
    isGpu_ = useGpu && torch::cuda::is_available() && config.useGpu;
    if (isGpu_) {
        device_ = torch::Device(torch::kCUDA, 0);
    } else {
        device_ = torch::Device(torch::kCPU);
    }
#else
    isGpu_ = false;
#endif
    
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
    
    // Set input channels and action space size based on game type
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
    
#ifndef LIBTORCH_OFF
    // Load model if path is provided, otherwise create a new one
    if (!modelPath.empty()) {
        spdlog::info("TorchNeuralNetwork: Loading model from {}", modelPath);
        try {
            // Load the TorchScript model using torch::jit::load
            model_ = torch::jit::load(modelPath);
            model_.to(device_);
            model_.eval(); // Set the model to evaluation mode
            spdlog::info("TorchNeuralNetwork: Model loaded successfully to {}", getDeviceInfo());
        } catch (const c10::Error& e) {
            spdlog::error("TorchNeuralNetwork: Error loading model: {}", e.what());
            throw std::runtime_error("Failed to load the TorchScript model.");
        }
    } else {
        // Handle case where no model path is provided but LibTorch is enabled
        // Potentially create a default model or log a warning
        spdlog::warn("TorchNeuralNetwork: No model path provided. Model not loaded.");
    }
    
    // Initialize batch queue if async execution is enabled (AFTER model loading)
    if (config_.useAsyncExecution) {
        spdlog::info("TorchNeuralNetwork: Initializing batch queue with size {} and batch size {}", 
                     config_.maxQueueSize, config_.batchSize);
        
        // Create BatchQueueConfig from TorchNeuralNetworkConfig
        BatchQueueConfig bqConfig;
        bqConfig.batchSize = config_.batchSize;
        bqConfig.maxQueueSize = config_.maxQueueSize;
        // Use defaults for timeoutMs, numWorkerThreads etc. unless specified in config_
        // bqConfig.timeoutMs = config_.timeoutMs; // Example if available
        // bqConfig.numWorkerThreads = config_.numWorkerThreads; // Example if available
        
        // Pass 'this' (as NeuralNetwork*) and the config object
        batchQueue_ = std::make_unique<BatchQueue>(this, bqConfig);
    } else {
        spdlog::info("TorchNeuralNetwork: Asynchronous execution disabled.");
    }

#else
    // LibTorch is disabled
    spdlog::warn("TorchNeuralNetwork: LibTorch is disabled. Neural network functionality will be limited.");
    if (!modelPath.empty()) {
        // Throw error if a model path was provided but LibTorch is off
        spdlog::error("TorchNeuralNetwork: Cannot load model '{}' - LibTorch is disabled.", modelPath);
        throw std::runtime_error("Cannot load model: LibTorch is disabled");
    }
    // Async execution is inherently disabled if LibTorch is off, so no need for the check below
#endif

    // Configuration settings (apply regardless of LibTorch status)
    setConfig(config);
    batchSize_ = config_.batchSize;

    // Warmup if enabled and LibTorch is available
#ifndef LIBTORCH_OFF
    if (config_.useWarmup) {
        performWarmup();
    }
#else
     if (config_.useWarmup) {
        spdlog::warn("TorchNeuralNetwork: Warmup disabled because LibTorch is off.");
     }
#endif

    spdlog::info("TorchNeuralNetwork: Initialization complete.");
}

TorchNeuralNetwork::~TorchNeuralNetwork() {
    // Clean up
    batchQueue_.reset();
    
#ifndef LIBTORCH_OFF
    // Clear tensor cache
    clearCache();
#endif
}

std::pair<std::vector<float>, float> TorchNeuralNetwork::predict(const core::IGameState& state) {
#ifndef LIBTORCH_OFF
    // If async execution is enabled, use the batch queue
    if (config_.useAsyncExecution && batchQueue_) {
        auto future = batchQueue_->enqueue(state);
        return future.get();
    }
    
    // Convert state to tensor
    torch::Tensor input = getCachedStateTensor(state);
    
    // Measure inference time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Forward pass
    torch::NoGradGuard no_grad;
    torch::jit::IValue output;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Convert to FP16 if configured
        if (config_.useFp16 && isGpu_) {
            input = input.to(torch::kHalf);
        }
        
        output = model_.forward({input});
    }
    
    // Measure inference time
    auto end = std::chrono::high_resolution_clock::now();
    float inferenceTime = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Update average inference time with exponential moving average
    {
        std::lock_guard<std::mutex> lock(mutex_);
        avgInferenceTimeMs_ = (avgInferenceTimeMs_ * 0.95f) + (inferenceTime * 0.05f);
    }
    
    // Process output
    return processOutput(output, state.getActionSpaceSize());
#else
    // LibTorch disabled - return random policy
    std::vector<float> policy(state.getActionSpaceSize(), 0.0f);
    
    // Get legal moves
    auto legalMoves = state.getLegalMoves();
    float value = 0.0f;
    
    // Set uniform probabilities for legal moves
    if (!legalMoves.empty()) {
        float prob = 1.0f / legalMoves.size();
        for (int move : legalMoves) {
            if (move >= 0 && move < static_cast<int>(policy.size())) {
                policy[move] = prob;
            }
        }
    }
    
    return {policy, value};
#endif
}

void TorchNeuralNetwork::predictBatch(
    const std::vector<std::reference_wrapper<const core::IGameState>>& states,
    std::vector<std::vector<float>>& policies,
    std::vector<float>& values) {
#ifndef LIBTORCH_OFF
    if (states.empty()) {
        return;
    }
    
    // Prepare output vectors
    int batchSize = states.size();
    policies.resize(batchSize);
    values.resize(batchSize);
    
    try {
        // Measure inference time
        auto start = std::chrono::high_resolution_clock::now();
        
        // Create a batch tensor by stacking all state tensors
        std::vector<torch::Tensor> stateTensors;
        stateTensors.reserve(batchSize);
        
        // Use thread-local batch tensor reuse to avoid constant allocations
        for (const auto& stateRef : states) {
            const auto& state = stateRef.get();
            stateTensors.push_back(getCachedStateTensor(state));
        }
        
        // Stack the tensors to form a batch
        // Use non-blocking copy to overlap CPU-GPU transfer with computation
        torch::Tensor batchTensor = torch::stack(stateTensors, 0).to(device_, torch::kFloat, true, true);
        
        // Convert to FP16 if configured (faster on GPU)
        if (config_.useFp16 && isGpu_) {
            batchTensor = batchTensor.to(torch::kHalf);
        }
        
        // Forward pass (no need for lock here, model inference is thread-safe)
        torch::NoGradGuard no_grad;
        auto outputTuple = model_.forward({batchTensor}).toTuple();
        
        // Get policy and value tensors
        torch::Tensor policyTensor = outputTuple->elements()[0].toTensor();
        torch::Tensor valueTensor = outputTuple->elements()[1].toTensor();
        
        // Ensure CPU tensors for processing (non-blocking to overlap with next computation)
        policyTensor = policyTensor.to(torch::kCPU, torch::kFloat, true, true);
        valueTensor = valueTensor.to(torch::kCPU, torch::kFloat, true, true);
        
        // Wait for async transfers to complete
        if (policyTensor.is_cuda() || valueTensor.is_cuda()) {
            torch::cuda::synchronize();
        }
        
        // Convert tensors to output vectors
        auto policyAccessor = policyTensor.accessor<float, 2>();
        auto valueAccessor = valueTensor.accessor<float, 1>();
        
        // Process each item in batch
        for (int i = 0; i < batchSize; ++i) {
            // Get action size for this state
            int actionSize = states[i].get().getActionSpaceSize();
            
            // Convert policy tensor to vector
            std::vector<float> policy(actionSize);
            for (int j = 0; j < actionSize && j < policyAccessor.size(1); ++j) {
                policy[j] = policyAccessor[i][j];
            }
            
            // Apply softmax if needed (some models output logits instead of probabilities)
            float sum = 0.0f;
            float maxVal = *std::max_element(policy.begin(), policy.end());
            
            for (auto& p : policy) {
                p = std::exp(p - maxVal);
                sum += p;
            }
            
            if (sum > 0.0f) {
                for (auto& p : policy) {
                    p /= sum;
                }
            }
            
            // Store results
            policies[i] = std::move(policy);
            values[i] = valueAccessor[i];
        }
        
        // Measure inference time
        auto end = std::chrono::high_resolution_clock::now();
        float inferenceTime = std::chrono::duration<float, std::milli>(end - start).count();
        float timePerState = inferenceTime / batchSize;
        
        // Update average inference time with exponential moving average
        {
            std::lock_guard<std::mutex> lock(mutex_);
            avgInferenceTimeMs_ = (avgInferenceTimeMs_ * 0.95f) + (timePerState * 0.05f);
        }
    } catch (const c10::Error& e) {
        // Special handling for torch-specific errors
        spdlog::error("TorchNeuralNetwork: Torch error in predictBatch: {}", e.what());
        
        // Return uniform policies and zero values as fallback
        for (int i = 0; i < batchSize; ++i) {
            int actionSize = states[i].get().getActionSpaceSize();
            policies[i].assign(actionSize, 1.0f / actionSize);
            values[i] = 0.0f;
        }
    } catch (const std::exception& e) {
        // Handle other exceptions
        spdlog::error("TorchNeuralNetwork: Exception in predictBatch: {}", e.what());
        
        // Return uniform policies and zero values as fallback
        for (int i = 0; i < batchSize; ++i) {
            int actionSize = states[i].get().getActionSpaceSize();
            policies[i].assign(actionSize, 1.0f / actionSize);
            values[i] = 0.0f;
        }
    }
#else
    // LibTorch disabled fallback
    for (size_t i = 0; i < states.size(); ++i) {
        const auto& state = states[i].get();
        int actionSize = state.getActionSpaceSize();
        policies[i].assign(actionSize, 1.0f / actionSize);
        values[i] = 0.0f;
    }
#endif
}

std::future<std::pair<std::vector<float>, float>> TorchNeuralNetwork::predictAsync(
    const core::IGameState& state
) {
    // If batch queue is enabled, use it
    if (config_.useAsyncExecution && batchQueue_) {
        return batchQueue_->enqueue(state);
    }
    
    // Otherwise, create a future that runs prediction in a separate thread
    return std::async(std::launch::async, [this, state = state.clone()]() mutable {
        return this->predict(*state);
    });
}

bool TorchNeuralNetwork::isGpuAvailable() const {
#ifndef LIBTORCH_OFF
    return isGpu_;
#else
    return false;
#endif
}

std::string TorchNeuralNetwork::getDeviceInfo() const {
    std::ostringstream oss;
    
#ifndef LIBTORCH_OFF
    if (isGpu_) {
        // Get CUDA device properties
        oss << "GPU: CUDA Device " << device_.index();
        
        try {
            // Try to get device name if available
            #if defined(TORCH_VERSION_MAJOR) && defined(TORCH_VERSION_MINOR)
            #if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 9
            int deviceIndex = device_.index();
            auto deviceProperties = c10::cuda::getDeviceProperties(deviceIndex);
            oss << " (" << deviceProperties->name << ")";
            #endif
            #endif
        } catch (const std::exception& e) {
            // Ignore errors
        }
        
        // Add precision info
        if (config_.useFp16) {
            oss << " (FP16)";
        } else {
            oss << " (FP32)";
        }
    } else {
        oss << "CPU";
    }
#else
    oss << "CPU (LibTorch disabled)";
#endif
    
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
    
#ifdef LIBTORCH_OFF
    oss << " (LibTorch disabled)";
#endif
    
    return oss.str();
}

size_t TorchNeuralNetwork::getModelSizeBytes() const {
#ifndef LIBTORCH_OFF
    size_t size = 0;
    
    try {
        // For JIT modules, we can't easily get the exact size
        // Instead, estimate based on number of parameters
        int64_t paramCount = 0;
        
        // We can't easily iterate parameters in a jit module in this version
        // Just return an approximate size
        return 1024 * 1024; // Return 1MB as approximate size
    } catch (const std::exception& e) {
        if (debugMode_) {
            std::cout << "Error calculating model size: " << e.what() << std::endl;
        }
    }
    
    return size;
#else
    return 0;
#endif
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
    
    // Async inference benchmark
    if (config_.useAsyncExecution && batchQueue_) {
        std::cout << "Async inference benchmark:" << std::endl;
        
        // Warmup
        std::vector<std::future<std::pair<std::vector<float>, float>>> futures;
        for (int i = 0; i < 10; ++i) {
            futures.push_back(predictAsync(*state));
        }
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < numIterations; ++i) {
            futures.push_back(predictAsync(*state));
        }
        
        // Wait for all futures
        for (auto& future : futures) {
            future.wait();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        float totalTimeMs = std::chrono::duration<float, std::milli>(end - start).count();
        
        std::cout << "  Average time: " << (totalTimeMs / numIterations) << " ms" << std::endl;
    }
}

void TorchNeuralNetwork::enableDebugMode(bool enable) {
    debugMode_ = enable;
}

void TorchNeuralNetwork::printModelSummary() const {
    std::cout << "Model summary:" << std::endl;
    std::cout << "  Game type: " << static_cast<int>(gameType_) << std::endl;
    std::cout << "  Board size: " << boardSize_ << std::endl;
    std::cout << "  Input channels: " << inputChannels_ << std::endl;
    std::cout << "  Action space size: " << actionSpaceSize_ << std::endl;
    std::cout << "  Device: " << getDeviceInfo() << std::endl;
    
    // Print model size
    size_t modelSizeBytes = getModelSizeBytes();
    std::cout << "  Model size: " << (modelSizeBytes / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // Print inference time
    std::cout << "  Average inference time: " << getInferenceTimeMs() << " ms" << std::endl;
    
    // Print tensor cache stats
    std::cout << getCacheStats() << std::endl;
    
    // Print batch queue stats if available
    if (batchQueue_) {
        std::cout << getBatchQueueStats() << std::endl;
    }
    
#ifdef LIBTORCH_OFF
    std::cout << "  LibTorch is disabled - using random policy" << std::endl;
#endif
}

void TorchNeuralNetwork::setConfig(const TorchNeuralNetworkConfig& config) {
    // Acquire lock to safely update configuration
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Store previous async execution setting
    bool previousAsyncExecution = config_.useAsyncExecution;
    
    // Update configuration
    config_ = config;
    
    // Update batch size
    batchSize_ = config.batchSize;
    
    // Update batch queue if needed
    if (config.useAsyncExecution != previousAsyncExecution) {
        if (config.useAsyncExecution) {
            BatchQueueConfig bqConfig;
            bqConfig.batchSize = config.batchSize;
            bqConfig.maxQueueSize = config.maxQueueSize;
            bqConfig.useAdaptiveBatching = true;
            batchQueue_ = std::make_unique<BatchQueue>(this, bqConfig);
        } else {
            batchQueue_.reset();
        }
    } else if (batchQueue_ && config.batchSize != batchQueue_->getBatchSize()) {
        batchQueue_->setBatchSize(config.batchSize);
    }
    
#ifndef LIBTORCH_OFF
    // If cache size changed, manage cache
    if (config.useTensorCaching && config.maxCacheSize != 0) {
        std::lock_guard<std::mutex> cacheLock(cacheMutex_);
        
        // If cache size reduced, remove oldest entries
        if (config.maxCacheSize < static_cast<int>(tensorCache_.size())) {
            // Sort entries by last access time
            std::vector<std::pair<uint64_t, std::chrono::steady_clock::time_point>> entries;
            entries.reserve(tensorCache_.size());
            
            for (const auto& pair : tensorCache_) {
                entries.emplace_back(pair.first, pair.second.lastAccess);
            }
            
            // Sort by time (oldest first)
            std::sort(entries.begin(), entries.end(), 
                     [](const auto& a, const auto& b) { return a.second < b.second; });
            
            // Remove oldest entries
            int numToRemove = static_cast<int>(tensorCache_.size()) - config.maxCacheSize;
            for (int i = 0; i < numToRemove; ++i) {
                tensorCache_.erase(entries[i].first);
            }
            
            // Update cache size
            cacheSize_.store(tensorCache_.size(), std::memory_order_relaxed);
        }
    } else if (!config.useTensorCaching) {
        // Clear cache if disabled
        clearCache();
    }
    
    // Apply FP16 if needed
    if (config.useFp16 != config_.useFp16 && isGpu_) {
        if (config.useFp16) {
            try {
                // For JIT modules, we can't directly use to(), 
                // but we can set parameters manually if needed
                if (debugMode_) {
                    std::cout << "FP16 requested but direct conversion not available for JIT module" << std::endl;
                }
            } catch (const std::exception& e) {
                if (debugMode_) {
                    std::cout << "Failed to handle FP16 setting: " << e.what() << std::endl;
                }
                config_.useFp16 = false;
            }
        } else {
            try {
                // For JIT modules, we can't directly use to(), 
                // but we can note the switch back to FP32
                if (debugMode_) {
                    std::cout << "Switching back to FP32 mode" << std::endl;
                }
                // We can't modify the config parameter since it's const
                // The actual config_ will be updated below
            } catch (const std::exception& e) {
                if (debugMode_) {
                    std::cout << "Failed to handle FP32 setting: " << e.what() << std::endl;
                }
            }
        }
    }
#endif
}

void TorchNeuralNetwork::clearCache() {
#ifndef LIBTORCH_OFF
    std::lock_guard<std::mutex> lock(cacheMutex_);
    tensorCache_.clear();
    cacheSize_.store(0, std::memory_order_relaxed);
#endif
}

std::string TorchNeuralNetwork::getCacheStats() const {
    std::ostringstream oss;
    oss << "Tensor cache stats:" << std::endl;
    
#ifndef LIBTORCH_OFF
    size_t hits = cacheHits_.load(std::memory_order_relaxed);
    size_t misses = cacheMisses_.load(std::memory_order_relaxed);
    size_t size = cacheSize_.load(std::memory_order_relaxed);
    
    float hitRate = hits + misses > 0 ? 
                  static_cast<float>(hits) / (hits + misses) : 0.0f;
    
    oss << "  Cache enabled: " << (config_.useTensorCaching ? "yes" : "no") << std::endl;
    oss << "  Cache size: " << size << " entries" << std::endl;
    oss << "  Cache hits: " << hits << std::endl;
    oss << "  Cache misses: " << misses << std::endl;
    oss << "  Hit rate: " << std::fixed << std::setprecision(2) << (hitRate * 100.0f) << "%" << std::endl;
    
    // Calculate memory usage
    size_t memoryUsage = 0;
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        for (const auto& pair : tensorCache_) {
            memoryUsage += pair.second.tensor.nbytes();
        }
    }
    
    oss << "  Memory usage: " << std::fixed << std::setprecision(2) 
       << (memoryUsage / 1024.0 / 1024.0) << " MB" << std::endl;
#else
    oss << "  Cache disabled: LibTorch not available" << std::endl;
#endif
    
    return oss.str();
}

bool TorchNeuralNetwork::exportToOnnx(const std::string& outputPath) const {
#ifndef LIBTORCH_OFF
    try {
        // This version of PyTorch may not directly support ONNX export through the C++ API
        // Instead, we'll print instructions for the user
        if (debugMode_) {
            std::cout << "Direct ONNX export not supported in this PyTorch version." << std::endl;
            std::cout << "To export to ONNX format, use the Python API instead:" << std::endl;
            std::cout << "    import torch" << std::endl;
            std::cout << "    model = torch.jit.load('" << outputPath << ".pt')" << std::endl;
            std::cout << "    torch.onnx.export(model, dummy_input, '" << outputPath << "')" << std::endl;
        }
        
        // Save the model in pt format instead
        std::string torchPath = outputPath + ".pt";
        model_.save(torchPath);
        
        return true;
    } catch (const std::exception& e) {
        if (debugMode_) {
            std::cout << "Error exporting model: " << e.what() << std::endl;
        }
        return false;
    }
#else
    return false;
#endif
}

void TorchNeuralNetwork::clearBatchQueue() {
    // Reset batch queue
    if (batchQueue_) {
        BatchQueueConfig bqConfig;
        bqConfig.batchSize = config_.batchSize;
        bqConfig.maxQueueSize = config_.maxQueueSize;
        bqConfig.useAdaptiveBatching = true;
        batchQueue_ = std::make_unique<BatchQueue>(this, bqConfig);
    }
}

std::string TorchNeuralNetwork::getBatchQueueStats() const {
    if (batchQueue_) {
        return batchQueue_->getStats().toString();
    }
    
    return "Batch queue not enabled";
}

#ifndef LIBTORCH_OFF
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
    torch::Tensor tensor;
    
    if (config_.useNhwcFormat) {
        // NHWC format: [1, H, W, C]
        tensor = torch::from_blob(flatTensor.data(), {1, height, width, channels}, 
                                torch::kFloat32).clone();
        // Convert to NCHW for model input
        tensor = tensor.permute({0, 3, 1, 2});
    } else {
        // NCHW format: [1, C, H, W]
        tensor = torch::from_blob(flatTensor.data(), {1, channels, height, width}, 
                                torch::kFloat32).clone();
    }
    
    return tensor.to(device_);
}

torch::Tensor TorchNeuralNetwork::getCachedStateTensor(const core::IGameState& state) const {
    // If caching is disabled, generate tensor directly
    if (!config_.useTensorCaching) {
        return stateTensor(state);
    }
    
    // Get state hash for cache lookup
    uint64_t hash = state.getHash();
    
    // Check cache
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = tensorCache_.find(hash);
        if (it != tensorCache_.end()) {
            // Update last access time
            it->second.lastAccess = std::chrono::steady_clock::now();
            
            // Update stats
            cacheHits_.fetch_add(1, std::memory_order_relaxed);
            
            return it->second.tensor;
        }
    }
    
    // Cache miss, generate tensor
    cacheHits_.fetch_add(1, std::memory_order_relaxed);
    torch::Tensor tensor = stateTensor(state);
    
    // Add to cache if not full
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        
        // Check if cache is full
        if (tensorCache_.size() >= static_cast<size_t>(config_.maxCacheSize)) {
            // Find oldest entry
            uint64_t oldestKey = 0;
            auto oldestTime = std::chrono::steady_clock::now();
            bool foundEntry = false;
            
            for (const auto& pair : tensorCache_) {
                if (!foundEntry || pair.second.lastAccess < oldestTime) {
                    oldestKey = pair.first;
                    oldestTime = pair.second.lastAccess;
                    foundEntry = true;
                }
            }
            
            // Remove oldest entry
            if (foundEntry) {
                tensorCache_.erase(oldestKey);
            }
        }
        
        // Add new entry
        tensorCache_.emplace(hash, CacheEntry(tensor));
        cacheSize_.store(tensorCache_.size(), std::memory_order_relaxed);
    }
    
    return tensor;
}

std::pair<std::vector<float>, float> TorchNeuralNetwork::processOutput(
    const torch::jit::IValue& output, int actionSize) const {
    
    auto outputTuple = output.toTuple();
    auto policyTensor = outputTuple->elements()[0].toTensor();
    auto valueTensor = outputTuple->elements()[1].toTensor();
    
    // Convert from FP16 if needed
    if (config_.useFp16 && isGpu_) {
        policyTensor = policyTensor.to(torch::kFloat);
        valueTensor = valueTensor.to(torch::kFloat);
    }
    
    // Move tensors to CPU for processing
    policyTensor = policyTensor.to(torch::kCPU);
    valueTensor = valueTensor.to(torch::kCPU);
    
    // Convert policy to vector
    std::vector<float> policy(actionSize);
    int tensorSize = policyTensor.size(1);
    
    if (tensorSize == actionSize) {
        // Direct copy
        std::memcpy(policy.data(), policyTensor.data_ptr<float>(), actionSize * sizeof(float));
    } else {
        // Resize policy if needed
        if (debugMode_) {
            std::cerr << "Warning: Policy size mismatch. Expected " << actionSize
                     << ", got " << tensorSize << std::endl;
        }
        
        // Copy what we can
        int copySize = std::min(actionSize, tensorSize);
        std::memcpy(policy.data(), policyTensor.data_ptr<float>(), copySize * sizeof(float));
        
        // Fill the rest with zeros
        for (int i = copySize; i < actionSize; ++i) {
            policy[i] = 0.0f;
        }
    }
    
    // Apply compression if configured
    if (config_.useOutputCompression) {
        policy = decompressPolicy(compressPolicy(policy), actionSize);
    }
    
    // Get value
    float value = valueTensor.item<float>();
    
    return {policy, value};
}

void TorchNeuralNetwork::createModel() {
    throw std::runtime_error("Direct model creation is not supported. Please provide a pre-trained model.");
}

void TorchNeuralNetwork::performWarmup() {
    if (debugMode_) {
        std::cout << "Performing warmup inferences..." << std::endl;
    }
    
    // Create dummy state
    std::unique_ptr<core::IGameState> state = core::createGameState(gameType_, boardSize_);
    
    // Single inference warmup
    for (int i = 0; i < config_.numWarmupIterations; ++i) {
        predict(*state);
    }
    
    // Batch inference warmup
    std::vector<std::reference_wrapper<const core::IGameState>> states;
    states.reserve(batchSize_);
    for (int i = 0; i < batchSize_; ++i) {
        states.push_back(std::cref(*state));
    }
    
    std::vector<std::vector<float>> policies;
    std::vector<float> values;
    
    for (int i = 0; i < config_.numWarmupIterations; ++i) {
        predictBatch(states, policies, values);
    }
    
    if (debugMode_) {
        std::cout << "Warmup complete" << std::endl;
    }
}
#endif

std::vector<float> TorchNeuralNetwork::compressPolicy(const std::vector<float>& policy) const {
    // Basic compression: only keep non-zero values above a threshold
    // In a real implementation, this would be more sophisticated
    
    const float threshold = 0.01f;  // Only keep values above 1%
    std::vector<float> compressed;
    
    for (size_t i = 0; i < policy.size(); ++i) {
        if (policy[i] > threshold) {
            // Store index and value
            compressed.push_back(static_cast<float>(i));
            compressed.push_back(policy[i]);
        }
    }
    
    return compressed;
}

std::vector<float> TorchNeuralNetwork::decompressPolicy(
    const std::vector<float>& compressedPolicy, int actionSize) const {
    
    std::vector<float> policy(actionSize, 0.0f);
    
    // Must have pairs of (index, value)
    if (compressedPolicy.size() % 2 != 0) {
        return policy;  // Return zero policy on error
    }
    
    // Reconstruct policy
    for (size_t i = 0; i < compressedPolicy.size(); i += 2) {
        int idx = static_cast<int>(compressedPolicy[i]);
        if (idx >= 0 && idx < actionSize) {
            policy[idx] = compressedPolicy[i + 1];
        }
    }
    
    return policy;
}

} // namespace nn
} // namespace alphazero