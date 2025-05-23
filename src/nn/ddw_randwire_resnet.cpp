#include "alphazero/nn/ddw_randwire_resnet.h"
#include <fstream>
#include <queue>
#include <spdlog/spdlog.h>

namespace alphazero {
namespace nn {

// ---------- SEBlock Implementation ----------
SEBlock::SEBlock(int64_t channels, int64_t reduction) {
    squeeze = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1));
    
    excitation = torch::nn::Sequential(
        torch::nn::Linear(channels, channels / reduction),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(channels / reduction, channels),
        torch::nn::Sigmoid()
    );
    
    register_module("squeeze", squeeze);
    register_module("excitation", excitation);
}

torch::Tensor SEBlock::forward(torch::Tensor x) {
    int64_t b_size = x.size(0);
    int64_t c_size = x.size(1);
    
    torch::Tensor y = squeeze(x).view({b_size, c_size});
    y = excitation->forward(y).view({b_size, c_size, 1, 1});
    
    return x * y;
}

// ---------- ResidualBlock Implementation ----------
ResidualBlock::ResidualBlock(int64_t channels) {
    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                              .padding(1).bias(false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3)
                              .padding(1).bias(false));
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    se = std::make_shared<SEBlock>(channels);
    
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("se", se);
}

torch::Tensor ResidualBlock::forward(torch::Tensor x) {
    torch::Tensor residual = x;
    torch::Tensor out = torch::relu(bn1(conv1(x)));
    out = bn2(conv2(out));
    out = se->forward(out);
    out += residual;
    out = torch::relu(out);
    return out;
}

// ---------- RouterModule Implementation ----------
RouterModule::RouterModule(int64_t in_channels, int64_t out_channels) {
    conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                             .bias(false));
    bn = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels));
    
    register_module("conv", conv);
    register_module("bn", bn);
}

torch::Tensor RouterModule::forward(torch::Tensor x) {
    return torch::relu(bn(conv(x)));
}

// ---------- DiGraph Implementation ----------
void DiGraph::add_node(int node) {
    nodes_.insert(node);
    if (adjacency_list_.find(node) == adjacency_list_.end()) {
        adjacency_list_[node] = std::vector<int>();
    }
    if (reverse_adjacency_list_.find(node) == reverse_adjacency_list_.end()) {
        reverse_adjacency_list_[node] = std::vector<int>();
    }
}

void DiGraph::add_edge(int from, int to) {
    add_node(from);
    add_node(to);
    
    adjacency_list_[from].push_back(to);
    reverse_adjacency_list_[to].push_back(from);
}

std::vector<int> DiGraph::nodes() const {
    std::vector<int> result;
    result.reserve(nodes_.size());
    for (int node : nodes_) {
        result.push_back(node);
    }
    return result;
}

std::vector<int> DiGraph::predecessors(int node) const {
    auto it = reverse_adjacency_list_.find(node);
    if (it != reverse_adjacency_list_.end()) {
        return it->second;
    }
    return std::vector<int>();
}

std::vector<int> DiGraph::successors(int node) const {
    auto it = adjacency_list_.find(node);
    if (it != adjacency_list_.end()) {
        return it->second;
    }
    return std::vector<int>();
}

int DiGraph::in_degree(int node) const {
    auto it = reverse_adjacency_list_.find(node);
    if (it != reverse_adjacency_list_.end()) {
        return static_cast<int>(it->second.size());
    }
    return 0;
}

int DiGraph::out_degree(int node) const {
    auto it = adjacency_list_.find(node);
    if (it != adjacency_list_.end()) {
        return static_cast<int>(it->second.size());
    }
    return 0;
}

std::vector<Edge> DiGraph::edges() const {
    std::vector<Edge> result;
    for (const auto& [from, to_list] : adjacency_list_) {
        for (int to : to_list) {
            result.push_back({from, to});
        }
    }
    return result;
}

size_t DiGraph::size() const {
    return nodes_.size();
}

bool DiGraph::_dfs_topo_sort(int node, std::unordered_set<int>& visited, 
                           std::unordered_set<int>& temp_visited, 
                           std::vector<int>& result) const {
    if (temp_visited.find(node) != temp_visited.end()) {
        // Cycle detected
        return false;
    }
    
    if (visited.find(node) == visited.end()) {
        temp_visited.insert(node);
        
        auto it = adjacency_list_.find(node);
        if (it != adjacency_list_.end()) {
            for (int successor : it->second) {
                if (!_dfs_topo_sort(successor, visited, temp_visited, result)) {
                    return false;
                }
            }
        }
        
        temp_visited.erase(node);
        visited.insert(node);
        result.push_back(node);
    }
    
    return true;
}

std::vector<int> DiGraph::topological_sort() const {
    std::vector<int> result;
    std::unordered_set<int> visited;
    std::unordered_set<int> temp_visited;
    
    // Run DFS from each unvisited node
    for (int node : nodes_) {
        if (visited.find(node) == visited.end()) {
            if (!_dfs_topo_sort(node, visited, temp_visited, result)) {
                // Cycle detected, return empty vector
                return std::vector<int>();
            }
        }
    }
    
    // Reverse the result for correct order
    std::reverse(result.begin(), result.end());
    return result;
}

// ---------- RandWireBlock Implementation ----------
RandWireBlock::RandWireBlock(int64_t channels, int64_t num_nodes, double p, int64_t seed)
    : channels_(channels), num_nodes_(num_nodes) {
    
    // Generate random graph
    graph_ = _generate_graph(num_nodes, p, seed);
    
    // Find input and output nodes
    for (int node : graph_.nodes()) {
        if (graph_.in_degree(node) == 0) {
            input_nodes_.push_back(node);
        }
        if (graph_.out_degree(node) == 0) {
            output_nodes_.push_back(node);
        }
    }
    
    // Ensure at least one input and output node
    if (input_nodes_.empty()) {
        input_nodes_.push_back(0);
    }
    if (output_nodes_.empty()) {
        output_nodes_.push_back(num_nodes - 1);
    }
    
    // Create router modules and register them
    for (int node : graph_.nodes()) {
        int in_degree = graph_.in_degree(node);
        if (in_degree > 0) {
            auto router = std::make_shared<RouterModule>(in_degree * channels, channels);
            register_module("router_" + std::to_string(node), router);
            routers_.emplace(std::to_string(node), router);
        }
    }
    
    // Create residual blocks and register them
    for (int node : graph_.nodes()) {
        auto block = std::make_shared<ResidualBlock>(channels);
        register_module("block_" + std::to_string(node), block);
        blocks_.emplace(std::to_string(node), block);
    }
    
    // Create output router if needed
    if (output_nodes_.size() > 1) {
        output_router_ = std::make_shared<RouterModule>(output_nodes_.size() * channels, channels);
        register_module("output_router", output_router_);
    }
}

DiGraph RandWireBlock::_generate_graph(int64_t num_nodes, double p, int64_t seed) {
    // Set random seed for reproducibility
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(static_cast<unsigned int>(seed));
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    
    // Create Watts-Strogatz small-world graph
    int k = 4;  // Each node is connected to k nearest neighbors
    
    // Create ring lattice
    DiGraph G;
    for (int i = 0; i < num_nodes; i++) {
        G.add_node(i);
    }
    
    // Add initial edges to k nearest neighbors
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 1; j <= k / 2; j++) {
            int target = (i + j) % num_nodes;
            // Make edges directed to avoid cycles
            if (i < target) {
                G.add_edge(i, target);
            } else {
                G.add_edge(target, i);
            }
        }
    }
    
    // Rewire edges with probability p
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> node_dist(0, num_nodes - 1);
    
    // Get all edges
    std::vector<Edge> edges = G.edges();
    
    // Create a new graph for rewiring
    DiGraph DG;
    for (int i = 0; i < num_nodes; i++) {
        DG.add_node(i);
    }
    
    // Rewire each edge with probability p
    for (const Edge& edge : edges) {
        int u = edge.from;
        int v = edge.to;
        
        if (dist(gen) < p) {
            // Rewire edge
            int w = node_dist(gen);
            while (w == u || w == v || (u < w && DG.successors(u).end() != std::find(DG.successors(u).begin(), DG.successors(u).end(), w)) ||
                  (w < u && DG.successors(w).end() != std::find(DG.successors(w).begin(), DG.successors(w).end(), u))) {
                w = node_dist(gen);
            }
            
            // Add new edge, ensuring it flows from lower to higher index
            if (u < w) {
                DG.add_edge(u, w);
            } else {
                DG.add_edge(w, u);
            }
        } else {
            // Keep original edge
            DG.add_edge(u, v);
        }
    }
    
    return DG;
}

torch::Tensor RandWireBlock::forward(torch::Tensor x) {
    // Node outputs map
    std::unordered_map<int, torch::Tensor> node_outputs;
    
    // Process input nodes
    for (int node : input_nodes_) {
        auto it = blocks_.find(std::to_string(node));
        if (it != blocks_.end()) {
            node_outputs[node] = it->second->forward(x);
        }
    }
    
    // Process nodes in topological order
    std::vector<int> topo_order = graph_.topological_sort();
    for (int node : topo_order) {
        // Skip input nodes
        if (std::find(input_nodes_.begin(), input_nodes_.end(), node) != input_nodes_.end()) {
            continue;
        }
        
        // Get inputs from predecessor nodes
        std::vector<int> predecessors = graph_.predecessors(node);
        if (predecessors.empty()) {
            continue;
        }
        
        // Concatenate inputs
        std::vector<torch::Tensor> inputs;
        for (int pred : predecessors) {
            inputs.push_back(node_outputs[pred]);
        }
        
        torch::Tensor routed;
        if (inputs.size() > 1) {
            torch::Tensor combined = torch::cat(inputs, 1);
            auto it = routers_.find(std::to_string(node));
            if (it != routers_.end()) {
                routed = it->second->forward(combined);
            } else {
                routed = combined;
            }
        } else {
            routed = inputs[0];
        }
        
        // Process through residual block
        auto it = blocks_.find(std::to_string(node));
        if (it != blocks_.end()) {
            node_outputs[node] = it->second->forward(routed);
        }
    }
    
    // Combine outputs
    if (output_nodes_.size() > 1) {
        std::vector<torch::Tensor> outputs;
        for (int node : output_nodes_) {
            outputs.push_back(node_outputs[node]);
        }
        torch::Tensor combined = torch::cat(outputs, 1);
        return output_router_->forward(combined);
    } else {
        return node_outputs[output_nodes_[0]];
    }
}

// ---------- DDWRandWireResNet Implementation ----------
DDWRandWireResNet::DDWRandWireResNet(int64_t input_channels, int64_t output_size, 
                                   int64_t channels, int64_t num_blocks)
    : input_channels_(input_channels) {
    
    // Input layer
    input_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, channels, 3)
                                   .padding(1).bias(false));
    input_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(channels));
    
    // Random wire blocks
    rand_wire_blocks_ = torch::nn::ModuleList();
    for (int64_t i = 0; i < num_blocks; ++i) {
        rand_wire_blocks_->push_back(std::make_shared<RandWireBlock>(channels, 32, 0.75, i));
    }
    
    // Policy head
    policy_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 32, 1).bias(false));
    policy_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    policy_fc_ = torch::nn::Linear(32 * 8 * 8, output_size);
    
    // Value head
    value_conv_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, 32, 1).bias(false));
    value_bn_ = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(32));
    value_fc1_ = torch::nn::Linear(32 * 8 * 8, 256);
    value_fc2_ = torch::nn::Linear(256, 1);
    
    // Register modules
    register_module("input_conv", input_conv_);
    register_module("input_bn", input_bn_);
    register_module("rand_wire_blocks", rand_wire_blocks_);
    register_module("policy_conv", policy_conv_);
    register_module("policy_bn", policy_bn_);
    register_module("policy_fc", policy_fc_);
    register_module("value_conv", value_conv_);
    register_module("value_bn", value_bn_);
    register_module("value_fc1", value_fc1_);
    register_module("value_fc2", value_fc2_);
    
    // Initialize weights
    _initialize_weights();
}

std::tuple<torch::Tensor, torch::Tensor> DDWRandWireResNet::forward(torch::Tensor x) {
    // Input layer
    x = torch::relu(input_bn_(input_conv_(x)));
    
    // Random wire blocks
    for (const auto& block : *rand_wire_blocks_) {
        x = block->as<RandWireBlock>()->forward(x);
    }
    
    // Adaptive pooling to handle different board sizes
    auto sizes = x.sizes();
    int64_t batch = sizes[0];
    int64_t channels = sizes[1];
    int64_t height = sizes[2];
    int64_t width = sizes[3];
    
    // Target size of 8x8
    int64_t target_size = 8;
    target_size = std::min(target_size, std::min(height, width));
    
    torch::Tensor x_pooled;
    if (height != target_size || width != target_size) {
        x_pooled = torch::adaptive_avg_pool2d(x, {target_size, target_size});
    } else {
        x_pooled = x;
    }
    
    // Policy head
    torch::Tensor policy = torch::relu(policy_bn_(policy_conv_(x_pooled)));
    policy = policy.view({policy.size(0), -1});
    policy = policy_fc_(policy);
    
    // Value head
    torch::Tensor value = torch::relu(value_bn_(value_conv_(x_pooled)));
    value = value.view({value.size(0), -1});
    value = torch::relu(value_fc1_(value));
    value = torch::tanh(value_fc2_(value));
    
    return {policy, value};
}

void DDWRandWireResNet::_initialize_weights() {
    // He initialization for all layers
    for (auto& module : modules(/*include_self=*/false)) {
        if (auto* conv = module->as<torch::nn::Conv2d>()) {
            torch::nn::init::kaiming_normal_(
                conv->weight, 0.0, torch::kFanOut, torch::kReLU);
            if (conv->options.bias()) {
                torch::nn::init::constant_(conv->bias, 0.0);
            }
        } else if (auto* bn = module->as<torch::nn::BatchNorm2d>()) {
            torch::nn::init::constant_(bn->weight, 1.0);
            torch::nn::init::constant_(bn->bias, 0.0);
        } else if (auto* linear = module->as<torch::nn::Linear>()) {
            torch::nn::init::kaiming_normal_(
                linear->weight, 0.0, torch::kFanOut, torch::kReLU);
            torch::nn::init::constant_(linear->bias, 0.0);
        }
    }
}

void DDWRandWireResNet::save(const std::string& path) {
    try {
        auto self = shared_from_this();
        torch::save(self, path);
        spdlog::info("Model saved to {}", path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to save model: {}", e.what());
        throw;
    }
}

void DDWRandWireResNet::load(const std::string& path) {
    try {
        auto self = shared_from_this();
        torch::load(self, path);
        spdlog::info("Model loaded from {}", path);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load model: {}", e.what());
        throw;
    }
}

void DDWRandWireResNet::export_to_torchscript(const std::string& path, std::vector<int64_t> input_shape) {
    try {
        // Set model to evaluation mode
        eval();
        
        // Create dummy input for tracing
        if (input_shape[1] == 0) {
            input_shape[1] = input_channels_;
        }
        if (input_shape[2] == 0 || input_shape[3] == 0) {
            input_shape[2] = input_shape[3] = 8;  // Default board size
        }
        
        torch::Tensor dummy_input = torch::zeros(input_shape);
        
        // Trace the model
        torch::jit::script::Module traced_module;
        try {
            // Save directly
            traced_module.save(path);
            spdlog::info("Model exported to TorchScript format at {}", path);
            return;
        } catch (const c10::Error& e) {
            spdlog::warn("Failed to save model: {}. Trying alternative method.", e.what());
        }
        
        try {
            // Alternative save method
            auto model_copy = std::dynamic_pointer_cast<torch::nn::Module>(shared_from_this());
            if (!model_copy) {
                throw std::runtime_error("Failed to cast model to Module");
            }
            torch::save(model_copy, path);
            spdlog::info("Model exported to regular format at {}", path);
        } catch (const std::exception& e) {
            spdlog::error("Failed to export model: {}", e.what());
            throw;
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to export model to TorchScript: {}", e.what());
        throw;
    }
}

} // namespace nn
} // namespace alphazero