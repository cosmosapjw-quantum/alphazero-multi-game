// src/pybind/python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

// Include torch/script.h BEFORE your headers that use torch::nn::Module
#include <torch/script.h> 

#include "alphazero/core/igamestate.h"
#include "alphazero/games/gomoku/gomoku_state.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"
#include "alphazero/nn/torch_neural_network.h"
#include "alphazero/nn/random_policy_network.h"
#include "alphazero/nn/ddw_randwire_resnet.h"
#include "alphazero/selfplay/self_play_manager.h"
#include "alphazero/selfplay/game_record.h"
#include "alphazero/selfplay/dataset.h"

namespace py = pybind11;

namespace alphazero {
namespace python {

PYBIND11_MODULE(_alphazero_cpp, m) {
    m.doc() = "AlphaZero Multi-Game AI Engine C++ Bindings";
    
    // Enums
    py::enum_<core::GameType>(m, "GameType")
        .value("GOMOKU", core::GameType::GOMOKU)
        .value("CHESS", core::GameType::CHESS)
        .value("GO", core::GameType::GO)
        .export_values();
    
    py::enum_<core::GameResult>(m, "GameResult")
        .value("ONGOING", core::GameResult::ONGOING)
        .value("DRAW", core::GameResult::DRAW)
        .value("WIN_PLAYER1", core::GameResult::WIN_PLAYER1)
        .value("WIN_PLAYER2", core::GameResult::WIN_PLAYER2)
        .export_values();
    
    py::enum_<mcts::MCTSNodeSelection>(m, "MCTSNodeSelection")
        .value("UCB", mcts::MCTSNodeSelection::UCB)
        .value("PUCT", mcts::MCTSNodeSelection::PUCT)
        .value("PROGRESSIVE_BIAS", mcts::MCTSNodeSelection::PROGRESSIVE_BIAS)
        .value("RAVE", mcts::MCTSNodeSelection::RAVE)
        .export_values();
    
    py::enum_<mcts::MCTSSearchMode>(m, "MCTSSearchMode")
        .value("SERIAL", mcts::MCTSSearchMode::SERIAL)
        .value("PARALLEL", mcts::MCTSSearchMode::PARALLEL)
        .value("BATCHED", mcts::MCTSSearchMode::BATCHED)
        .export_values();
    
    // Game State interface
    py::class_<core::IGameState>(m, "IGameState")
        .def("getLegalMoves", &core::IGameState::getLegalMoves)
        .def("isLegalMove", &core::IGameState::isLegalMove)
        .def("makeMove", &core::IGameState::makeMove)
        .def("undoMove", &core::IGameState::undoMove)
        .def("isTerminal", &core::IGameState::isTerminal)
        .def("getGameResult", &core::IGameState::getGameResult)
        .def("getCurrentPlayer", &core::IGameState::getCurrentPlayer)
        .def("getBoardSize", &core::IGameState::getBoardSize)
        .def("getActionSpaceSize", &core::IGameState::getActionSpaceSize)
        .def("getTensorRepresentation", &core::IGameState::getTensorRepresentation)
        .def("getEnhancedTensorRepresentation", &core::IGameState::getEnhancedTensorRepresentation)
        .def("actionToString", &core::IGameState::actionToString)
        .def("stringToAction", &core::IGameState::stringToAction)
        .def("toString", &core::IGameState::toString)
        .def("getMoveHistory", &core::IGameState::getMoveHistory)
        .def("getGameType", &core::IGameState::getGameType);
    
    // Gomoku State
    py::class_<gomoku::GomokuState, core::IGameState>(m, "GomokuState")
        .def(py::init<int, bool, bool, int, bool>(),
            py::arg("board_size") = 15,
            py::arg("use_renju") = false,
            py::arg("use_omok") = false,
            py::arg("seed") = 0,
            py::arg("use_pro_long_opening") = false)
        .def("is_occupied", &gomoku::GomokuState::is_occupied)
        .def("get_board", &gomoku::GomokuState::get_board);
    
    // Factory function
    m.def("createGameState", &core::createGameState,
        py::arg("type"),
        py::arg("boardSize") = 0,
        py::arg("variantRules") = false);
    
    // Neural Network interface
    py::class_<nn::NeuralNetwork>(m, "NeuralNetwork")
        .def("predict", [](nn::NeuralNetwork &self, const core::IGameState &state) {
            // Release GIL during potentially long-running neural network inference
            py::gil_scoped_release release;
            return self.predict(state);
        })
        .def("predictBatch", [](nn::NeuralNetwork &self, 
                              const std::vector<std::reference_wrapper<const core::IGameState>> &states,
                              std::vector<std::vector<float>> &policies,
                              std::vector<float> &values) {
            // Release GIL during batch inference
            py::gil_scoped_release release;
            self.predictBatch(states, policies, values);
        })
        .def("isGpuAvailable", &nn::NeuralNetwork::isGpuAvailable)
        .def("getDeviceInfo", &nn::NeuralNetwork::getDeviceInfo)
        .def("getInferenceTimeMs", &nn::NeuralNetwork::getInferenceTimeMs)
        .def("getBatchSize", &nn::NeuralNetwork::getBatchSize)
        .def("getModelInfo", &nn::NeuralNetwork::getModelInfo)
        .def("getModelSizeBytes", &nn::NeuralNetwork::getModelSizeBytes)
        .def("benchmark", [](nn::NeuralNetwork &self, int numIterations, int batchSize) {
            // Release GIL during benchmark
            py::gil_scoped_release release;
            self.benchmark(numIterations, batchSize);
        }, py::arg("numIterations") = 100, py::arg("batchSize") = 16)
        .def("enableDebugMode", &nn::NeuralNetwork::enableDebugMode)
        // Check if a neural network is implemented in C++ and won't trigger GIL issues
        .def("is_gil_safe", [](nn::NeuralNetwork &self) {
            // This heuristic checks if the network is one of our known C++ implementations
            return dynamic_cast<nn::TorchNeuralNetwork*>(&self) != nullptr || 
                   dynamic_cast<nn::RandomPolicyNetwork*>(&self) != nullptr;
        });
    
    // Factory function
    m.def("createNeuralNetwork", &nn::NeuralNetwork::create,
        py::arg("modelPath"),
        py::arg("gameType"),
        py::arg("boardSize") = 0,
        py::arg("useGpu") = true);
    
    // Register the torch::nn::Module base class FIRST
    // This is crucial for pybind11 to understand the inheritance chain
    py::class_<torch::nn::Module, std::shared_ptr<torch::nn::Module>>(m, "TorchModule")
        .def(py::init<>())
        .def("train", &torch::nn::Module::train, py::arg("on") = true)
        .def("eval", &torch::nn::Module::eval)
        .def("is_training", &torch::nn::Module::is_training);
    
    // Now register all the derived classes    
    // Add DDWRandWireResNet and components
    py::class_<nn::SEBlock, std::shared_ptr<nn::SEBlock>, torch::nn::Module>(m, "SEBlock")
        .def(py::init<int64_t, int64_t>(),
             py::arg("channels"),
             py::arg("reduction") = 16)
        .def("forward", &nn::SEBlock::forward);
    
    py::class_<nn::ResidualBlock, std::shared_ptr<nn::ResidualBlock>, torch::nn::Module>(m, "ResidualBlock")
        .def(py::init<int64_t>(),
             py::arg("channels"))
        .def("forward", &nn::ResidualBlock::forward);
    
    py::class_<nn::RouterModule, std::shared_ptr<nn::RouterModule>, torch::nn::Module>(m, "RouterModule")
        .def(py::init<int64_t, int64_t>(),
             py::arg("in_channels"),
             py::arg("out_channels"))
        .def("forward", &nn::RouterModule::forward);
    
    py::class_<nn::RandWireBlock, std::shared_ptr<nn::RandWireBlock>, torch::nn::Module>(m, "RandWireBlock")
        .def(py::init<int64_t, int64_t, double, int64_t>(),
             py::arg("channels"),
             py::arg("num_nodes") = 32,
             py::arg("p") = 0.75,
             py::arg("seed") = -1)
        .def("forward", &nn::RandWireBlock::forward);
    
    py::class_<nn::DDWRandWireResNet, std::shared_ptr<nn::DDWRandWireResNet>, torch::nn::Module>(m, "DDWRandWireResNetCpp")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(),
             py::arg("input_channels"),
             py::arg("output_size"),
             py::arg("channels") = 128,
             py::arg("num_blocks") = 20)
        .def("forward", &nn::DDWRandWireResNet::forward)
        .def("save", &nn::DDWRandWireResNet::save)
        .def("load", &nn::DDWRandWireResNet::load)
        .def("export_to_torchscript", &nn::DDWRandWireResNet::export_to_torchscript,
             py::arg("path"),
             py::arg("input_shape") = std::vector<int64_t>{1, 0, 0, 0});
    
    // Factory function for DDWRandWireResNet
    m.def("createDDWRandWireResNet", &nn::TorchNeuralNetwork::createDDWRandWireResNet,
        py::arg("input_channels"),
        py::arg("output_size"),
        py::arg("channels") = 128,
        py::arg("num_blocks") = 20);
    
    // MCTS Config struct
    py::class_<mcts::MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("numThreads", &mcts::MCTSConfig::numThreads)
        .def_readwrite("numSimulations", &mcts::MCTSConfig::numSimulations)
        .def_readwrite("cPuct", &mcts::MCTSConfig::cPuct)
        .def_readwrite("fpuReduction", &mcts::MCTSConfig::fpuReduction)
        .def_readwrite("virtualLoss", &mcts::MCTSConfig::virtualLoss)
        .def_readwrite("maxSearchDepth", &mcts::MCTSConfig::maxSearchDepth)
        .def_readwrite("useDirichletNoise", &mcts::MCTSConfig::useDirichletNoise)
        .def_readwrite("dirichletAlpha", &mcts::MCTSConfig::dirichletAlpha)
        .def_readwrite("dirichletEpsilon", &mcts::MCTSConfig::dirichletEpsilon)
        .def_readwrite("useBatchInference", &mcts::MCTSConfig::useBatchInference)
        .def_readwrite("useTemporalDifference", &mcts::MCTSConfig::useTemporalDifference)
        .def_readwrite("tdLambda", &mcts::MCTSConfig::tdLambda)
        .def_readwrite("useProgressiveWidening", &mcts::MCTSConfig::useProgressiveWidening)
        .def_readwrite("minVisitsForWidening", &mcts::MCTSConfig::minVisitsForWidening)
        .def_readwrite("progressiveWideningBase", &mcts::MCTSConfig::progressiveWideningBase)
        .def_readwrite("progressiveWideningExponent", &mcts::MCTSConfig::progressiveWideningExponent)
        .def_readwrite("selectionStrategy", &mcts::MCTSConfig::selectionStrategy)
        .def_readwrite("batchSize", &mcts::MCTSConfig::batchSize)
        .def_readwrite("useBatchedMCTS", &mcts::MCTSConfig::useBatchedMCTS)
        .def_readwrite("batchTimeoutMs", &mcts::MCTSConfig::batchTimeoutMs)
        .def_readwrite("searchMode", &mcts::MCTSConfig::searchMode);
    
    // MCTS Stats struct
    py::class_<mcts::MCTSStats>(m, "MCTSStats")
        .def(py::init<>())
        .def_property_readonly("nodesCreated", [](const mcts::MCTSStats& stats) {
            return stats.nodesCreated.load();
        })
        .def_property_readonly("nodesExpanded", [](const mcts::MCTSStats& stats) {
            return stats.nodesExpanded.load();
        })
        .def_property_readonly("nodesTotalVisits", [](const mcts::MCTSStats& stats) {
            return stats.nodesTotalVisits.load();
        })
        .def_property_readonly("simulationCount", [](const mcts::MCTSStats& stats) {
            return stats.simulationCount.load();
        })
        .def_property_readonly("evaluationCalls", [](const mcts::MCTSStats& stats) {
            return stats.evaluationCalls.load();
        })
        .def_property_readonly("cacheHits", [](const mcts::MCTSStats& stats) {
            return stats.cacheHits.load();
        })
        .def_property_readonly("cacheMisses", [](const mcts::MCTSStats& stats) {
            return stats.cacheMisses.load();
        })
        .def_property_readonly("batchedEvaluations", [](const mcts::MCTSStats& stats) {
            return stats.batchedEvaluations.load();
        })
        .def_property_readonly("totalBatches", [](const mcts::MCTSStats& stats) {
            return stats.totalBatches.load();
        });
    
    // MCTS implementation
    py::class_<mcts::MCTSNode>(m, "MCTSNode")
        .def("getUcbScore", &mcts::MCTSNode::getUcbScore)
        .def("getTerminalValue", &mcts::MCTSNode::getTerminalValue)
        .def("getValue", &mcts::MCTSNode::getValue)
        .def("getBestAction", &mcts::MCTSNode::getBestAction)
        .def("getVisitCountDistribution", &mcts::MCTSNode::getVisitCountDistribution,
            py::arg("temperature") = 1.0f)
        .def("toString", &mcts::MCTSNode::toString,
            py::arg("maxDepth") = 1);
    
    py::class_<mcts::TranspositionTable>(m, "TranspositionTable")
        .def(py::init<size_t, size_t>(),
            py::arg("size") = 1048576,
            py::arg("numShards") = 1024)
        .def("getSize", &mcts::TranspositionTable::getSize)
        .def("getHitRate", &mcts::TranspositionTable::getHitRate)
        .def("getLookups", &mcts::TranspositionTable::getLookups)
        .def("getHits", &mcts::TranspositionTable::getHits)
        .def("getEntryCount", &mcts::TranspositionTable::getEntryCount)
        .def("getMemoryUsageBytes", &mcts::TranspositionTable::getMemoryUsageBytes)
        .def("clear", &mcts::TranspositionTable::clear)
        .def("resize", &mcts::TranspositionTable::resize);
    
    py::class_<mcts::ParallelMCTS>(m, "ParallelMCTS")
        .def(py::init<const core::IGameState&, nn::NeuralNetwork*, mcts::TranspositionTable*, int, int, float, float, int>(),
            py::arg("rootState"),
            py::arg("nn") = nullptr,
            py::arg("tt") = nullptr,
            py::arg("numThreads") = 1,
            py::arg("numSimulations") = 800,
            py::arg("cPuct") = 1.5f,
            py::arg("fpuReduction") = 0.0f,
            py::arg("virtualLoss") = 3)
        .def(py::init<const core::IGameState&, const mcts::MCTSConfig&, nn::NeuralNetwork*, mcts::TranspositionTable*>(),
            py::arg("rootState"),
            py::arg("config"),
            py::arg("nn") = nullptr,
            py::arg("tt") = nullptr)
        .def("search", [](mcts::ParallelMCTS &self) {
            // Release GIL during the computationally intensive search
            py::gil_scoped_release release;
            self.search();
        })
        .def("selectAction", &mcts::ParallelMCTS::selectAction,
            py::arg("isTraining") = false,
            py::arg("temperature") = 1.0f)
        .def("getActionProbabilities", &mcts::ParallelMCTS::getActionProbabilities,
            py::arg("temperature") = 1.0f)
        .def("getRootValue", &mcts::ParallelMCTS::getRootValue)
        .def("updateWithMove", &mcts::ParallelMCTS::updateWithMove)
        .def("addDirichletNoise", &mcts::ParallelMCTS::addDirichletNoise,
            py::arg("alpha") = 0.03f,
            py::arg("epsilon") = 0.25f)
        .def("setNumThreads", &mcts::ParallelMCTS::setNumThreads)
        .def("setNumSimulations", &mcts::ParallelMCTS::setNumSimulations)
        .def("setCPuct", &mcts::ParallelMCTS::setCPuct)
        .def("setFpuReduction", &mcts::ParallelMCTS::setFpuReduction)
        .def("setVirtualLoss", &mcts::ParallelMCTS::setVirtualLoss)
        .def("setNeuralNetwork", &mcts::ParallelMCTS::setNeuralNetwork)
        .def("setTranspositionTable", &mcts::ParallelMCTS::setTranspositionTable)
        .def("setSelectionStrategy", &mcts::ParallelMCTS::setSelectionStrategy)
        .def("setConfig", &mcts::ParallelMCTS::setConfig)
        .def("enableBatchedMCTS", &mcts::ParallelMCTS::enableBatchedMCTS)
        .def("setBatchSize", &mcts::ParallelMCTS::setBatchSize)
        .def("setBatchTimeout", &mcts::ParallelMCTS::setBatchTimeout)
        .def("setDeterministicMode", &mcts::ParallelMCTS::setDeterministicMode)
        .def("setDebugMode", &mcts::ParallelMCTS::setDebugMode)
        .def("printSearchStats", &mcts::ParallelMCTS::printSearchStats)
        .def("getSearchInfo", &mcts::ParallelMCTS::getSearchInfo)
        .def("printSearchPath", &mcts::ParallelMCTS::printSearchPath)
        .def("getMemoryUsage", &mcts::ParallelMCTS::getMemoryUsage);
    
    // Game Record and Dataset
    py::class_<selfplay::MoveData>(m, "MoveData")
        .def_readwrite("action", &selfplay::MoveData::action)
        .def_readwrite("policy", &selfplay::MoveData::policy)
        .def_readwrite("value", &selfplay::MoveData::value)
        .def_readwrite("thinking_time_ms", &selfplay::MoveData::thinking_time_ms);
    
    py::class_<selfplay::GameRecord>(m, "GameRecord")
        .def(py::init<core::GameType, int, bool>(),
            py::arg("gameType"),
            py::arg("boardSize"),
            py::arg("useVariantRules") = false)
        .def("addMove", &selfplay::GameRecord::addMove)
        .def("setResult", &selfplay::GameRecord::setResult)
        .def("getMetadata", &selfplay::GameRecord::getMetadata)
        .def("getMoves", &selfplay::GameRecord::getMoves)
        .def("getResult", &selfplay::GameRecord::getResult)
        .def("toJson", &selfplay::GameRecord::toJson)
        .def("saveToFile", &selfplay::GameRecord::saveToFile)
        .def_static("fromJson", &selfplay::GameRecord::fromJson)
        .def_static("loadFromFile", &selfplay::GameRecord::loadFromFile);
    
    py::class_<selfplay::TrainingExample>(m, "TrainingExample")
        .def_readwrite("state", &selfplay::TrainingExample::state)
        .def_readwrite("policy", &selfplay::TrainingExample::policy)
        .def_readwrite("value", &selfplay::TrainingExample::value)
        .def("toJson", &selfplay::TrainingExample::toJson)
        .def_static("fromJson", &selfplay::TrainingExample::fromJson);
    
    py::class_<selfplay::Dataset>(m, "Dataset")
        .def(py::init<>())
        .def("addGameRecord", &selfplay::Dataset::addGameRecord,
            py::arg("record"),
            py::arg("useEnhancedFeatures") = true)
        .def("extractExamples", &selfplay::Dataset::extractExamples,
            py::arg("includeAugmentations") = true)
        .def("size", &selfplay::Dataset::size)
        .def("getBatch", &selfplay::Dataset::getBatch)
        .def("shuffle", &selfplay::Dataset::shuffle)
        .def("saveToFile", &selfplay::Dataset::saveToFile)
        .def("loadFromFile", &selfplay::Dataset::loadFromFile)
        .def("getRandomSubset", &selfplay::Dataset::getRandomSubset);
    
    // Self-Play Manager
    py::class_<selfplay::SelfPlayManager>(m, "SelfPlayManager")
        .def(py::init<nn::NeuralNetwork*, int, int, int>(),
            py::arg("neuralNetwork"),
            py::arg("numGames") = 100,
            py::arg("numSimulations") = 800,
            py::arg("numThreads") = 4)
        .def("generateGames", [](selfplay::SelfPlayManager &self, 
                               core::GameType gameType, 
                               int boardSize, 
                               bool useVariantRules) {
            // Release GIL for the entire self-play process
            py::gil_scoped_release release;
            return self.generateGames(gameType, boardSize, useVariantRules);
        })
        .def("setExplorationParams", &selfplay::SelfPlayManager::setExplorationParams,
            py::arg("dirichletAlpha") = 0.03f,
            py::arg("dirichletEpsilon") = 0.25f,
            py::arg("initialTemperature") = 1.0f,
            py::arg("temperatureDropMove") = 30,
            py::arg("finalTemperature") = 0.0f)
        .def("setProgressCallback", [](selfplay::SelfPlayManager &self, 
                                    std::function<void(int,int,int,int)> callback) {
            self.setProgressCallback([callback](int gameId, int moveNum, int totalGames, int totalMoves) {
                // Acquire GIL before calling Python callback
                py::gil_scoped_acquire acquire;
                callback(gameId, moveNum, totalGames, totalMoves);
            });
        })
        .def("setBatchConfig", &selfplay::SelfPlayManager::setBatchConfig)
        .def("setSaveGames", &selfplay::SelfPlayManager::setSaveGames)
        .def("setAbort", &selfplay::SelfPlayManager::setAbort)
        .def("isRunning", &selfplay::SelfPlayManager::isRunning)
        .def("setMctsConfig", [](selfplay::SelfPlayManager &self, py::dict config) {
            // Convert Python dict to MCTSConfig
            mcts::MCTSConfig mctsConfig;
            
            // Read values from dict with proper error handling
            if (config.contains("numThreads") && py::isinstance<py::int_>(config["numThreads"]))
                mctsConfig.numThreads = config["numThreads"].cast<int>();
                
            if (config.contains("numSimulations") && py::isinstance<py::int_>(config["numSimulations"]))
                mctsConfig.numSimulations = config["numSimulations"].cast<int>();
                
            if (config.contains("cPuct") && py::isinstance<py::float_>(config["cPuct"]))
                mctsConfig.cPuct = config["cPuct"].cast<float>();
                
            if (config.contains("fpuReduction") && py::isinstance<py::float_>(config["fpuReduction"]))
                mctsConfig.fpuReduction = config["fpuReduction"].cast<float>();
                
            if (config.contains("virtualLoss") && py::isinstance<py::int_>(config["virtualLoss"]))
                mctsConfig.virtualLoss = config["virtualLoss"].cast<int>();
                
            if (config.contains("useDirichletNoise") && py::isinstance<py::bool_>(config["useDirichletNoise"]))
                mctsConfig.useDirichletNoise = config["useDirichletNoise"].cast<bool>();
                
            if (config.contains("dirichletAlpha") && py::isinstance<py::float_>(config["dirichletAlpha"]))
                mctsConfig.dirichletAlpha = config["dirichletAlpha"].cast<float>();
                
            if (config.contains("dirichletEpsilon") && py::isinstance<py::float_>(config["dirichletEpsilon"]))
                mctsConfig.dirichletEpsilon = config["dirichletEpsilon"].cast<float>();
                
            if (config.contains("useBatchInference") && py::isinstance<py::bool_>(config["useBatchInference"]))
                mctsConfig.useBatchInference = config["useBatchInference"].cast<bool>();
                
            if (config.contains("useBatchedMCTS") && py::isinstance<py::bool_>(config["useBatchedMCTS"]))
                mctsConfig.useBatchedMCTS = config["useBatchedMCTS"].cast<bool>();
                
            if (config.contains("batchSize") && py::isinstance<py::int_>(config["batchSize"]))
                mctsConfig.batchSize = config["batchSize"].cast<int>();
                
            if (config.contains("batchTimeoutMs") && py::isinstance<py::int_>(config["batchTimeoutMs"]))
                mctsConfig.batchTimeoutMs = config["batchTimeoutMs"].cast<int>();
                
            if (config.contains("searchMode") && py::isinstance<py::str>(config["searchMode"])) {
                std::string searchModeStr = config["searchMode"].cast<std::string>();
                if (searchModeStr == "SERIAL")
                    mctsConfig.searchMode = mcts::MCTSSearchMode::SERIAL;
                else if (searchModeStr == "PARALLEL")
                    mctsConfig.searchMode = mcts::MCTSSearchMode::PARALLEL;
                else if (searchModeStr == "BATCHED")
                    mctsConfig.searchMode = mcts::MCTSSearchMode::BATCHED;
            }
            
            if (config.contains("useTemporalDifference") && py::isinstance<py::bool_>(config["useTemporalDifference"]))
                mctsConfig.useTemporalDifference = config["useTemporalDifference"].cast<bool>();
                
            if (config.contains("useProgressiveWidening") && py::isinstance<py::bool_>(config["useProgressiveWidening"]))
                mctsConfig.useProgressiveWidening = config["useProgressiveWidening"].cast<bool>();
                
            if (config.contains("useFmapCache") && py::isinstance<py::bool_>(config["useFmapCache"]))
                mctsConfig.useFmapCache = config["useFmapCache"].cast<bool>();
            
            // Apply the config
            self.setMctsConfig(mctsConfig);
        })
        .def("getCompletedGamesCount", &selfplay::SelfPlayManager::getCompletedGamesCount)
        .def("getTotalMovesCount", &selfplay::SelfPlayManager::getTotalMovesCount);
}

} // namespace python
} // namespace alphazero