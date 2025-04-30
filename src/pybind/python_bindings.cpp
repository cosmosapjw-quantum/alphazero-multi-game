// src/pybind/python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "alphazero/core/igamestate.h"
#include "alphazero/games/gomoku/gomoku_state.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"
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
        .def("predict", &nn::NeuralNetwork::predict)
        .def("isGpuAvailable", &nn::NeuralNetwork::isGpuAvailable)
        .def("getDeviceInfo", &nn::NeuralNetwork::getDeviceInfo)
        .def("getInferenceTimeMs", &nn::NeuralNetwork::getInferenceTimeMs)
        .def("getBatchSize", &nn::NeuralNetwork::getBatchSize)
        .def("getModelInfo", &nn::NeuralNetwork::getModelInfo)
        .def("getModelSizeBytes", &nn::NeuralNetwork::getModelSizeBytes)
        .def("benchmark", &nn::NeuralNetwork::benchmark,
            py::arg("numIterations") = 100,
            py::arg("batchSize") = 16)
        .def("enableDebugMode", &nn::NeuralNetwork::enableDebugMode);
    
    // Factory function
    m.def("createNeuralNetwork", &nn::NeuralNetwork::create,
        py::arg("modelPath"),
        py::arg("gameType"),
        py::arg("boardSize") = 0,
        py::arg("useGpu") = true);
    
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
        .def("search", &mcts::ParallelMCTS::search)
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
        .def("generateGames", &selfplay::SelfPlayManager::generateGames,
            py::arg("gameType"),
            py::arg("boardSize") = 0,
            py::arg("useVariantRules") = false)
        .def("setExplorationParams", &selfplay::SelfPlayManager::setExplorationParams,
            py::arg("dirichletAlpha") = 0.03f,
            py::arg("dirichletEpsilon") = 0.25f,
            py::arg("initialTemperature") = 1.0f,
            py::arg("temperatureDropMove") = 30,
            py::arg("finalTemperature") = 0.0f)
        .def("setSaveGames", &selfplay::SelfPlayManager::setSaveGames,
            py::arg("saveGames"),
            py::arg("outputDir") = "games")
        .def("setAbort", &selfplay::SelfPlayManager::setAbort)
        .def("isRunning", &selfplay::SelfPlayManager::isRunning);
}

} // namespace python
} // namespace alphazero