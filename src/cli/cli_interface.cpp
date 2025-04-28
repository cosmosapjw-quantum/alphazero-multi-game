// src/cli/cli_interface.cpp
#include "alphazero/cli/cli_interface.h"
#include "alphazero/cli/command_parser.h"
#include "alphazero/mcts/transposition_table.h"
#include "alphazero/elo/elo_tracker.h"
#include "alphazero/selfplay/game_record.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <fstream>

namespace alphazero {
namespace cli {

CLIInterface::CLIInterface(nn::NeuralNetwork* neuralNetwork)
    : neuralNetwork_(neuralNetwork) {
    
    // Set default callbacks
    outputCallback_ = [](const std::string& message) {
        std::cout << message << std::endl;
    };
    
    inputCallback_ = []() {
        std::string line;
        std::getline(std::cin, line);
        return line;
    };
    
    // Register commands
    registerCommands();
}

CLIInterface::~CLIInterface() {
    // Clean up
}

int CLIInterface::run() {
    output("AlphaZero Multi-Game AI Engine CLI");
    output("==================================");
    output("Type 'help' for a list of commands.");
    
    bool running = true;
    while (running) {
        // Display prompt
        std::cout << "> ";
        
        // Get input
        std::string line = input();
        
        // Parse command and arguments
        std::vector<std::string> tokens = CommandParser::tokenize(line);
        if (tokens.empty()) {
            continue;
        }
        
        std::string command = tokens[0];
        std::vector<std::string> args(tokens.begin() + 1, tokens.end());
        
        // Handle quit command directly
        if (command == "quit" || command == "exit") {
            running = false;
            continue;
        }
        
        // Execute command
        if (!executeCommand(command, args)) {
            output("Unknown command: " + command);
            output("Type 'help' for a list of commands.");
        }
    }
    
    return 0;
}

bool CLIInterface::executeCommand(const std::string& command, const std::vector<std::string>& args) {
    // Convert command to lowercase
    std::string lcCommand = command;
    std::transform(lcCommand.begin(), lcCommand.end(), lcCommand.begin(), ::tolower);
    
    // Find and execute command
    auto it = commands_.find(lcCommand);
    if (it != commands_.end()) {
        return it->second(args);
    }
    
    return false;
}

void CLIInterface::setOutputCallback(std::function<void(const std::string&)> callback) {
    if (callback) {
        outputCallback_ = callback;
    }
}

void CLIInterface::setInputCallback(std::function<std::string()> callback) {
    if (callback) {
        inputCallback_ = callback;
    }
}

void CLIInterface::registerCommands() {
    // Help command
    commands_["help"] = [this](const std::vector<std::string>& args) {
        return cmdHelp(args);
    };
    commandHelp_["help"] = "Display help information. Usage: help [command]";
    
    // New game command
    commands_["new"] = [this](const std::vector<std::string>& args) {
        return cmdNew(args);
    };
    commandHelp_["new"] = "Start a new game. Usage: new <game_type> [board_size] [variant]";
    
    // Play move command
    commands_["play"] = [this](const std::vector<std::string>& args) {
        return cmdPlay(args);
    };
    commandHelp_["play"] = "Make a move. Usage: play <move>";
    
    // AI move command
    commands_["aimove"] = [this](const std::vector<std::string>& args) {
        return cmdAiMove(args);
    };
    commandHelp_["aimove"] = "Let the AI make a move. Usage: aimove [simulations] [temperature]";
    
    // Undo command
    commands_["undo"] = [this](const std::vector<std::string>& args) {
        return cmdUndo(args);
    };
    commandHelp_["undo"] = "Undo the last move. Usage: undo";
    
    // Show board command
    commands_["show"] = [this](const std::vector<std::string>& args) {
        return cmdShow(args);
    };
    commandHelp_["show"] = "Show the current board. Usage: show";
    
    // Info command
    commands_["info"] = [this](const std::vector<std::string>& args) {
        return cmdInfo(args);
    };
    commandHelp_["info"] = "Show information about the current game. Usage: info";
    
    // Set option command
    commands_["setoption"] = [this](const std::vector<std::string>& args) {
        return cmdSetOption(args);
    };
    commandHelp_["setoption"] = "Set an option. Usage: setoption <name> <value>";
    
    // Quit command
    commands_["quit"] = [this](const std::vector<std::string>& args) {
        return cmdQuit(args);
    };
    commandHelp_["quit"] = "Quit the program. Usage: quit";
    commands_["exit"] = commands_["quit"];
    commandHelp_["exit"] = commandHelp_["quit"];
    
    // Save command
    commands_["save"] = [this](const std::vector<std::string>& args) {
        return cmdSave(args);
    };
    commandHelp_["save"] = "Save the current game. Usage: save <filename>";
    
    // Load command
    commands_["load"] = [this](const std::vector<std::string>& args) {
        return cmdLoad(args);
    };
    commandHelp_["load"] = "Load a saved game. Usage: load <filename>";
    
    // Benchmark command
    commands_["benchmark"] = [this](const std::vector<std::string>& args) {
        return cmdBenchmark(args);
    };
    commandHelp_["benchmark"] = "Run a benchmark. Usage: benchmark [simulations] [games]";
}

void CLIInterface::output(const std::string& message) {
    outputCallback_(message);
}

std::string CLIInterface::input() {
    return inputCallback_();
}

bool CLIInterface::cmdHelp(const std::vector<std::string>& args) {
    if (args.empty()) {
        // Show all commands
        output("Available commands:");
        
        std::vector<std::string> cmdNames;
        for (const auto& cmd : commandHelp_) {
            cmdNames.push_back(cmd.first);
        }
        
        // Sort command names
        std::sort(cmdNames.begin(), cmdNames.end());
        
        // Display commands and their help
        for (const auto& name : cmdNames) {
            output("  " + name + " - " + commandHelp_[name]);
        }
    } else {
        // Show help for specific command
        std::string cmd = args[0];
        std::transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);
        
        auto it = commandHelp_.find(cmd);
        if (it != commandHelp_.end()) {
            output(it->second);
        } else {
            output("Unknown command: " + cmd);
        }
    }
    
    return true;
}

bool CLIInterface::cmdNew(const std::vector<std::string>& args) {
    if (args.empty()) {
        output("Missing game type. Usage: new <game_type> [board_size] [variant]");
        output("Available game types: gomoku, chess, go");
        return false;
    }
    
    // Get game type
    std::string gameTypeStr = args[0];
    std::transform(gameTypeStr.begin(), gameTypeStr.end(), gameTypeStr.begin(), ::tolower);
    
    core::GameType gameType;
    if (gameTypeStr == "gomoku") {
        gameType = core::GameType::GOMOKU;
    } else if (gameTypeStr == "chess") {
        gameType = core::GameType::CHESS;
    } else if (gameTypeStr == "go") {
        gameType = core::GameType::GO;
    } else {
        output("Unknown game type: " + gameTypeStr);
        output("Available game types: gomoku, chess, go");
        return false;
    }
    
    // Get board size
    int boardSize = 0;  // Default
    if (args.size() > 1) {
        try {
            boardSize = std::stoi(args[1]);
        } catch (const std::exception& e) {
            output("Invalid board size: " + args[1]);
            return false;
        }
    }
    
    // Get variant flag
    bool variant = false;
    if (args.size() > 2) {
        std::string variantStr = args[2];
        std::transform(variantStr.begin(), variantStr.end(), variantStr.begin(), ::tolower);
        variant = (variantStr == "true" || variantStr == "1" || variantStr == "yes");
    }
    
    // Create game state
    try {
        currentState_ = core::createGameState(gameType, boardSize, variant);
    } catch (const std::exception& e) {
        output("Error creating game: " + std::string(e.what()));
        return false;
    }
    
    // Create MCTS for AI
    if (neuralNetwork_) {
        auto tt = std::make_shared<mcts::TranspositionTable>(1048576, 1024);
        mcts_ = std::make_unique<mcts::ParallelMCTS>(
            *currentState_, neuralNetwork_, tt.get(), numThreads_, numSimulations_);
    }
    
    // Show board
    output("New game created: " + gameTypeStr);
    return cmdShow(std::vector<std::string>());
}

bool CLIInterface::cmdPlay(const std::vector<std::string>& args) {
    if (!currentState_) {
        output("No active game. Use 'new' to start a game.");
        return false;
    }
    
    if (args.empty()) {
        output("Missing move. Usage: play <move>");
        return false;
    }
    
    std::string moveStr = args[0];
    
    // Try to convert move string to action
    auto actionOpt = currentState_->stringToAction(moveStr);
    if (!actionOpt) {
        output("Invalid move: " + moveStr);
        return false;
    }
    
    int action = *actionOpt;
    
    // Check if move is legal
    if (!currentState_->isLegalMove(action)) {
        output("Illegal move: " + moveStr);
        return false;
    }
    
    // Make the move
    currentState_->makeMove(action);
    
    // Update MCTS tree
    if (mcts_) {
        mcts_->updateWithMove(action);
    }
    
    // Show board
    output("Move played: " + moveStr);
    cmdShow(std::vector<std::string>());
    
    // Check if game is over
    if (currentState_->isTerminal()) {
        core::GameResult result = currentState_->getGameResult();
        switch (result) {
            case core::GameResult::WIN_PLAYER1:
                output("Game over: Player 1 wins");
                break;
            case core::GameResult::WIN_PLAYER2:
                output("Game over: Player 2 wins");
                break;
            case core::GameResult::DRAW:
                output("Game over: Draw");
                break;
            default:
                output("Game over with unknown result");
                break;
        }
    }
    
    return true;
}

bool CLIInterface::cmdAiMove(const std::vector<std::string>& args) {
    if (!currentState_) {
        output("No active game. Use 'new' to start a game.");
        return false;
    }
    
    if (!mcts_) {
        output("Neural network not available for AI moves.");
        return false;
    }
    
    if (currentState_->isTerminal()) {
        output("Game is already over.");
        return false;
    }
    
    // Parse arguments
    int simulations = numSimulations_;
    float temperature = 0.0f;
    
    if (args.size() > 0) {
        try {
            simulations = std::stoi(args[0]);
        } catch (const std::exception& e) {
            output("Invalid simulations count: " + args[0]);
            return false;
        }
    }
    
    if (args.size() > 1) {
        try {
            temperature = std::stof(args[1]);
        } catch (const std::exception& e) {
            output("Invalid temperature: " + args[1]);
            return false;
        }
    }
    
    // Set simulations
    mcts_->setNumSimulations(simulations);
    
    // Progress callback
    mcts_->setProgressCallback([this](int current, int total) {
        static int lastPercent = -1;
        int percent = static_cast<int>((current * 100) / total);
        if (percent != lastPercent && percent % 10 == 0) {
            std::cout << "\rThinking... " << percent << "% (" 
                      << current << "/" << total << " simulations)" << std::flush;
            lastPercent = percent;
        }
    });
    
    // Run search
    output("AI thinking...");
    auto startTime = std::chrono::high_resolution_clock::now();
    mcts_->search();
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << std::endl;
    
    // Select best move
    int action = mcts_->selectAction(false, temperature);
    std::string moveStr = currentState_->actionToString(action);
    
    // Make the move
    currentState_->makeMove(action);
    
    // Update MCTS tree
    mcts_->updateWithMove(action);
    
    // Show info
    std::ostringstream ss;
    ss << "AI move: " << moveStr << " (in " 
       << duration.count() << " ms, " 
       << simulations << " simulations)";
    output(ss.str());
    
    // Show board
    cmdShow(std::vector<std::string>());
    
    // Check if game is over
    if (currentState_->isTerminal()) {
        core::GameResult result = currentState_->getGameResult();
        switch (result) {
            case core::GameResult::WIN_PLAYER1:
                output("Game over: Player 1 wins");
                break;
            case core::GameResult::WIN_PLAYER2:
                output("Game over: Player 2 wins");
                break;
            case core::GameResult::DRAW:
                output("Game over: Draw");
                break;
            default:
                output("Game over with unknown result");
                break;
        }
    }
    
    return true;
}

bool CLIInterface::cmdUndo(const std::vector<std::string>& args) {
    if (!currentState_) {
        output("No active game. Use 'new' to start a game.");
        return false;
    }
    
    if (!currentState_->undoMove()) {
        output("Cannot undo move: no moves to undo");
        return false;
    }
    
    // We can't easily update MCTS tree when undoing, so recreate it
    if (mcts_ && neuralNetwork_) {
        auto tt = std::make_shared<mcts::TranspositionTable>(1048576, 1024);
        mcts_ = std::make_unique<mcts::ParallelMCTS>(
            *currentState_, neuralNetwork_, tt.get(), numThreads_, numSimulations_);
    }
    
    output("Move undone");
    cmdShow(std::vector<std::string>());
    return true;
}

bool CLIInterface::cmdShow(const std::vector<std::string>& args) {
    if (!currentState_) {
        output("No active game. Use 'new' to start a game.");
        return false;
    }
    
    // Show board
    output(currentState_->toString());
    
    // Show move history
    std::vector<int> history = currentState_->getMoveHistory();
    if (!history.empty()) {
        std::ostringstream ss;
        ss << "Move history: ";
        for (size_t i = 0; i < history.size(); ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ss << currentState_->actionToString(history[i]);
        }
        output(ss.str());
    }
    
    // Show current player
    output("Current player: " + std::to_string(currentState_->getCurrentPlayer()));
    
    return true;
}

bool CLIInterface::cmdInfo(const std::vector<std::string>& args) {
    if (!currentState_) {
        output("No active game. Use 'new' to start a game.");
        return false;
    }
    
    // Game type
    std::string gameTypeStr;
    switch (currentState_->getGameType()) {
        case core::GameType::GOMOKU:
            gameTypeStr = "Gomoku";
            break;
        case core::GameType::CHESS:
            gameTypeStr = "Chess";
            break;
        case core::GameType::GO:
            gameTypeStr = "Go";
            break;
        default:
            gameTypeStr = "Unknown";
            break;
    }
    
    output("Game type: " + gameTypeStr);
    output("Board size: " + std::to_string(currentState_->getBoardSize()));
    output("Current player: " + std::to_string(currentState_->getCurrentPlayer()));
    output("Is terminal: " + std::string(currentState_->isTerminal() ? "yes" : "no"));
    
    if (currentState_->isTerminal()) {
        std::string resultStr;
        switch (currentState_->getGameResult()) {
            case core::GameResult::WIN_PLAYER1:
                resultStr = "Player 1 wins";
                break;
            case core::GameResult::WIN_PLAYER2:
                resultStr = "Player 2 wins";
                break;
            case core::GameResult::DRAW:
                resultStr = "Draw";
                break;
            default:
                resultStr = "Unknown";
                break;
        }
        output("Game result: " + resultStr);
    }
    
    output("Action space size: " + std::to_string(currentState_->getActionSpaceSize()));
    output("Legal moves: " + std::to_string(currentState_->getLegalMoves().size()));
    output("Move history: " + std::to_string(currentState_->getMoveHistory().size()));
    
    if (mcts_) {
        output("MCTS info:");
        output("  Threads: " + std::to_string(numThreads_));
        output("  Simulations: " + std::to_string(numSimulations_));
    }
    
    if (neuralNetwork_) {
        output("Neural network info:");
        output("  Device: " + neuralNetwork_->getDeviceInfo());
        output("  Model: " + neuralNetwork_->getModelInfo());
        output("  Inference time: " + std::to_string(neuralNetwork_->getInferenceTimeMs()) + " ms");
    }
    
    return true;
}

bool CLIInterface::cmdSetOption(const std::vector<std::string>& args) {
    if (args.size() < 2) {
        output("Usage: setoption <name> <value>");
        return false;
    }
    
    std::string name = args[0];
    std::string valueStr = args[1];
    
    // Convert to lowercase
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    
    if (name == "threads") {
        try {
            int threads = std::stoi(valueStr);
            if (threads <= 0) {
                output("Invalid thread count: must be positive");
                return false;
            }
            
            numThreads_ = threads;
            
            // Update MCTS if active
            if (mcts_) {
                mcts_->setNumThreads(numThreads_);
            }
            
            output("Threads set to " + std::to_string(numThreads_));
        } catch (const std::exception& e) {
            output("Invalid thread count: " + valueStr);
            return false;
        }
    } else if (name == "simulations") {
        try {
            int simulations = std::stoi(valueStr);
            if (simulations <= 0) {
                output("Invalid simulation count: must be positive");
                return false;
            }
            
            numSimulations_ = simulations;
            
            // Update MCTS if active
            if (mcts_) {
                mcts_->setNumSimulations(numSimulations_);
            }
            
            output("Simulations set to " + std::to_string(numSimulations_));
        } catch (const std::exception& e) {
            output("Invalid simulation count: " + valueStr);
            return false;
        }
    } else if (name == "cpuct") {
        try {
            float cPuct = std::stof(valueStr);
            if (cPuct <= 0.0f) {
                output("Invalid cPuct: must be positive");
                return false;
            }
            
            // Update MCTS if active
            if (mcts_) {
                mcts_->setCPuct(cPuct);
                output("cPuct set to " + valueStr);
            } else {
                output("No active MCTS: option will be applied to next game");
            }
        } catch (const std::exception& e) {
            output("Invalid cPuct value: " + valueStr);
            return false;
        }
    } else if (name == "fpureduction") {
        try {
            float fpuReduction = std::stof(valueStr);
            if (fpuReduction < 0.0f) {
                output("Invalid FPU reduction: must be non-negative");
                return false;
            }
            
            // Update MCTS if active
            if (mcts_) {
                mcts_->setFpuReduction(fpuReduction);
                output("FPU reduction set to " + valueStr);
            } else {
                output("No active MCTS: option will be applied to next game");
            }
        } catch (const std::exception& e) {
            output("Invalid FPU reduction value: " + valueStr);
            return false;
        }
    } else {
        output("Unknown option: " + name);
        return false;
    }
    
    return true;
}

bool CLIInterface::cmdQuit(const std::vector<std::string>& args) {
    output("Goodbye!");
    return true;
}

bool CLIInterface::cmdSave(const std::vector<std::string>& args) {
    if (!currentState_) {
        output("No active game to save.");
        return false;
    }
    
    if (args.empty()) {
        output("Missing filename. Usage: save <filename>");
        return false;
    }
    
    std::string filename = args[0];
    
    // Create game record
    selfplay::GameRecord record(
        currentState_->getGameType(),
        currentState_->getBoardSize(),
        false  // variant rules not recorded
    );
    
    // Add moves from history
    std::vector<int> history = currentState_->getMoveHistory();
    for (int action : history) {
        // We don't have the actual policy/value from the game
        // so we use placeholder values
        std::vector<float> policy(currentState_->getActionSpaceSize(), 0.0f);
        policy[action] = 1.0f;  // Set probability to 1.0 for the actual move
        
        record.addMove(action, policy, 0.0f, 0);
    }
    
    // Set result if game is terminal
    if (currentState_->isTerminal()) {
        record.setResult(currentState_->getGameResult());
    }
    
    // Save to file
    if (!record.saveToFile(filename)) {
        output("Error saving game to file: " + filename);
        return false;
    }
    
    output("Game saved to " + filename);
    return true;
}

bool CLIInterface::cmdLoad(const std::vector<std::string>& args) {
    if (args.empty()) {
        output("Missing filename. Usage: load <filename>");
        return false;
    }
    
    std::string filename = args[0];
    
    // Load game record
    selfplay::GameRecord record = selfplay::GameRecord::loadFromFile(filename);
    
    // Extract metadata
    auto [gameType, boardSize, useVariantRules] = record.getMetadata();
    
    // Create new game state
    try {
        currentState_ = core::createGameState(gameType, boardSize, useVariantRules);
    } catch (const std::exception& e) {
        output("Error creating game: " + std::string(e.what()));
        return false;
    }
    
    // Apply moves
    const auto& moves = record.getMoves();
    for (const auto& move : moves) {
        int action = move.action;
        
        // Validate move
        if (!currentState_->isLegalMove(action)) {
            output("Error: illegal move in record: " + currentState_->actionToString(action));
            return false;
        }
        
        // Make move
        currentState_->makeMove(action);
    }
    
    // Create MCTS for AI
    if (neuralNetwork_) {
        auto tt = std::make_shared<mcts::TranspositionTable>(1048576, 1024);
        mcts_ = std::make_unique<mcts::ParallelMCTS>(
            *currentState_, neuralNetwork_, tt.get(), numThreads_, numSimulations_);
    }
    
    output("Game loaded from " + filename);
    cmdShow(std::vector<std::string>());
    return true;
}

bool CLIInterface::cmdBenchmark(const std::vector<std::string>& args) {
    if (!neuralNetwork_) {
        output("Neural network not available for benchmarking.");
        return false;
    }
    
    // Parse arguments
    int simulations = 1000;
    int games = 1;
    
    if (args.size() > 0) {
        try {
            simulations = std::stoi(args[0]);
        } catch (const std::exception& e) {
            output("Invalid simulations count: " + args[0]);
            return false;
        }
    }
    
    if (args.size() > 1) {
        try {
            games = std::stoi(args[1]);
        } catch (const std::exception& e) {
            output("Invalid games count: " + args[1]);
            return false;
        }
    }
    
    // Ask for game type
    output("Select game type for benchmark:");
    output("1. Gomoku");
    output("2. Chess");
    output("3. Go");
    std::cout << "Enter selection (1-3): ";
    std::string selection = input();
    
    core::GameType gameType;
    int boardSize = 0;
    
    if (selection == "1") {
        gameType = core::GameType::GOMOKU;
        boardSize = 15;
    } else if (selection == "2") {
        gameType = core::GameType::CHESS;
        boardSize = 0;  // Default
    } else if (selection == "3") {
        gameType = core::GameType::GO;
        boardSize = 9;  // Use smaller board for benchmark
    } else {
        output("Invalid selection");
        return false;
    }
    
    // Create transposition table
    auto tt = std::make_shared<mcts::TranspositionTable>(1048576, 1024);
    
    // Timing variables
    std::vector<double> searchTimes;
    std::vector<double> moveTimes;
    int totalNodes = 0;
    
    output("Running benchmark with:");
    output("  Game type: " + selection);
    output("  Simulations: " + std::to_string(simulations));
    output("  Threads: " + std::to_string(numThreads_));
    output("  Games: " + std::to_string(games));
    
    // Run benchmark
    for (int game = 0; game < games; ++game) {
        output("Game " + std::to_string(game + 1) + "/" + std::to_string(games));
        
        // Create game state
        auto state = core::createGameState(gameType, boardSize, false);
        
        // Create MCTS
        mcts::ParallelMCTS mcts(*state, neuralNetwork_, tt.get(), numThreads_, simulations);
        
        // Play game until terminal or max 30 moves (for benchmark)
        int moveCount = 0;
        while (!state->isTerminal() && moveCount < 30) {
            // Search for best move
            auto searchStart = std::chrono::high_resolution_clock::now();
            mcts.search();
            auto searchEnd = std::chrono::high_resolution_clock::now();
            
            double searchTimeMs = std::chrono::duration<double, std::milli>(
                searchEnd - searchStart).count();
            
            // Get best move
            auto moveStart = std::chrono::high_resolution_clock::now();
            int action = mcts.selectAction();
            auto moveEnd = std::chrono::high_resolution_clock::now();
            
            double moveTimeMs = std::chrono::duration<double, std::milli>(
                moveEnd - moveStart).count();
            
            // Make move
            if (state->isLegalMove(action)) {
                state->makeMove(action);
                mcts.updateWithMove(action);
                
                // Record times
                searchTimes.push_back(searchTimeMs);
                moveTimes.push_back(moveTimeMs);
                totalNodes += simulations;
                
                std::ostringstream ss;
                ss << "  Move " << moveCount + 1 
                   << ": search=" << searchTimeMs << "ms, move=" << moveTimeMs << "ms";
                output(ss.str());
                
                moveCount++;
            } else {
                output("  Error: illegal move selected");
                break;
            }
        }
    }
    
    // Calculate statistics
    double totalSearchTime = 0.0;
    double minSearchTime = searchTimes.empty() ? 0.0 : searchTimes[0];
    double maxSearchTime = searchTimes.empty() ? 0.0 : searchTimes[0];
    
    for (double time : searchTimes) {
        totalSearchTime += time;
        minSearchTime = std::min(minSearchTime, time);
        maxSearchTime = std::max(maxSearchTime, time);
    }
    
    double avgSearchTime = searchTimes.empty() ? 0.0 : totalSearchTime / searchTimes.size();
    double nodesPerSecond = totalSearchTime > 0.0 ? 
                           (totalNodes * 1000.0) / totalSearchTime : 0.0;
    
    output("Benchmark results:");
    output("  Total searches: " + std::to_string(searchTimes.size()));
    output("  Total nodes: " + std::to_string(totalNodes));
    output("  Search time (avg): " + std::to_string(avgSearchTime) + " ms");
    output("  Search time (min): " + std::to_string(minSearchTime) + " ms");
    output("  Search time (max): " + std::to_string(maxSearchTime) + " ms");
    output("  Nodes per second: " + std::to_string(nodesPerSecond));
    
    output("Transposition table stats:");
    output("  Size: " + std::to_string(tt->getSize()));
    output("  Entries: " + std::to_string(tt->getEntryCount()));
    output("  Lookups: " + std::to_string(tt->getLookups()));
    output("  Hits: " + std::to_string(tt->getHits()));
    output("  Hit rate: " + std::to_string(tt->getHitRate() * 100.0) + "%");
    output("  Memory: " + std::to_string(tt->getMemoryUsageBytes() / 1024.0 / 1024.0) + " MB");
    
    return true;
}

} // namespace cli
} // namespace alphazero