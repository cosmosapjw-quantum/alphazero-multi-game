// include/alphazero/cli/cli_interface.h
#ifndef CLI_INTERFACE_H
#define CLI_INTERFACE_H

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include "alphazero/core/igamestate.h"
#include "alphazero/mcts/parallel_mcts.h"
#include "alphazero/nn/neural_network.h"

namespace alphazero {
namespace cli {

/**
 * @brief Command-line interface for AlphaZero
 * 
 * This class implements a command-line interface for interacting
 * with the AlphaZero engine.
 */
class CLIInterface {
public:
    /**
     * @brief Constructor
     * 
     * @param neuralNetwork Neural network to use for evaluation
     */
    explicit CLIInterface(nn::NeuralNetwork* neuralNetwork = nullptr);
    
    /**
     * @brief Destructor
     */
    ~CLIInterface();
    
    /**
     * @brief Run the CLI in interactive mode
     * 
     * @return Exit code
     */
    int run();
    
    /**
     * @brief Execute a single command
     * 
     * @param command Command to execute
     * @param args Arguments for the command
     * @return true if successful, false otherwise
     */
    bool executeCommand(const std::string& command, const std::vector<std::string>& args);
    
    /**
     * @brief Set output callback for flexibility in displaying output
     * 
     * @param callback Callback function that takes a string
     */
    void setOutputCallback(std::function<void(const std::string&)> callback);
    
    /**
     * @brief Set input callback for flexibility in getting input
     * 
     * @param callback Callback function that returns a string
     */
    void setInputCallback(std::function<std::string()> callback);
    
    /**
     * @brief Set the neural network
     * 
     * @param nn Neural network to use
     */
    void setNeuralNetwork(nn::NeuralNetwork* nn) { neuralNetwork_ = nn; }
    
    /**
     * @brief Get current game state
     * 
     * @return Current game state, or nullptr if no game is active
     */
    core::IGameState* getCurrentState() const { return currentState_.get(); }
    
private:
    // Callbacks for I/O
    std::function<void(const std::string&)> outputCallback_;
    std::function<std::string()> inputCallback_;
    
    // Game state and MCTS
    std::unique_ptr<core::IGameState> currentState_;
    std::unique_ptr<mcts::ParallelMCTS> mcts_;
    nn::NeuralNetwork* neuralNetwork_;
    
    // MCTS parameters
    int numThreads_ = 4;
    int numSimulations_ = 800;
    
    // Command handlers
    using CommandHandler = std::function<bool(const std::vector<std::string>&)>;
    std::map<std::string, CommandHandler> commands_;
    std::map<std::string, std::string> commandHelp_;
    
    // Register all commands
    void registerCommands();
    
    // Output a message
    void output(const std::string& message);
    
    // Get input from user
    std::string input();
    
    // Command implementations
    bool cmdHelp(const std::vector<std::string>& args);
    bool cmdNew(const std::vector<std::string>& args);
    bool cmdPlay(const std::vector<std::string>& args);
    bool cmdAiMove(const std::vector<std::string>& args);
    bool cmdUndo(const std::vector<std::string>& args);
    bool cmdShow(const std::vector<std::string>& args);
    bool cmdInfo(const std::vector<std::string>& args);
    bool cmdSetOption(const std::vector<std::string>& args);
    bool cmdQuit(const std::vector<std::string>& args);
    bool cmdSave(const std::vector<std::string>& args);
    bool cmdLoad(const std::vector<std::string>& args);
    bool cmdBenchmark(const std::vector<std::string>& args);
};

} // namespace cli
} // namespace alphazero

#endif // CLI_INTERFACE_H