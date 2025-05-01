// Add options for the self_play binary
py::options options;
options.add(py::arg("--model", ""), "Path to model file");
options.add(py::arg("--game", "gomoku"), "Game type (gomoku, chess, go)");
options.add(py::arg("--size", 0), "Board size (0 for default)");
options.add(py::arg("--num-games", 100), "Number of games to generate");
options.add(py::arg("--simulations", 800), "Number of MCTS simulations per move");
options.add(py::arg("--threads", 4), "Number of threads to use");
options.add(py::arg("--output-dir", "games"), "Output directory for game records");
options.add(py::arg("--batch-size", 16), "Batch size for neural network inference");
options.add(py::arg("--batch-timeout", 10), "Maximum time to wait for batch completion (ms)");
options.add(py::arg("--temperature", 1.0f), "Initial temperature for move selection");
options.add(py::arg("--temp-drop", 30), "Move number to drop temperature");
options.add(py::arg("--final-temp", 0.0f), "Final temperature after drop point");
options.add(py::arg("--dirichlet-alpha", 0.03f), "Dirichlet noise alpha parameter");
options.add(py::arg("--dirichlet-epsilon", 0.25f), "Weight of Dirichlet noise");
options.add(py::arg("--variant", false), "Use variant rules");
options.add(py::arg("--no-gpu", false), "Disable GPU acceleration");
options.add(py::arg("--fp16", false), "Use FP16 precision");
options.add(py::arg("--c-puct", 1.5f), "Exploration constant");
options.add(py::arg("--fpu-reduction", 0.1f), "First play urgency reduction");
options.add(py::arg("--virtual-loss", 3), "Virtual loss amount");
options.add(py::arg("--no-tt", false), "Disable transposition table");
options.add(py::arg("--progressive-widening", false), "Use progressive widening");

// Add new performance optimization options
options.add(py::arg("--cache-size", 2097152), "Size of transposition table cache (default: 2M entries)");
options.add(py::arg("--pin-threads", false), "Pin threads to CPU cores");
options.add(py::arg("--compact-size", 0), "Compact board representation size (0 for no compaction)");
options.add(py::arg("--thread-affinity", ""), "Thread affinity range (e.g., '0-3')"); 