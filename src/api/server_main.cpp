// src/api/server_main.cpp
#include <iostream>
#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <csignal>
#include "alphazero/api/http_server.h"
#include "alphazero/api/rest_api.h"

using namespace alphazero;

// Global server pointer for signal handling
std::shared_ptr<api::HttpServer> g_server;

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --port PORT       Port to listen on (default: 8080)" << std::endl;
    std::cout << "  --help            Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int port = 8080; // Default port
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--port" && i + 1 < argc) {
            try {
                port = std::stoi(argv[++i]);
                if (port <= 0 || port > 65535) {
                    std::cerr << "Invalid port number. Must be between 1 and 65535." << std::endl;
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Invalid port number: " << e.what() << std::endl;
                return 1;
            }
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "AlphaZero Multi-Game AI Engine API Server" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Register signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Create REST API handler
    auto api = std::make_shared<api::RestApi>();
    
    // Create HTTP server
    g_server = std::make_shared<api::HttpServer>(port, api);
    
    // Start server
    if (!g_server->start()) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }
    
    std::cout << "Server started on port " << port << std::endl;
    std::cout << "Press Ctrl+C to stop the server" << std::endl;
    
    // Wait until server is stopped
    while (g_server->isRunning()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
}