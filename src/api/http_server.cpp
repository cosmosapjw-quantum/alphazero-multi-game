// src/api/http_server.cpp
#include "alphazero/api/http_server.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <regex>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

namespace alphazero {
namespace api {

HttpServer::HttpServer(int port, std::shared_ptr<RestApi> api)
    : port_(port), 
      api_(api), 
      running_(false), 
      serverSocket_(-1),
      stopWorkers_(false) {
}

HttpServer::~HttpServer() {
    stop();
}

bool HttpServer::start() {
    if (running_.load()) {
        std::cerr << "Server already running" << std::endl;
        return false;
    }
    
    // Create socket
    serverSocket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket_ < 0) {
        std::cerr << "Error creating socket: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Allow socket reuse
    int opt = 1;
    if (setsockopt(serverSocket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Error setting socket options: " << strerror(errno) << std::endl;
        close(serverSocket_);
        return false;
    }
    
    // Bind to port
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);
    
    if (bind(serverSocket_, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Error binding socket: " << strerror(errno) << std::endl;
        close(serverSocket_);
        return false;
    }
    
    // Listen for connections
    if (listen(serverSocket_, 10) < 0) {
        std::cerr << "Error listening on socket: " << strerror(errno) << std::endl;
        close(serverSocket_);
        return false;
    }
    
    // Start worker threads
    for (int i = 0; i < NUM_WORKERS; ++i) {
        workerThreads_.emplace_back(&HttpServer::workerThread, this);
    }
    
    // Start server thread
    running_.store(true);
    serverThread_ = std::thread(&HttpServer::serverLoop, this);
    
    std::cout << "Server started on port " << port_ << std::endl;
    return true;
}

void HttpServer::stop() {
    if (!running_.load()) {
        return;
    }
    
    // Stop the server
    running_.store(false);
    
    // Close the server socket to interrupt accept()
    if (serverSocket_ >= 0) {
        close(serverSocket_);
        serverSocket_ = -1;
    }
    
    // Signal workers to stop
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        stopWorkers_.store(true);
    }
    queueCondition_.notify_all();
    
    // Wait for worker threads to finish
    for (auto& thread : workerThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    workerThreads_.clear();
    
    // Wait for server thread to finish
    if (serverThread_.joinable()) {
        serverThread_.join();
    }
    
    std::cout << "Server stopped" << std::endl;
}

bool HttpServer::isRunning() const {
    return running_.load();
}

void HttpServer::serverLoop() {
    while (running_.load()) {
        // Accept a connection
        struct sockaddr_in clientAddr;
        socklen_t clientAddrLen = sizeof(clientAddr);
        int clientSocket = accept(serverSocket_, (struct sockaddr*)&clientAddr, &clientAddrLen);
        
        if (clientSocket < 0) {
            if (running_.load()) {
                std::cerr << "Error accepting connection: " << strerror(errno) << std::endl;
            }
            continue;
        }
        
        // Log connection
        char clientIp[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &clientAddr.sin_addr, clientIp, INET_ADDRSTRLEN);
        std::cout << "Connection from " << clientIp << ":" << ntohs(clientAddr.sin_port) << std::endl;
        
        // Read request
        char buffer[4096] = {0};
        ssize_t bytesRead = read(clientSocket, buffer, sizeof(buffer) - 1);
        if (bytesRead <= 0) {
            std::cerr << "Error reading from socket: " << strerror(errno) << std::endl;
            close(clientSocket);
            continue;
        }
        
        // Parse request
        std::string requestStr(buffer, bytesRead);
        HttpRequest request = parseRequest(requestStr);
        
        // Add task to queue
        RequestTask task = {clientSocket, request};
        addTask(task);
    }
}

void HttpServer::workerThread() {
    while (!stopWorkers_.load()) {
        RequestTask task;
        
        // Get task from queue
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCondition_.wait(lock, [this] {
                return !taskQueue_.empty() || stopWorkers_.load();
            });
            
            if (stopWorkers_.load() && taskQueue_.empty()) {
                break;
            }
            
            if (!taskQueue_.empty()) {
                task = taskQueue_.front();
                taskQueue_.pop();
            } else {
                continue;
            }
        }
        
        // Process the request
        try {
            // Call the API to handle the request
            std::string responseBody = api_->handleRequest(
                task.request.method, 
                task.request.path, 
                task.request.body
            );
            
            // Generate HTTP response
            HttpResponse response;
            response.statusCode = 200;
            response.body = responseBody;
            response.headers["Content-Type"] = "application/json";
            response.headers["Access-Control-Allow-Origin"] = "*";  // CORS
            
            // Send response
            std::string responseStr = generateResponse(response);
            send(task.clientSocket, responseStr.c_str(), responseStr.size(), 0);
        } catch (const std::exception& e) {
            // Send error response
            HttpResponse errorResponse;
            errorResponse.statusCode = 500;
            errorResponse.body = "{\"error\":\"" + std::string(e.what()) + "\"}";
            errorResponse.headers["Content-Type"] = "application/json";
            errorResponse.headers["Access-Control-Allow-Origin"] = "*";  // CORS
            
            std::string responseStr = generateResponse(errorResponse);
            send(task.clientSocket, responseStr.c_str(), responseStr.size(), 0);
        }
        
        // Close the connection
        close(task.clientSocket);
    }
}

void HttpServer::addTask(const RequestTask& task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        taskQueue_.push(task);
    }
    queueCondition_.notify_one();
}

HttpServer::HttpRequest HttpServer::parseRequest(const std::string& request) {
    HttpRequest req;
    std::istringstream iss(request);
    std::string line;
    
    // Parse first line (method, path, HTTP version)
    if (std::getline(iss, line)) {
        std::istringstream lineStream(line);
        lineStream >> req.method >> req.path;
    }
    
    // Parse headers
    while (std::getline(iss, line) && !line.empty() && line != "\r") {
        std::string::size_type colonPos = line.find(':');
        if (colonPos != std::string::npos) {
            std::string key = line.substr(0, colonPos);
            std::string value = line.substr(colonPos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t\r") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t\r") + 1);
            
            req.headers[key] = value;
        }
    }
    
    // Find body (everything after empty line)
    size_t bodyStart = request.find("\r\n\r\n");
    if (bodyStart != std::string::npos) {
        req.body = request.substr(bodyStart + 4);
        
        // Check Content-Length header to determine actual body length
        if (req.headers.count("Content-Length") > 0) {
            int contentLength = std::stoi(req.headers["Content-Length"]);
            if (contentLength < req.body.size()) {
                req.body.resize(contentLength);
            }
        }
    }
    
    return req;
}

std::string HttpServer::generateResponse(const HttpResponse& response) {
    std::ostringstream ss;
    
    // Status line
    ss << "HTTP/1.1 " << response.statusCode << " ";
    switch (response.statusCode) {
        case 200: ss << "OK"; break;
        case 201: ss << "Created"; break;
        case 400: ss << "Bad Request"; break;
        case 401: ss << "Unauthorized"; break;
        case 403: ss << "Forbidden"; break;
        case 404: ss << "Not Found"; break;
        case 500: ss << "Internal Server Error"; break;
        default: ss << "Unknown"; break;
    }
    ss << "\r\n";
    
    // Headers
    for (const auto& header : response.headers) {
        ss << header.first << ": " << header.second << "\r\n";
    }
    
    // Add Content-Length header
    ss << "Content-Length: " << response.body.size() << "\r\n";
    
    // Empty line separating headers from body
    ss << "\r\n";
    
    // Body
    ss << response.body;
    
    return ss.str();
}

void HttpServer::handleConnection(int clientSocket) {
    // Read request
    char buffer[4096] = {0};
    ssize_t bytesRead = read(clientSocket, buffer, sizeof(buffer) - 1);
    if (bytesRead <= 0) {
        close(clientSocket);
        return;
    }
    
    // Parse request
    std::string requestStr(buffer, bytesRead);
    HttpRequest request = parseRequest(requestStr);
    
    // Call the API to handle the request
    std::string responseBody = api_->handleRequest(
        request.method, 
        request.path, 
        request.body
    );
    
    // Generate HTTP response
    HttpResponse response;
    response.statusCode = 200;
    response.body = responseBody;
    response.headers["Content-Type"] = "application/json";
    response.headers["Access-Control-Allow-Origin"] = "*";  // CORS
    
    // Send response
    std::string responseStr = generateResponse(response);
    send(clientSocket, responseStr.c_str(), responseStr.size(), 0);
    
    // Close connection
    close(clientSocket);
}

} // namespace api
} // namespace alphazero