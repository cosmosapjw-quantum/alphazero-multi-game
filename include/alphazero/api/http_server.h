// include/alphazero/api/http_server.h
#ifndef HTTP_SERVER_H
#define HTTP_SERVER_H

#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include "alphazero/api/rest_api.h"

namespace alphazero {
namespace api {

/**
 * @brief Simple HTTP server for the REST API
 */
class HttpServer {
public:
    /**
     * @brief Constructor
     * 
     * @param port Port to listen on
     * @param api REST API handler
     */
    HttpServer(int port, std::shared_ptr<RestApi> api);
    
    /**
     * @brief Destructor
     */
    ~HttpServer();
    
    /**
     * @brief Start the server
     * 
     * @return true if started successfully, false otherwise
     */
    bool start();
    
    /**
     * @brief Stop the server
     */
    void stop();
    
    /**
     * @brief Check if server is running
     * 
     * @return true if running, false otherwise
     */
    bool isRunning() const;
    
private:
    int port_;
    std::shared_ptr<RestApi> api_;
    std::atomic<bool> running_;
    std::thread serverThread_;
    
    // Socket descriptor
    int serverSocket_;
    
    // Connection handling
    void serverLoop();
    void handleConnection(int clientSocket);
    
    // HTTP request parsing
    struct HttpRequest {
        std::string method;
        std::string path;
        std::string body;
        std::map<std::string, std::string> headers;
    };
    
    // HTTP response generation
    struct HttpResponse {
        int statusCode;
        std::string body;
        std::map<std::string, std::string> headers;
    };
    
    HttpRequest parseRequest(const std::string& request);
    std::string generateResponse(const HttpResponse& response);
    
    // Worker threads for handling requests
    static const int NUM_WORKERS = 4;
    struct RequestTask {
        int clientSocket;
        HttpRequest request;
    };
    
    std::queue<RequestTask> taskQueue_;
    std::vector<std::thread> workerThreads_;
    std::mutex queueMutex_;
    std::condition_variable queueCondition_;
    std::atomic<bool> stopWorkers_;
    
    void workerThread();
    void addTask(const RequestTask& task);
};

} // namespace api
} // namespace alphazero

#endif // HTTP_SERVER_H