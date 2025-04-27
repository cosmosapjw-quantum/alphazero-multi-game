// thread_pool.cpp
#include "alphazero/mcts/parallel_mcts.h"

namespace alphazero {
namespace mcts {

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    
                    // Wait for task or stop signal
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    
                    // Exit if stop signal and no tasks
                    if (this->stop && this->tasks.empty()) {
                        return;
                    }
                    
                    // Get task from front
                    task = std::move(this->tasks.front());
                    this->tasks.pop_back();
                }
                
                // Execute task
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    
    // Wake up all threads
    condition.notify_all();
    
    // Join all threads
    for (std::thread& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace mcts
} // namespace alphazero