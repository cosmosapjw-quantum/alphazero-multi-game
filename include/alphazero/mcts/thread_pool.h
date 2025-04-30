// thread_pool.h
#ifndef ALPHAZERO_MCTS_THREAD_POOL_H
#define ALPHAZERO_MCTS_THREAD_POOL_H

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <deque>
#include <memory>

namespace alphazero {
namespace mcts {

/**
 * @class ThreadPool
 * @brief A simple thread pool implementation for parallel task execution
 * 
 * This class manages a pool of worker threads and a task queue.
 * Tasks can be enqueued and will be executed by the next available thread.
 */
class ThreadPool {
public:
    /**
     * @brief Constructor
     * 
     * @param numThreads Number of worker threads to create
     */
    explicit ThreadPool(size_t numThreads);
    
    /**
     * @brief Destructor
     * 
     * Stops all threads and waits for them to finish
     */
    ~ThreadPool();
    
    /**
     * @brief Enqueue a task for execution
     * 
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return Future containing the result of the function
     */
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    /**
     * @brief Get the number of threads in the pool
     * 
     * @return The number of threads
     */
    size_t size() const { return workers.size(); }

private:
    // Worker threads
    std::vector<std::thread> workers;
    
    // Task queue
    std::deque<std::function<void()>> tasks;
    
    // Synchronization
    std::mutex queueMutex;
    std::condition_variable condition;
    
    // Flag to signal threads to stop
    bool stop;

    // Non-copyable and non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
};

// Template method implementation
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    
    using return_type = typename std::result_of<F(Args...)>::type;
    
    // Create a packaged task with the function and its arguments
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    // Get future from the task
    std::future<return_type> result = task->get_future();
    
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        
        // Don't allow enqueueing after stopping the pool
        if (stop) {
            throw std::runtime_error("Cannot enqueue on a stopped ThreadPool");
        }
        
        // Add the task to the queue
        tasks.emplace_back([task]() { (*task)(); });
    }
    
    // Wake up one worker thread
    condition.notify_one();
    
    return result;
}

} // namespace mcts
} // namespace alphazero

#endif // ALPHAZERO_MCTS_THREAD_POOL_H