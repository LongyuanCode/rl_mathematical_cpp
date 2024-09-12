#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace algorithm {
class ThreadPool {
 public:
  explicit ThreadPool(size_t num_threads);
  ~ThreadPool();

  // 添加任务到任务队列中
  template <class F>
  void enqueue(F&& task);

 private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;

  // 每个线程运行的函数，处理任务队列
  void worker_thread();
};

template <class F>
void ThreadPool::enqueue(F&& task) {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    tasks_.emplace(std::forward<F>(task));
  }
  condition_.notify_one();
}
}  // namespace algorithm
