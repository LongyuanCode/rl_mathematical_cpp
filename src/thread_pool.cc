#include "thread_pool.h"

namespace algorithm {
ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
  for (int i = 0; i < num_threads; ++i) {
    workers_.emplace_back([this] { worker_thread(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    stop_ = true;
  }
  condition_.notify_all();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

void ThreadPool::worker_thread() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      // Unlock queue_mutex_ when work thread is waiting and lock it when work
      // thread is woke up.
      condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
      if (stop_ && tasks_.empty()) return;
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();  // 执行任务
  }
}
}  // namespace algorithm