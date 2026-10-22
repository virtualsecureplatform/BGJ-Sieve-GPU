#pragma once
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <vector>

// LIFO preserves the reducer's existing scheduling policy. Closing wakes all
// consumers, which drain published tasks before observing completion.
template<class T> class bounded_task_queue {
    std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<T> tasks_;
    const size_t capacity_;
    bool closed_ = false;
public:
    explicit bounded_task_queue(size_t capacity) : capacity_(capacity) {
        tasks_.reserve(capacity);
    }
    void push(T task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_ || tasks_.size() == capacity_)
                throw std::runtime_error("reducer task queue closed/full");
            tasks_.push_back(task);
        }
        ready_.notify_one();
    }
    bool pop(T &task) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return closed_ || !tasks_.empty(); });
        if (tasks_.empty()) return false;
        task = tasks_.back();
        tasks_.pop_back();
        return true;
    }
    void close() {
        { std::lock_guard<std::mutex> lock(mutex_); closed_ = true; }
        ready_.notify_all();
    }
};
