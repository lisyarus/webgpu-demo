#pragma once

#include <deque>
#include <mutex>
#include <condition_variable>

template <typename T>
struct SynchronizedQueue
{
    void push(T value)
    {
        {
            std::lock_guard lock{mutex_};
            queue_.push_back(std::move(value));
        }
        cv_.notify_one();
    }

    std::deque<T> grab()
    {
        std::lock_guard lock{mutex_};
        return std::move(queue_);
    }

    T pop()
    {
        std::unique_lock lock{mutex_};
        cv_.wait(lock, [&]{ return !queue_.empty(); });
        auto value = queue_.front();
        queue_.pop_front();
        return value;
    }

private:
    std::mutex mutex_;
    std::deque<T> queue_;
    std::condition_variable cv_;
};
