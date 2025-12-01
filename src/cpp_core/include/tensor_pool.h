#ifndef ATHERIA_TENSOR_POOL_H
#define ATHERIA_TENSOR_POOL_H

#include <vector>
#include <stack>
#include <mutex>
#include <torch/torch.h>

namespace atheria {

/**
 * TensorPool: A simple pool to reuse torch::Tensor objects.
 * 
 * Allocating and deallocating millions of small tensors (state vectors) 
 * causes significant overhead and memory fragmentation.
 * This pool keeps a stack of unused tensors to be recycled.
 */
class TensorPool {
public:
    TensorPool(int64_t d_state, torch::Device device) 
        : d_state_(d_state), device_(device) {}

    // Acquire a tensor from the pool or create a new one
    torch::Tensor acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.empty()) {
            // Create new tensor if pool is empty
            // Initialize with zeros or undefined? 
            // Better undefined for speed, but let's do zeros for safety initially.
            // Actually, we usually overwrite it immediately, so undefined is better.
            // But complex tensors need careful initialization.
            // Let's return a zero-filled complex tensor.
            return torch::zeros({d_state_}, torch::TensorOptions().dtype(torch::kComplexFloat).device(device_));
        } else {
            torch::Tensor t = pool_.top();
            pool_.pop();
            return t;
        }
    }

    // Return a tensor to the pool
    void release(torch::Tensor t) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Ideally we should check if t matches d_state_ and device_, 
        // but for performance we assume the caller is correct.
        pool_.push(t);
    }

    // Pre-allocate tensors
    void reserve(size_t n) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < n; ++i) {
            pool_.push(torch::zeros({d_state_}, torch::TensorOptions().dtype(torch::kComplexFloat).device(device_)));
        }
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pool_.empty()) {
            pool_.pop();
        }
    }

private:
    int64_t d_state_;
    torch::Device device_;
    std::stack<torch::Tensor> pool_;
    mutable std::mutex mutex_;
};

} // namespace atheria

#endif // ATHERIA_TENSOR_POOL_H
