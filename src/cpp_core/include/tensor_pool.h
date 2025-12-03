#ifndef ATHERIA_TENSOR_POOL_H
#define ATHERIA_TENSOR_POOL_H

#include <vector>
#include <stack>
#include <mutex>
#include <memory>
#include <torch/torch.h>
#include <omp.h>

namespace atheria {

/**
 * ThreadLocalTensorPool: A pool to reuse torch::Tensor objects, optimized for OpenMP.
 * 
 * Instead of a single pool with a mutex (which causes contention),
 * we maintain a separate pool for each thread.
 */
class ThreadLocalTensorPool {
public:
    ThreadLocalTensorPool(int64_t d_state, torch::Device device) 
        : d_state_(d_state), device_(device) {
        // Initialize pools for max threads
        int max_threads = omp_get_max_threads();
        pools_.resize(max_threads);
    }

    // Acquire a tensor from the current thread's pool
    torch::Tensor acquire() {
        int tid = omp_get_thread_num();
        // Ensure tid is within bounds (just in case)
        if (tid >= pools_.size()) return create_new();

        auto& pool = pools_[tid];
        if (pool.empty()) {
            return create_new();
        } else {
            torch::Tensor t = pool.top();
            pool.pop();
            return t;
        }
    }

    // Return a tensor to the current thread's pool
    void release(torch::Tensor t) {
        int tid = omp_get_thread_num();
        if (tid < pools_.size()) {
            pools_[tid].push(t);
        }
    }

    // Pre-allocate tensors for all threads
    void reserve(size_t n_per_thread) {
        for (auto& pool : pools_) {
            for (size_t i = 0; i < n_per_thread; ++i) {
                pool.push(create_new());
            }
        }
    }

    size_t size() const {
        size_t total = 0;
        for (const auto& pool : pools_) {
            total += pool.size();
        }
        return total;
    }

    void clear() {
        for (auto& pool : pools_) {
            while (!pool.empty()) pool.pop();
        }
    }

private:
    torch::Tensor create_new() {
        return torch::zeros({d_state_}, torch::TensorOptions().dtype(torch::kComplexFloat).device(device_));
    }

    int64_t d_state_;
    torch::Device device_;
    
    // Vector of stacks, one per thread.
    // Note: std::vector is not thread-safe for resizing, but we size it once in constructor.
    // Accessing different elements from different threads is safe.
    std::vector<std::stack<torch::Tensor>> pools_;
};

} // namespace atheria

#endif // ATHERIA_TENSOR_POOL_H
