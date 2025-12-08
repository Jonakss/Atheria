#ifndef ATHERIA_DENSE_ENGINE_H
#define ATHERIA_DENSE_ENGINE_H

#include "base_engine.h"
#include <torch/script.h>

namespace atheria {

/**
 * DenseEngine: Motor nativo para simulaciones densas (Cartesian/Polar).
 * Mantiene el estado como un tensor contiguo [1, C, H, W].
 */
class DenseEngine : public BaseEngine {
public:
    DenseEngine(int64_t grid_size, int64_t d_state, const std::string& device_str = "cpu");
    ~DenseEngine() override = default;

    int64_t step() override;
    torch::Tensor get_state() override;
    bool load_model(const std::string& model_path) override;
    bool apply_tool(const std::string& action, const std::map<std::string, float>& params) override;

    // Métodos específicos de DenseEngine
    void set_state(torch::Tensor new_state);
    
    // Holographic Principle
    torch::Tensor generate_bulk_state(torch::Tensor base_field, int64_t depth);
    
    
    // Getters
    int64_t get_step_count() const { return step_count_; }

private:
    int64_t grid_size_;
    int64_t d_state_;
    torch::Device device_;
    
    torch::Tensor state_; // [1, C, H, W] (Complex64)
    
    torch::jit::script::Module model_;
    bool model_loaded_;
    int64_t step_count_;
    
    // Helpers
    void ensure_device();
    
    // Optimization: Buffer pre-allocated
    torch::Tensor input_buffer_;
};

} // namespace atheria

#endif // ATHERIA_DENSE_ENGINE_H
