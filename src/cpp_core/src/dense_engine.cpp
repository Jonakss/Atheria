#include "../include/dense_engine.h"
#include <iostream>

namespace atheria {

DenseEngine::DenseEngine(int64_t grid_size, int64_t d_state, const std::string& device_str)
    : grid_size_(grid_size), d_state_(d_state), device_(torch::kCPU), model_loaded_(false), step_count_(0) {
    
    if (device_str == "cuda" && torch::cuda::is_available()) {
        device_ = torch::kCUDA;
    }
    
    // Inicializar estado con ceros (complex64)
    // [1, C, H, W] format for PyTorch models (Channels First)
    state_ = torch::zeros({1, d_state, grid_size, grid_size}, 
                          torch::TensorOptions().dtype(torch::kComplexFloat).device(device_));
                          
    // Pre-allocate input buffer [1, 2*C, H, W] (Float)
    input_buffer_ = torch::zeros({1, 2 * d_state_, grid_size, grid_size}, 
                                 torch::TensorOptions().dtype(torch::kFloat32).device(device_));
}

void DenseEngine::ensure_device() {
    if (state_.device() != device_) {
        state_ = state_.to(device_);
    }
}

bool DenseEngine::load_model(const std::string& model_path) {
    try {
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        model_loaded_ = true;
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

int64_t DenseEngine::step() {
    if (!model_loaded_) return 0;
    
    torch::NoGradGuard no_grad; 
    
    try {
        // Ensure input buffer matches state size (in case state_ changed size)
        if (input_buffer_.size(2) != state_.size(2) || input_buffer_.size(3) != state_.size(3)) {
            input_buffer_ = torch::zeros({1, 2 * d_state_, state_.size(2), state_.size(3)}, 
                                         torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        }
    
        // Copy Real/Imag parts to buffer (Avoids allocation of 'input')
        using namespace torch::indexing;
        input_buffer_.index({Slice(), Slice(0, d_state_), Slice(), Slice()}).copy_(torch::real(state_));
        input_buffer_.index({Slice(), Slice(d_state_, 2 * d_state_), Slice(), Slice()}).copy_(torch::imag(state_));
        
        // Execute model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_buffer_); // Use buffer
        
        // Output: [1, 2*C, H, W]
        auto output = model_.forward(inputs).toTensor();
        
        // Reconstruct complex delta_psi
        // Optimization: Create views, don't copy
        auto out_real = output.slice(1, 0, d_state_);
        auto out_imag = output.slice(1, d_state_, 2 * d_state_);
        auto delta_psi = torch::complex(out_real, out_imag);
        
        // In-place update: state_ += delta_psi
        state_.add_(delta_psi);
        
        // Normalization (In-place where possible)
        // norm = sqrt(sum(|psi|^2)) over channel dim
        // Use keepdim=true to allow broadcasting
        // Replaced torch::linalg::vector_norm with torch::norm for compatibility
        auto norm = torch::norm(state_, 2, {1}, true);
        
        // state_ /= (norm + eps)
        state_.div_(norm.add_(1e-9));
        
        step_count_++;
        return state_.size(2) * state_.size(3);
        
    } catch (const c10::Error& e) {
        std::cerr << "Error in step(): " << e.what() << std::endl;
        return -1;
    }
}

torch::Tensor DenseEngine::get_state() {
    return state_;
}

void DenseEngine::set_state(torch::Tensor new_state) {
    if (new_state.device() != device_) {
        new_state = new_state.to(device_);
    }
    // Asegurar formato [1, C, H, W]
    if (new_state.dim() == 3) {
        new_state = new_state.unsqueeze(0);
    }
    state_ = new_state;
}

bool DenseEngine::apply_tool(const std::string& action, const std::map<std::string, float>& params) {
    // Implementación básica de tools en C++ (o placeholders)
    // Por ahora retornamos false para que Python maneje el fallback si es necesario
    // O implementamos lógica simple aquí.
    
    // TODO: Implementar tools nativos
    return false; 
}

torch::Tensor DenseEngine::generate_bulk_state(torch::Tensor base_field, int64_t depth) {
    // Asegurar dimensiones [1, C, H, W]
    torch::Tensor current = base_field;
    if (current.dim() == 2) current = current.view({1, 1, current.size(0), current.size(1)});
    else if (current.dim() == 3) current = current.unsqueeze(0);
    
    // Mover a device
    if (current.device() != device_) current = current.to(device_);
    
    // Preparar Kernel Gaussiano 3x3
    // [[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16.0
    auto kernel = torch::tensor({
        {0.0625f, 0.125f, 0.0625f},
        {0.125f, 0.25f,  0.125f},
        {0.0625f, 0.125f, 0.0625f}
    }, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    
    kernel = kernel.reshape({1, 1, 3, 3});
    
    // Ajustar kernel a canales (Grouped Conv: input channels = groups)
    int64_t channels = current.size(1);
    kernel = kernel.repeat({channels, 1, 1, 1});
    
    std::vector<torch::Tensor> layers;
    layers.reserve(depth);
    layers.push_back(current);
    
    // Iterative Blurring (Renormalization Flow)
    for (int i = 1; i < depth; ++i) {
        // Padding=1 para mantener tamaño
        current = torch::conv2d(current, kernel, {}, {1}, {1}, {1}, channels);
        layers.push_back(current);
    }
    
    // Retornar stack [1, Depth*C, H, W]
    // Si C=1 -> [1, Depth, H, W]
    return torch::cat(layers, 1);
}

} // namespace atheria
