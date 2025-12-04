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
    
    torch::NoGradGuard no_grad; // Deshabilitar gradientes para inferencia
    
    try {
        // Preparar input para el modelo
        // El modelo espera [1, 2*C, H, W] (concatenado real+imag)
        auto real = torch::real(state_);
        auto imag = torch::imag(state_);
        
        // Fix for torch::cat: Explicitly create vector
        std::vector<torch::Tensor> tensors = {real, imag};
        auto input = torch::cat(tensors, 1); // Concatenar en canales
        
        // Ejecutar modelo
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        
        // Asumimos que el modelo retorna delta_psi (complejo simulado)
        // Salida: [1, 2*C, H, W]
        auto output = model_.forward(inputs).toTensor();
        
        // Reconstruir complejo: delta_psi
        auto out_real = output.slice(1, 0, d_state_);
        auto out_imag = output.slice(1, d_state_, 2 * d_state_);
        auto delta_psi = torch::complex(out_real, out_imag);
        
        // Euler step: psi_new = psi + delta_psi
        state_ = state_ + delta_psi;
        
        // Normalización (opcional pero recomendada para estabilidad)
        // Norm per pixel: sqrt(sum(|psi|^2))
        auto norm = torch::sqrt(state_.abs().pow(2).sum(1, true)); // Sum over channels
        state_ = state_ / (norm + 1e-9);
        
        step_count_++;
        return grid_size_ * grid_size_; // Retornar total de celdas (denso)
        
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

} // namespace atheria
