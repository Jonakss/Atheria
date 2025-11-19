// src/cpp_core/src/sparse_engine.cpp
#include "../include/sparse_engine.h"
#include <algorithm>
#include <stdexcept>

namespace atheria {

Engine::Engine(int64_t d_state, const std::string& device_str)
    : d_state_(d_state)
    , device_(device_str == "cuda" ? torch::kCUDA : torch::kCPU)
    , model_loaded_(false)
    , vacuum_(d_state_, device_)
    , step_count_(0) {
    
    // Mover el modelo al dispositivo si está disponible
    if (device_ == torch::kCUDA && !torch::cuda::is_available()) {
        device_ = torch::kCPU;
    }
}

Engine::~Engine() {
    // Destructor por defecto es suficiente
}

bool Engine::load_model(const std::string& model_path) {
    try {
        // Cargar modelo TorchScript
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        
        // Mover a modo de inferencia (no gradientes)
        torch::NoGradGuard no_grad;
        
        model_loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        // Error al cargar el modelo
        model_loaded_ = false;
        return false;
    }
}

void Engine::add_particle(const Coord3D& coord, const torch::Tensor& state) {
    // Asegurar que el tensor esté en el dispositivo correcto
    torch::Tensor state_on_device = state.to(device_);
    
    // Almacenar en el mapa
    matter_map_.insert_tensor(coord, state_on_device);
    
    // Activar vecindario
    activate_neighborhood(coord);
}

torch::Tensor Engine::get_state_at(const Coord3D& coord) {
    // Verificar si hay materia en esta coordenada
    if (matter_map_.contains_coord(coord)) {
        return matter_map_.get_tensor(coord);
    }
    
    // Si no hay materia, devolver fluctuación del vacío
    return vacuum_.get_fluctuation(coord, step_count_);
}

int64_t Engine::step_native() {
    if (!model_loaded_) {
        // Sin modelo, solo conservar las partículas existentes
        std::vector<Coord3D> coords_to_process(matter_map_.coord_keys().begin(), 
                                               matter_map_.coord_keys().end());
        active_region_.clear();
        for (const auto& coord : coords_to_process) {
            activate_neighborhood(coord);
        }
        step_count_++;
        return matter_map_.size();
    }
    
    step_count_++;
    
    // Crear nuevo mapa para el siguiente estado
    SparseMap next_matter_map;
    std::unordered_set<Coord3D, Coord3DHash> next_active_region;
    
    // Procesar todas las coordenadas activas
    std::vector<Coord3D> processed_coords(active_region_.begin(), active_region_.end());
    
    // Agrupar coordenadas para procesamiento por batch
    // Por ahora procesamos una por una, pero esto puede optimizarse para batch processing
    std::vector<torch::Tensor> batch_inputs;
    std::vector<Coord3D> batch_coords;
    std::vector<torch::Tensor> batch_current_states;
    
    const int64_t batch_size = 32; // Procesar en batches de 32
    
    for (size_t i = 0; i < processed_coords.size(); i++) {
        const Coord3D& coord = processed_coords[i];
        
        // Obtener estado actual (materia o vacío)
        torch::Tensor current_state = get_state_at(coord);
        
        // Calcular energía
        float energy = torch::sum(torch::abs(current_state).pow(2)).item<float>();
        
        if (energy > 0.01f) { // Umbral de existencia
            // Preparar entrada para el modelo
            // El modelo espera un tensor de forma [batch, channels, H, W] o similar
            // Para el motor disperso, necesitamos adaptar la forma
            
            // Por ahora, simplemente conservamos el estado actual
            // TODO: Implementar procesamiento real con el modelo
            // Esto requiere entender la forma exacta que espera el modelo
            
            // Guardar para batch processing
            batch_coords.push_back(coord);
            batch_current_states.push_back(current_state);
            
            // Si el batch está lleno o es el último, procesar
            if (batch_coords.size() >= static_cast<size_t>(batch_size) || 
                i == processed_coords.size() - 1) {
                
                // Construir entrada del batch
                torch::Tensor batch_input = build_batch_input(batch_coords);
                
                // Ejecutar inferencia con el modelo
                torch::NoGradGuard no_grad;
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(batch_input);
                
                torch::Tensor batch_output = model_.forward(inputs).toTensor();
                
                // Procesar salida del modelo
                // El modelo devuelve [batch, 2*d_state, 3, 3]
                // Extraer el centro del patch (posición [1, 1])
                int64_t center_idx = 1 * 3 + 1; // Centro del patch 3x3
                
                for (size_t j = 0; j < batch_coords.size(); j++) {
                    // Extraer salida del centro del patch
                    torch::Tensor delta_real = batch_output[j].slice(0, 0, d_state_).slice(1, 1, 2).slice(2, 1, 2).squeeze();
                    torch::Tensor delta_imag = batch_output[j].slice(0, d_state_, 2*d_state_).slice(1, 1, 2).slice(2, 1, 2).squeeze();
                    // Crear tensor complejo usando torch::complex
                    // torch::complex espera tensores reales y devuelve un tensor complejo
                    torch::Tensor delta_complex = torch::complex(delta_real, delta_imag);
                    
                    // Aplicar delta al estado actual
                    torch::Tensor new_state = batch_current_states[j] + delta_complex;
                    
                    // Normalizar si es necesario
                    float norm = torch::sum(torch::abs(new_state).pow(2)).item<float>();
                    if (norm > 1e-6f) {
                        new_state = new_state / std::sqrt(norm);
                    }
                    
                    // Almacenar en el siguiente mapa
                    next_matter_map.insert_tensor(batch_coords[j], new_state);
                    
                    // Activar vecinos
                    update_active_region({batch_coords[j]}, next_active_region);
                }
                
                // Limpiar batch
                batch_coords.clear();
                batch_current_states.clear();
            }
        }
    }
    
    // Actualizar estado
    matter_map_ = std::move(next_matter_map);
    active_region_ = std::move(next_active_region);
    
    return matter_map_.size();
}

void Engine::activate_neighborhood(const Coord3D& coord, int radius) {
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dz = -radius; dz <= radius; dz++) {
                Coord3D neighbor(coord.x + dx, coord.y + dy, coord.z + dz);
                active_region_.insert(neighbor);
            }
        }
    }
}

std::vector<Coord3D> Engine::get_neighbors(const Coord3D& center, int radius) const {
    std::vector<Coord3D> neighbors;
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dz = -radius; dz <= radius; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue; // Excluir el centro
                neighbors.emplace_back(center.x + dx, center.y + dy, center.z + dz);
            }
        }
    }
    return neighbors;
}

torch::Tensor Engine::build_batch_input(const std::vector<Coord3D>& coords) {
    // Construir batch input para el modelo
    // El modelo espera: [batch, 2*d_state, H, W] donde H=W=3 (patch 3x3)
    int64_t batch_size = static_cast<int64_t>(coords.size());
    int64_t height = 3;  // Patch 3x3 alrededor de cada partícula
    int64_t width = 3;
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_)
        .requires_grad(false);
    
    // Crear tensor de entrada: [batch, 2*d_state, 3, 3]
    torch::Tensor batch_input = torch::zeros({batch_size, 2 * d_state_, height, width}, options);
    
    // Para cada coordenada, construir un patch 3x3
    for (int64_t i = 0; i < batch_size; i++) {
        const Coord3D& center = coords[i];
        
        // Obtener vecindario 3x3
        std::vector<Coord3D> patch_coords;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                patch_coords.emplace_back(center.x + dx, center.y + dy, center.z);
            }
        }
        
        // Construir patch: para cada posición en el patch, obtener estado
        for (int64_t py = 0; py < height; py++) {
            for (int64_t px = 0; px < width; px++) {
                const Coord3D& patch_coord = patch_coords[py * width + px];
                torch::Tensor state = get_state_at(patch_coord);
                
                // Convertir estado complejo a [real, imag] concatenado
                // Si el estado es complejo, extraer real e imag
                torch::Tensor real, imag;
                if (state.is_complex()) {
                    real = torch::real(state);
                    imag = torch::imag(state);
                } else {
                    // Si no es complejo, asumir que es real
                    real = state;
                    imag = torch::zeros_like(state);
                }
                
                // Copiar a batch_input [batch, channels, y, x]
                // Asegurarse de que real e imag tengan la forma correcta
                if (real.dim() == 1 && real.size(0) == d_state_) {
                    // Expandir a [1, d_state, 1, 1]
                    real = real.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
                    imag = imag.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
                    
                    batch_input[i].slice(0, 0, d_state_).slice(1, py, py+1).slice(2, px, px+1) = real;
                    batch_input[i].slice(0, d_state_, 2*d_state_).slice(1, py, py+1).slice(2, px, px+1) = imag;
                }
            }
        }
    }
    
    return batch_input;
}

void Engine::update_active_region(const std::vector<Coord3D>& coords, 
                                   std::unordered_set<Coord3D, Coord3DHash>& region) {
    for (const auto& coord : coords) {
        std::vector<Coord3D> neighbors = get_neighbors(coord, 1);
        for (const auto& neighbor : neighbors) {
            region.insert(neighbor);
        }
    }
}

int64_t Engine::get_matter_count() const {
    return matter_map_.size();
}

int64_t Engine::get_step_count() const {
    return step_count_;
}

void Engine::clear() {
    matter_map_.clear();
    active_region_.clear();
    step_count_ = 0;
}

} // namespace atheria

