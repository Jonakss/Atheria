// src/cpp_core/src/sparse_engine.cpp
#include "../include/sparse_engine.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace atheria {

// Helper function para determinar device correcto
static torch::Device determine_device(const std::string& device_str) {
    if (device_str == "cuda" && torch::cuda::is_available()) {
        return torch::kCUDA;
    } else {
        return torch::kCPU;
    }
}

Engine::Engine(int64_t d_state, const std::string& device_str, int64_t grid_size)
    : d_state_(d_state)
    , grid_size_(grid_size)
    , device_(determine_device(device_str))
    , model_loaded_(false)
    , vacuum_(d_state_, device_)
    , chunk_manager_(16) // CHUNK_SIZE = 16
    , step_count_(0)
    , last_error_("") {
}

Engine::~Engine() {
}

bool Engine::load_model(const std::string& model_path) {
    try {
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        torch::NoGradGuard no_grad;
        model_loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        last_error_ = std::string(e.what());
        model_loaded_ = false;
        return false;
    } catch (...) {
        last_error_ = "Error desconocido al cargar modelo";
        model_loaded_ = false;
        return false;
    }
}

void Engine::add_particle(const Coord3D& coord, const torch::Tensor& state) {
    torch::Tensor state_on_device = state.to(device_);
    matter_map_.insert_tensor(coord, state_on_device);
    
    // Activar chunk correspondiente
    chunk_manager_.activate_chunk_for_global_coord(coord);
    
    // Mantener active_region_ para compatibilidad con get_active_coords
    activate_neighborhood(coord);
}

torch::Tensor Engine::get_state_at(const Coord3D& coord) {
    if (matter_map_.contains_coord(coord)) {
        return matter_map_.get_tensor(coord);
    }
    return vacuum_.get_fluctuation(coord, step_count_);
}

int64_t Engine::step_native() {
    if (!model_loaded_) {
        step_count_++;
        return matter_map_.size();
    }
    
    step_count_++;
    
    SparseMap next_matter_map;
    ChunkManager next_chunk_manager(chunk_manager_.get_chunk_size());
    std::unordered_set<Coord3D, Coord3DHash> next_active_region; // Para compatibilidad
    
    // 1. Identificar chunks a procesar (activos + vecinos)
    std::unordered_set<ChunkCoord, ChunkCoordHash> chunks_to_process;
    const auto& active_chunks = chunk_manager_.get_active_chunks();
    
    for (const auto& chunk : active_chunks) {
        // Incluir el chunk activo y sus 26 vecinos (3x3x3)
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    chunks_to_process.insert({chunk.x + dx, chunk.y + dy, chunk.z + dz});
                }
            }
        }
    }
    
    // Constantes de chunking
    const int64_t CHUNK_SIZE = chunk_manager_.get_chunk_size();
    const int64_t PADDING = 2; // Padding para contexto del kernel
    const int64_t INPUT_SIZE = CHUNK_SIZE + 2 * PADDING;
    
    // 2. Procesar cada chunk
    for (const auto& chunk_coord : chunks_to_process) {
        Coord3D chunk_origin = chunk_manager_.get_chunk_origin(chunk_coord);
        
        // Construir tensor de entrada denso para el chunk + padding
        // Shape: [1, 2*d_state, INPUT_SIZE, INPUT_SIZE] (Slice 2D por ahora, asumiendo modelo 2D)
        // TODO: Soporte 3D real si el modelo es 3D
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        torch::Tensor chunk_input = torch::zeros({1, 2 * d_state_, INPUT_SIZE, INPUT_SIZE}, options);
        
        // Llenar input con estado actual (materia + vacío)
        // Iteramos sobre el área del chunk + padding
        for (int64_t ly = 0; ly < INPUT_SIZE; ly++) {
            for (int64_t lx = 0; lx < INPUT_SIZE; lx++) {
                // Coordenada global
                int64_t gx = chunk_origin.x - PADDING + lx;
                int64_t gy = chunk_origin.y - PADDING + ly;
                int64_t gz = chunk_origin.z; // Slice central Z del chunk
                
                Coord3D global_coord(gx, gy, gz);
                torch::Tensor state = get_state_at(global_coord);
                
                // Copiar a input tensor
                if (state.is_complex()) {
                    for (int64_t c = 0; c < d_state_; c++) {
                        chunk_input[0][c][ly][lx] = torch::real(state[c]).item<float>();
                        chunk_input[0][c + d_state_][ly][lx] = torch::imag(state[c]).item<float>();
                    }
                } else {
                     for (int64_t c = 0; c < d_state_; c++) {
                        chunk_input[0][c][ly][lx] = state[c].item<float>();
                    }
                }
            }
        }
        
        // Inferencia
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(chunk_input);
        
        torch::Tensor output;
        try {
            auto output_ivalue = model_.forward(inputs);
            if (output_ivalue.isTuple()) {
                output = output_ivalue.toTuple()->elements()[0].toTensor();
            } else if (output_ivalue.isTensor()) {
                output = output_ivalue.toTensor();
            }
        } catch (...) {
            continue; // Skip chunk on error
        }
        
        if (!output.defined()) continue;
        
        // Procesar salida (actualizar materia)
        // Solo la región central (CHUNK_SIZE), descartando padding
        // Output shape: [1, 2*d_state, INPUT_SIZE, INPUT_SIZE] (asumiendo padding='same' en modelo)
        
        // Verificar dimensiones de salida
        if (output.size(2) != INPUT_SIZE || output.size(3) != INPUT_SIZE) {
            // Si el modelo reduce dimensiones (no padding), ajustar offsets
            // Por ahora asumimos padding='same'
            continue; 
        }
        
        for (int64_t ly = 0; ly < CHUNK_SIZE; ly++) {
            for (int64_t lx = 0; lx < CHUNK_SIZE; lx++) {
                // Indices en el tensor de salida (saltando padding)
                int64_t oy = ly + PADDING;
                int64_t ox = lx + PADDING;
                
                // Coordenada global
                int64_t gx = chunk_origin.x + lx;
                int64_t gy = chunk_origin.y + ly;
                int64_t gz = chunk_origin.z;
                Coord3D global_coord(gx, gy, gz);
                
                // Extraer nuevo estado
                torch::Tensor new_state_real = output[0].slice(0, 0, d_state_).slice(1, oy, oy+1).slice(2, ox, ox+1).reshape({d_state_});
                torch::Tensor new_state_imag = output[0].slice(0, d_state_, 2*d_state_).slice(1, oy, oy+1).slice(2, ox, ox+1).reshape({d_state_});
                torch::Tensor new_state = torch::complex(new_state_real, new_state_imag);
                
                // Calcular energía
                float energy = torch::sum(torch::abs(new_state).pow(2)).item<float>();
                
                // Thresholding
                if (energy > 0.01f) {
                    next_matter_map.insert_tensor(global_coord, new_state);
                    next_chunk_manager.activate_chunk(chunk_coord);
                    
                    // Actualizar active_region_ para compatibilidad
                    // (Esto es costoso, tal vez deberíamos eliminar active_region_ y usar chunk_manager para todo)
                    // Por ahora, solo agregamos la coordenada actual
                    next_active_region.insert(global_coord);
                }
            }
        }
    }
    
    // Actualizar estado
    matter_map_ = std::move(next_matter_map);
    chunk_manager_ = std::move(next_chunk_manager);
    active_region_ = std::move(next_active_region); // Nota: esto ahora solo contiene celdas con materia, no vecinos vacíos
    
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
    // Implementación no usada en chunk-based step, pero mantenida por si acaso
    return neighbors;
}

void Engine::update_active_region(const std::vector<Coord3D>& coords, 
                                  std::unordered_set<Coord3D, Coord3DHash>& region) {
    // Implementación no usada en chunk-based step
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
    chunk_manager_.clear();
    step_count_ = 0;
}

std::vector<Coord3D> Engine::get_active_coords() const {
    return std::vector<Coord3D>(active_region_.begin(), active_region_.end());
}

std::string Engine::get_last_error() const {
    return last_error_;
}

} // namespace atheria

