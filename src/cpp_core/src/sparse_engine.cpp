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
    
    if (chunks_to_process.empty()) {
        return 0;
    }

    // Constantes de chunking
    const int64_t CHUNK_SIZE = chunk_manager_.get_chunk_size();
    const int64_t PADDING = 2; // Padding para contexto del kernel
    const int64_t INPUT_SIZE = CHUNK_SIZE + 2 * PADDING;
    
    // 2. Preparar Batch de Entrada
    std::vector<torch::Tensor> batch_inputs;
    std::vector<ChunkCoord> batch_coords;
    batch_inputs.reserve(chunks_to_process.size());
    batch_coords.reserve(chunks_to_process.size());

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    for (const auto& chunk_coord : chunks_to_process) {
        Coord3D chunk_origin = chunk_manager_.get_chunk_origin(chunk_coord);
        
        // Construir tensor de entrada denso para el chunk + padding
        // Construir tensor de entrada denso para el chunk + padding
        // Shape: [2*d_state, INPUT_SIZE, INPUT_SIZE]
        torch::Tensor chunk_input = torch::empty({2 * d_state_, INPUT_SIZE, INPUT_SIZE}, options);
        
        // 1. Llenar con VACÍO (Bulk Noise) para evitar llamadas costosas a get_fluctuation
        // Replicamos la lógica de HarmonicVacuum::get_fluctuation pero en batch
        float complex_noise_strength = 0.1f;
        torch::Tensor noise = torch::randn({d_state_, INPUT_SIZE, INPUT_SIZE}, options) * complex_noise_strength;
        torch::Tensor vacuum_real = torch::cos(noise);
        torch::Tensor vacuum_imag = torch::sin(noise);
        
        chunk_input.slice(0, 0, d_state_).copy_(vacuum_real);
        chunk_input.slice(0, d_state_, 2 * d_state_).copy_(vacuum_imag);
        
        // 2. Sobrescribir con MATERIA donde exista
        for (int64_t ly = 0; ly < INPUT_SIZE; ly++) {
            for (int64_t lx = 0; lx < INPUT_SIZE; lx++) {
                // Coordenada global
                int64_t gx = chunk_origin.x - PADDING + lx;
                int64_t gy = chunk_origin.y - PADDING + ly;
                int64_t gz = chunk_origin.z; // Slice central Z del chunk
                
                Coord3D global_coord(gx, gy, gz);
                
                // OPTIMIZACIÓN: Solo acceder al mapa si hay materia
                // Evita llamar a get_state_at que invoca vacuum individualmente
                if (matter_map_.contains_coord(global_coord)) {
                    torch::Tensor state = matter_map_.get_tensor(global_coord);
                    
                    // Copiar a input tensor
                    if (state.is_complex()) {
                        auto real_part = torch::real(state);
                        auto imag_part = torch::imag(state);
                        
                        chunk_input.slice(0, 0, d_state_).slice(1, ly, ly + 1).slice(2, lx, lx + 1).reshape({d_state_}).copy_(real_part);
                        chunk_input.slice(0, d_state_, 2 * d_state_).slice(1, ly, ly + 1).slice(2, lx, lx + 1).reshape({d_state_}).copy_(imag_part);
                    } else {
                        chunk_input.slice(0, 0, d_state_).slice(1, ly, ly + 1).slice(2, lx, lx + 1).reshape({d_state_}).copy_(state);
                    }
                }
            }
        }
        
        batch_inputs.push_back(chunk_input);
        batch_coords.push_back(chunk_coord);
    }
    
    // 3. Inferencia en Batch
    torch::Tensor batch_tensor = torch::stack(batch_inputs); // [B, 2*d_state, INPUT_SIZE, INPUT_SIZE]
    
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_tensor);
    
    torch::Tensor output_batch;
    try {
        auto output_ivalue = model_.forward(inputs);
        if (output_ivalue.isTuple()) {
            output_batch = output_ivalue.toTuple()->elements()[0].toTensor();
        } else if (output_ivalue.isTensor()) {
            output_batch = output_ivalue.toTensor();
        }
    } catch (...) {
        // Si falla la inferencia, abortamos este paso (o podríamos intentar fallback)
        return matter_map_.size(); 
    }
    
    if (!output_batch.defined() || output_batch.size(0) != batch_inputs.size()) {
        return matter_map_.size();
    }

    // 4. Procesar Salida y Dispersar
    // Output shape: [B, 2*d_state, INPUT_SIZE, INPUT_SIZE]
    
    // Verificar dimensiones de salida
    if (output_batch.size(2) != INPUT_SIZE || output_batch.size(3) != INPUT_SIZE) {
        // Si el modelo reduce dimensiones, no podemos mapear directamente por ahora
        return matter_map_.size();
    }
    
    // Mover output a CPU para iteración rápida si es necesario, 
    // pero idealmente mantenemos en GPU si insert_tensor soporta tensores GPU.
    // SparseMap::insert_tensor espera tensor, asumimos que maneja device.
    // Sin embargo, iterar por cada celda en C++ con tensores en GPU es lento por overhead de acceso.
    // MEJORA: Vectorizar la inserción o mantener todo en GPU. 
    // Por ahora, iteramos igual que antes pero desde el batch.
    
    for (size_t i = 0; i < batch_coords.size(); ++i) {
        const auto& chunk_coord = batch_coords[i];
        const auto& output_chunk = output_batch[i]; // [2*d_state, INPUT_SIZE, INPUT_SIZE]
        Coord3D chunk_origin = chunk_manager_.get_chunk_origin(chunk_coord);
        
        bool chunk_has_matter = false;

        for (int64_t ly = 0; ly < CHUNK_SIZE; ly++) {
            for (int64_t lx = 0; lx < CHUNK_SIZE; lx++) {
                // Indices en el tensor de salida (saltando padding)
                int64_t oy = ly + PADDING;
                int64_t ox = lx + PADDING;
                
                // Extraer nuevo estado
                // Slice es eficiente, devuelve vista
                torch::Tensor new_state_real = output_chunk.slice(0, 0, d_state_).slice(1, oy, oy+1).slice(2, ox, ox+1).reshape({d_state_});
                torch::Tensor new_state_imag = output_chunk.slice(0, d_state_, 2*d_state_).slice(1, oy, oy+1).slice(2, ox, ox+1).reshape({d_state_});
                torch::Tensor new_state = torch::complex(new_state_real, new_state_imag);
                
                // Calcular energía
                float energy = torch::sum(torch::abs(new_state).pow(2)).item<float>();
                
                // Thresholding
                if (energy > 0.01f) {
                    // Coordenada global
                    int64_t gx = chunk_origin.x + lx;
                    int64_t gy = chunk_origin.y + ly;
                    int64_t gz = chunk_origin.z;
                    Coord3D global_coord(gx, gy, gz);

                    next_matter_map.insert_tensor(global_coord, new_state);
                    next_active_region.insert(global_coord);
                    chunk_has_matter = true;
                }
            }
        }
        
        if (chunk_has_matter) {
            next_chunk_manager.activate_chunk(chunk_coord);
        }
    }
    
    // Actualizar estado
    matter_map_ = std::move(next_matter_map);
    chunk_manager_ = std::move(next_chunk_manager);
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

