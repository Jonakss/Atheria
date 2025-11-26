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

    // Convertir set a vector para acceso indexado
    std::vector<ChunkCoord> chunks_vector(chunks_to_process.begin(), chunks_to_process.end());
    
    // Preparar estructuras para el nuevo estado (acumuladas de todos los mini-batches)
    SparseMap next_matter_map;
    ChunkManager next_chunk_manager(chunk_manager_.get_chunk_size());
    std::unordered_set<Coord3D, Coord3DHash> next_active_region; // Para compatibilidad
    
    // Constantes de chunking
    const int64_t CHUNK_SIZE = chunk_manager_.get_chunk_size();
    const int64_t PADDING = 2; // Padding para contexto del kernel
    const int64_t INPUT_SIZE = CHUNK_SIZE + 2 * PADDING;
    
    // CONFIGURACIÓN DE MINI-BATCHING
    // Procesar chunks en grupos pequeños para evitar saturar la VRAM
    // 512 chunks * (20*20*float32*channels) es manejable
    const size_t MINI_BATCH_SIZE = 512; 
    size_t total_chunks = chunks_vector.size();
    
    // Iterar sobre mini-batches
    for (size_t batch_start = 0; batch_start < total_chunks; batch_start += MINI_BATCH_SIZE) {
        size_t current_batch_size = std::min(MINI_BATCH_SIZE, total_chunks - batch_start);
        
        // 2. Preparar Batch de Entrada (Mini-batch)
        // Shape: [B, 2*d_state, INPUT_SIZE, INPUT_SIZE]
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        torch::Tensor batch_tensor = torch::empty({(long)current_batch_size, 2 * d_state_, INPUT_SIZE, INPUT_SIZE}, options);
        
        // Llenar con VACÍO (Bulk Noise)
        float complex_noise_strength = 0.1f;
        torch::Tensor noise = torch::randn({(long)current_batch_size, d_state_, INPUT_SIZE, INPUT_SIZE}, options) * complex_noise_strength;
        torch::Tensor vacuum_real = torch::cos(noise);
        torch::Tensor vacuum_imag = torch::sin(noise);
        
        batch_tensor.slice(1, 0, d_state_).copy_(vacuum_real);
        batch_tensor.slice(1, d_state_, 2 * d_state_).copy_(vacuum_imag);
        
        // Recolectar materia de los chunks del mini-batch actual
        std::vector<torch::Tensor> all_matter;
        std::vector<int64_t> idx_b, idx_y, idx_x;
        // Reservar memoria estimada
        size_t estimated_matter = current_batch_size * INPUT_SIZE * INPUT_SIZE * 0.1;
        all_matter.reserve(estimated_matter);
        idx_b.reserve(estimated_matter);
        idx_y.reserve(estimated_matter);
        idx_x.reserve(estimated_matter);

        for (int64_t b = 0; b < (int64_t)current_batch_size; ++b) {
            const auto& chunk_coord = chunks_vector[batch_start + b];
            Coord3D chunk_origin = chunk_manager_.get_chunk_origin(chunk_coord);
            
            for (int64_t ly = 0; ly < INPUT_SIZE; ly++) {
                for (int64_t lx = 0; lx < INPUT_SIZE; lx++) {
                    int64_t gx = chunk_origin.x - PADDING + lx;
                    int64_t gy = chunk_origin.y - PADDING + ly;
                    int64_t gz = chunk_origin.z;
                    
                    Coord3D global_coord(gx, gy, gz);
                    
                    if (matter_map_.contains_coord(global_coord)) {
                        all_matter.push_back(matter_map_.get_tensor(global_coord));
                        idx_b.push_back(b); // Índice relativo al mini-batch
                        idx_y.push_back(ly);
                        idx_x.push_back(lx);
                    }
                }
            }
        }
        
        // Aplicar materia al batch_tensor
        if (!all_matter.empty()) {
            torch::Tensor stacked_matter = torch::stack(all_matter); // [TotalN, d_state]
            
            auto indices_b = torch::tensor(idx_b, torch::kLong).to(device_);
            auto indices_y = torch::tensor(idx_y, torch::kLong).to(device_);
            auto indices_x = torch::tensor(idx_x, torch::kLong).to(device_);
            
            if (stacked_matter.is_complex()) {
                auto real_part = torch::real(stacked_matter).t(); // [d, TotalN]
                auto imag_part = torch::imag(stacked_matter).t(); // [d, TotalN]
                
                for (int64_t c = 0; c < d_state_; ++c) {
                    auto c_tensor = torch::tensor(c, torch::kLong).to(device_);
                    auto c_imag_tensor = torch::tensor(c + d_state_, torch::kLong).to(device_);
                    
                    // Real: [b, c, y, x] = real_part[c]
                    batch_tensor.index_put_({indices_b, c_tensor, indices_y, indices_x}, real_part[c]);
                    
                    // Imag: [b, c+d, y, x] = imag_part[c]
                    batch_tensor.index_put_({indices_b, c_imag_tensor, indices_y, indices_x}, imag_part[c]);
                }
            } else {
                auto matter_t = stacked_matter.t(); // [d, TotalN]
                for (int64_t c = 0; c < d_state_; ++c) {
                    auto c_tensor = torch::tensor(c, torch::kLong).to(device_);
                    batch_tensor.index_put_({indices_b, c_tensor, indices_y, indices_x}, matter_t[c]);
                }
            }
        }
        
        // 3. Inferencia en Mini-Batch
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
        } catch (const std::exception& e) {
            std::cerr << "Inference error in mini-batch: " << e.what() << std::endl;
            // Continuar con el siguiente batch o abortar?
            // Por ahora continuamos, pero marcamos error
            last_error_ = e.what();
            continue; 
        }
        
        if (!output_batch.defined()) continue;

        // 4. Procesar Salida y Dispersar (Masked Scatter)
        // output_batch: [B, 2*d, H, W]
        float vacuum_threshold = 0.01f; 
        torch::Tensor magnitude = torch::abs(output_batch).mean(1); // [B, H, W]
        torch::Tensor active_mask = magnitude > vacuum_threshold; // [B, H, W] bool
        
        // Mover máscara a CPU para iterar
        auto active_mask_cpu = active_mask.to(torch::kCPU);
        auto active_accessor = active_mask_cpu.accessor<bool, 3>();
        
        for (int64_t b = 0; b < (int64_t)current_batch_size; ++b) {
            const auto& chunk_coord = chunks_vector[batch_start + b];
            Coord3D chunk_origin = chunk_manager_.get_chunk_origin(chunk_coord);
            
            bool chunk_has_matter = false;
            for (int64_t ly = 0; ly < INPUT_SIZE; ly++) {
                for (int64_t lx = 0; lx < INPUT_SIZE; lx++) {
                    if (!active_accessor[b][ly][lx]) {
                        continue;
                    }
                    
                    int64_t gx = chunk_origin.x - PADDING + lx;
                    int64_t gy = chunk_origin.y - PADDING + ly;
                    int64_t gz = chunk_origin.z;
                    Coord3D global_coord(gx, gy, gz);
                    
                    // Extract tensor
                    torch::Tensor new_state_flat = output_batch.slice(0, 0, current_batch_size).slice(2, ly, ly+1).slice(3, lx, lx+1).select(0, b).squeeze();
                    
                    // Reconstruct complex
                    torch::Tensor final_state;
                    if (d_state_ * 2 == new_state_flat.size(0)) {
                        auto real = new_state_flat.slice(0, 0, d_state_);
                        auto imag = new_state_flat.slice(0, d_state_, 2 * d_state_);
                        final_state = torch::complex(real, imag);
                    } else {
                        final_state = new_state_flat;
                    }
                    
                    next_matter_map.insert_tensor(global_coord, final_state.clone());
                    next_active_region.insert(global_coord);
                    chunk_has_matter = true;
                }
            }
            
            if (chunk_has_matter) {
                next_chunk_manager.activate_chunk(chunk_coord);
            }
        }
    } // Fin loop mini-batches
    
    // Actualizar estado global
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

torch::Tensor Engine::get_dense_tensor(const std::vector<int64_t>& roi) {
    // 1. Determinar dimensiones y ROI
    int64_t x_min = 0, y_min = 0;
    int64_t x_max = grid_size_, y_max = grid_size_;
    
    if (!roi.empty() && roi.size() >= 4) {
        x_min = std::max((int64_t)0, roi[0]);
        y_min = std::max((int64_t)0, roi[1]);
        x_max = std::min(grid_size_, roi[2]);
        y_max = std::min(grid_size_, roi[3]);
    }
    
    int64_t width = x_max - x_min;
    int64_t height = y_max - y_min;
    
    // Validar dimensiones
    if (width <= 0 || height <= 0) {
        auto options = torch::TensorOptions().dtype(torch::kComplexFloat).device(device_);
        return torch::zeros({1, 0, 0, d_state_}, options);
    }
    
    // 2. Generar Vacío (Ruido Determinista por Step)
    // Usamos step_count_ como semilla para que el fondo sea consistente en el mismo paso
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    
    // Guardar semilla anterior
    auto gen = torch::globalContext().defaultGenerator(device_);
    auto prev_seed = gen.current_seed();
    
    // Establecer semilla basada en step_count_
    uint64_t seed = (uint64_t)step_count_ * 2654435761; 
    torch::manual_seed(seed);
    
    // Generar ruido: [1, H, W, d_state]
    float strength = 0.1f;
    torch::Tensor noise = torch::randn({1, height, width, d_state_}, options) * strength;
    
    // Convertir a complejo: cos(noise) + i*sin(noise)
    torch::Tensor real = torch::cos(noise);
    torch::Tensor imag = torch::sin(noise);
    torch::Tensor dense = torch::complex(real, imag);
    
    // Restaurar semilla
    torch::manual_seed(prev_seed);
    
    // 3. Superponer Materia
    // Obtener todas las coordenadas con materia
    std::vector<Coord3D> active_coords = matter_map_.coord_keys();
    
    // Recolectar índices y tensores para operación batched
    std::vector<torch::Tensor> values;
    std::vector<int64_t> idx_b, idx_y, idx_x;
    
    // Reservar memoria aproximada
    size_t estimated = std::min(active_coords.size(), (size_t)(width * height));
    values.reserve(estimated);
    idx_b.reserve(estimated);
    idx_y.reserve(estimated);
    idx_x.reserve(estimated);
    
    for (const auto& coord : active_coords) {
        // Filtrar por ROI y Z=0
        if (coord.z == 0 && 
            coord.x >= x_min && coord.x < x_max &&
            coord.y >= y_min && coord.y < y_max) {
            
            // Obtener tensor de materia
            torch::Tensor state = matter_map_.get_tensor(coord);
            
            // Guardar para batch update
            values.push_back(state);
            idx_b.push_back(0); // Batch 0
            idx_y.push_back(coord.y - y_min); // Relativo a ROI
            idx_x.push_back(coord.x - x_min); // Relativo a ROI
        }
    }
    
    // Aplicar actualizaciones en batch si hay materia
    if (!values.empty()) {
        torch::Tensor stacked_values = torch::stack(values); // [N, d_state]
        
        auto indices_b = torch::tensor(idx_b, torch::kLong).to(device_);
        auto indices_y = torch::tensor(idx_y, torch::kLong).to(device_);
        auto indices_x = torch::tensor(idx_x, torch::kLong).to(device_);
        
        // dense[indices_b, indices_y, indices_x] = stacked_values
        dense.index_put_({indices_b, indices_y, indices_x}, stacked_values);
    }
    
    return dense;
}

} // namespace atheria

