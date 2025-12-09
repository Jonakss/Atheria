// src/cpp_core/src/sparse_engine.cpp
#include "../include/sparse_engine.h"
#include <algorithm>
#include <stdexcept>
#include <omp.h> // OpenMP re-enabled
#include "../include/morton_utils.h"
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
    , pool_(d_state_, determine_device(device_str))
    , step_count_(0)
    , last_error_("") {
    // Device ya está determinado y vacuum inicializado correctamente
    // grid_size se usa para construir inputs del modelo con el tamaño correcto
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
        // Error al cargar el modelo - guardar mensaje de error
        last_error_ = std::string(e.what());
        model_loaded_ = false;
        return false;
    } catch (...) {
        // Error desconocido
        last_error_ = "Error desconocido al cargar modelo";
        model_loaded_ = false;
        return false;
    }
}

void Engine::add_particle(const Coord3D& coord, const torch::Tensor& state) {
    // Asegurar que el tensor esté en el dispositivo correcto
    torch::Tensor state_on_device = state.to(device_);
    
    // Almacenar en el mapa
    matter_map_.insert_tensor(coord, state_on_device);
    
    // Actualizar índice espacial
    octree_.insert(coord);

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
        auto coord_keys = matter_map_.coord_keys();
        active_region_.clear();
        for (const auto& coord : coord_keys) {
            activate_neighborhood(coord);
        }
        step_count_++;
        return matter_map_.size();
    }
    
    step_count_++;
    
    // Asegurar que el octree esté construido antes de usarlo (para futuras optimizaciones)
    octree_.build();
    
    // Crear nuevo mapa para el siguiente estado
    SparseMap next_matter_map;
    std::unordered_set<Coord3D, Coord3DHash> next_active_region;
    
    // Procesar todas las coordenadas activas
    std::vector<Coord3D> processed_coords(active_region_.begin(), active_region_.end());
    
    // OPTIMIZATION: Sort by Morton code for spatial locality
    const int64_t OFFSET = 1048576; 
    std::sort(processed_coords.begin(), processed_coords.end(), [OFFSET](const Coord3D& a, const Coord3D& b) {
        // ... (sorting logic)
        int64_t ax = std::max(int64_t(0), std::min(a.x + OFFSET, int64_t(2097151)));
        int64_t ay = std::max(int64_t(0), std::min(a.y + OFFSET, int64_t(2097151)));
        int64_t az = std::max(int64_t(0), std::min(a.z + OFFSET, int64_t(2097151)));
        
        int64_t bx = std::max(int64_t(0), std::min(b.x + OFFSET, int64_t(2097151)));
        int64_t by = std::max(int64_t(0), std::min(b.y + OFFSET, int64_t(2097151)));
        int64_t bz = std::max(int64_t(0), std::min(b.z + OFFSET, int64_t(2097151)));

        return coord_to_morton(ax, ay, az) < coord_to_morton(bx, by, bz);
    });
    
    // Agrupar coordenadas para procesamiento por batch
    const int64_t batch_size = 32; // Procesar en batches de 32
    
    // Parallel region
    #pragma omp parallel
    {
        // Evitar que LibTorch lance su propio pool de threads dentro de OpenMP
        // torch::set_num_threads(1); // CAUSES DEADLOCK

        // Thread-local storage
        std::vector<Coord3D> local_batch_coords;
        std::vector<torch::Tensor> local_batch_states;
        SparseMap local_next_matter_map;
        std::unordered_set<Coord3D, Coord3DHash> local_next_active_region;
        
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < processed_coords.size(); i++) {
                const Coord3D& coord = processed_coords[i];
                
                // Obtener estado actual (materia o vacío)
                // get_state_at es thread-safe porque solo lee de matter_map_ (que es const durante este paso)
                // y vacuum_.get_fluctuation (que debe ser thread-safe o stateless)
                torch::Tensor current_state = get_state_at(coord);
                
                // Calcular energía
                float energy = torch::sum(torch::abs(current_state).pow(2)).item<float>();
                
                if (energy > 0.01f) { // Umbral de existencia
                    // Guardar para batch processing local
                    local_batch_coords.push_back(coord);
                    local_batch_states.push_back(current_state);
                    
                    // Si el batch local está lleno, procesar
                    if (local_batch_coords.size() >= static_cast<size_t>(batch_size)) {
                        try {
                            // Construir entrada del batch
                            torch::Tensor batch_input = build_batch_input(local_batch_coords);
                            
                            // Ejecutar inferencia con el modelo
                            // IMPORTANTE: torch::jit::script::Module::forward es thread-safe para inferencia
                            // pero debemos usar NoGradGuard localmente
                            torch::NoGradGuard no_grad;
                            std::vector<torch::jit::IValue> inputs;
                            inputs.push_back(batch_input);

                            torch::Tensor batch_output;
                            // std::cout << "DEBUG: Running model forward for batch of size " << local_batch_coords.size() << std::endl;
                            auto output_ivalue = model_.forward(inputs);
                            // std::cout << "DEBUG: Model forward done." << std::endl;

                            if (output_ivalue.isTuple()) {
                                auto output_tuple = output_ivalue.toTuple();
                                if (output_tuple->elements().size() > 0) {
                                    batch_output = output_tuple->elements()[0].toTensor();
                                } else {
                                    // En paralelo no podemos lanzar excepciones fácilmente, loggear o ignorar
                                    local_batch_coords.clear();
                                    local_batch_states.clear();
                                    continue;
                                }
                            } else if (output_ivalue.isTensor()) {
                                batch_output = output_ivalue.toTensor();
                            } else {
                                local_batch_coords.clear();
                                local_batch_states.clear();
                                continue;
                            }

                            // Procesar salida del modelo
                            int64_t patch_size = 3; // Debe coincidir con build_batch_input
                            int64_t center_idx = patch_size / 2;

                            for (size_t j = 0; j < local_batch_coords.size(); j++) {
                                // Extract center output
                                torch::Tensor output_center = batch_output[j].select(2, center_idx).select(1, center_idx);

                                // Get views of delta (Real/Imag) - No Allocation
                                auto delta_real = output_center.slice(0, 0, d_state_);
                                auto delta_imag = output_center.slice(0, d_state_, 2 * d_state_);
                                
                                // Acquire reuseable tensor from pool (avoid new_state allocation)
                                torch::Tensor candidate = pool_.acquire();
                                
                                // Initialize with current state
                                candidate.copy_(local_batch_states[j]);
                                
                                // Apply delta in-place using functional views
                                torch::real(candidate).add_(delta_real);
                                torch::imag(candidate).add_(delta_imag);

                                // Calculate norm efficiently
                                float norm = torch::norm(candidate).item<float>();
                                
                                if (norm > 1e-6f) {
                                    candidate.div_(norm); // In-place normalization
                                }

                                // Filter low energy
                                if (norm > 0.01f) {
                                    local_next_matter_map.insert_tensor(local_batch_coords[j], candidate);
                                    update_active_region({local_batch_coords[j]}, local_next_active_region);
                                } else {
                                    pool_.release(candidate);
                                }
                            }
                        } catch (const c10::Error& e) {
                             std::cerr << "LibTorch error in batch processing: " << e.msg() << std::endl;
                        } catch (const std::exception& e) {
                             std::cerr << "Standard exception in batch processing: " << e.what() << std::endl;
                        } catch (...) {
                             std::cerr << "Unknown error in batch processing." << std::endl;
                        }

                        // Limpiar batch local
                        local_batch_coords.clear();
                        local_batch_states.clear();
                    }
                }
        } // Fin del loop for

        // Procesar batch restante (si quedó algo)
        if (!local_batch_coords.empty()) {
             // Construir entrada del batch
            try {
                // Construir entrada del batch
                torch::Tensor batch_input = build_batch_input(local_batch_coords);
                
                torch::NoGradGuard no_grad;
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(batch_input);
                
                torch::Tensor batch_output;
                auto output_ivalue = model_.forward(inputs);

                if (output_ivalue.isTuple()) {
                    auto output_tuple = output_ivalue.toTuple();
                    if (output_tuple->elements().size() > 0) {
                        batch_output = output_tuple->elements()[0].toTensor();
                    }
                } else if (output_ivalue.isTensor()) {
                    batch_output = output_ivalue.toTensor();
                }
                
                if (batch_output.defined()) {
                    int64_t patch_size = 3; // Debe coincidir con build_batch_input
                    int64_t center_idx = patch_size / 2;

                    for (size_t j = 0; j < local_batch_coords.size(); j++) {
                        torch::Tensor output_center = batch_output[j].select(2, center_idx).select(1, center_idx);
                        
                        // Views
                        auto delta_real = output_center.slice(0, 0, d_state_);
                        auto delta_imag = output_center.slice(0, d_state_, 2 * d_state_);
                        
                        // Acquire from pool
                        torch::Tensor candidate = pool_.acquire();
                        
                        // Init
                        candidate.copy_(local_batch_states[j]);
                        
                        // In-place update using functional views
                        torch::real(candidate).add_(delta_real);
                        torch::imag(candidate).add_(delta_imag);
                        
                        // Norm
                        float norm = torch::norm(candidate).item<float>();
                        
                        if (norm > 1e-6f) {
                            candidate.div_(norm);
                        }
                        
                        if (norm > 0.01f) {
                            local_next_matter_map.insert_tensor(local_batch_coords[j], candidate);
                            update_active_region({local_batch_coords[j]}, local_next_active_region);
                        } else {
                            pool_.release(candidate);
                        }
                    }
                }
            } catch (const c10::Error& e) {
                std::cerr << "LibTorch error in final batch: " << e.msg() << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Standard exception in final batch: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "Unknown error in final batch." << std::endl;
            }
        }

        // Merge de resultados thread-safe
        #pragma omp critical
        {
            // Merge matter map
            auto local_coords = local_next_matter_map.coord_keys();
            for (const auto& coord : local_coords) {
                next_matter_map.insert_tensor(coord, local_next_matter_map.get_tensor(coord));
            }

            // Merge active region
            for (const auto& coord : local_next_active_region) {
                next_active_region.insert(coord);
            }
        }
    } // End parallel region
    
    // Recycle old tensors
    // Before overwriting matter_map_, we should return its tensors to the pool.
    // Note: We need to be careful about thread safety if we did this in parallel,
    // but here we are in the main thread (after parallel region).
    
    // However, matter_map_ holds tensors. We can iterate and release them.
    // BUT: SparseMap doesn't expose iterators to tensors directly easily without copy.
    // We added coord_keys(). Let's use that or add a method to SparseMap to "drain" tensors?
    // For now, let's iterate keys.
    auto old_coords = matter_map_.coord_keys();
    for (const auto& coord : old_coords) {
        pool_.release(matter_map_.get_tensor(coord));
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
    int64_t batch_size = static_cast<int64_t>(coords.size());
    int64_t patch_size = 3;
    int64_t patch_radius = patch_size / 2;
    int64_t height = patch_size;
    int64_t width = patch_size;

    // Pre-allocate vector to avoid re-allocations
    std::vector<torch::Tensor> collected_tensors;
    collected_tensors.reserve(batch_size * height * width);

    for (int64_t i = 0; i < batch_size; i++) {
        const Coord3D& center = coords[i];
        for (int64_t py = 0; py < height; py++) {
            for (int64_t px = 0; px < width; px++) {
                int64_t dx = static_cast<int64_t>(px) - patch_radius;
                int64_t dy = static_cast<int64_t>(py) - patch_radius;
                Coord3D patch_coord(center.x + dx, center.y + dy, center.z);
                
                torch::Tensor state = get_state_at(patch_coord);
                // Ensure state is 1D [d_state]
                if (state.dim() > 1) state = state.flatten();
                
                collected_tensors.push_back(state);
            }
        }
    }
    
    if (collected_tensors.empty()) {
         auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
         return torch::zeros({batch_size, 2 * d_state_, height, width}, options);
    }

    // Stack all tensors efficiently: [Batch*H*W, d_state]
    auto stacked = torch::stack(collected_tensors);
    
    // Handle Complex -> Real/Imag conversion
    torch::Tensor stacked_real, stacked_imag;
    if (stacked.is_complex()) {
        stacked_real = torch::real(stacked);
        stacked_imag = torch::imag(stacked);
    } else {
        stacked_real = stacked;
        stacked_imag = torch::zeros_like(stacked);
    }
    
    // Concatenate channels: [Batch*H*W, 2*d_state]
    auto combined = torch::cat({stacked_real, stacked_imag}, 1);
    
    // Reshape to [Batch, Height, Width, 2*d_state]
    auto reshaped = combined.view({batch_size, height, width, 2 * d_state_});
    
    // Permute to [Batch, 2*d_state, Height, Width] (NCHW format)
    auto permuted = reshaped.permute({0, 3, 1, 2});
    
    return permuted.contiguous();
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
    octree_.clear();
    active_region_.clear();
    step_count_ = 0;
}

std::vector<Coord3D> Engine::get_active_coords() const {
    return std::vector<Coord3D>(active_region_.begin(), active_region_.end());
}

std::string Engine::get_last_error() const {
    return last_error_;
}

torch::Tensor Engine::compute_visualization(const std::string& viz_type) {
    // Crear tensor de salida [H, W]
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_);
        
    torch::Tensor viz_tensor = torch::zeros({grid_size_, grid_size_}, options);
    
    // Iterar sobre el mapa disperso
    // Esto es mucho más eficiente que convertir todo a denso si la ocupación es baja
    auto coord_keys = matter_map_.coord_keys();
    
    // Usar accessor para acceso rápido a CPU si el tensor está en CPU
    // Si está en GPU, necesitamos otra estrategia o simplemente usar indexación de tensores
    // Por simplicidad y compatibilidad inicial, usaremos indexación de tensores que funciona en ambos
    
    // TODO: Para máxima performance en GPU, esto debería ser un kernel CUDA custom
    // o usar operaciones dispersas de PyTorch si matter_map_ fuera un tensor disperso.
    // Por ahora, iteramos en C++ que es más rápido que Python.
    
    if (viz_type == "density") {
        for (const auto& coord : coord_keys) {
            // Verificar límites (solo visualizamos slice Z=0 por ahora)
            if (coord.z != 0 || coord.x < 0 || coord.x >= grid_size_ || coord.y < 0 || coord.y >= grid_size_) {
                continue;
            }
            
            torch::Tensor state = matter_map_.get_tensor(coord);
            // Densidad = sum(|psi|^2)
            float density = torch::sum(torch::abs(state).pow(2)).item<float>();
            
            // Asignar al tensor de visualización
            // Nota: viz_tensor[y][x]
            viz_tensor[coord.y][coord.x] = density;
        }
    } else if (viz_type == "phase") {
        for (const auto& coord : coord_keys) {
            if (coord.z != 0 || coord.x < 0 || coord.x >= grid_size_ || coord.y < 0 || coord.y >= grid_size_) {
                continue;
            }
            
            torch::Tensor state = matter_map_.get_tensor(coord);
            // Fase del primer componente (o promedio ponderado)
            // Usamos el primer componente como referencia de fase global
            float phase = torch::angle(state[0]).item<float>();
            
            // Normalizar fase a [0, 1] para visualización (opcional, el frontend suele esperar radianes o normalizado)
            // Dejamos en radianes [-pi, pi]
            viz_tensor[coord.y][coord.x] = phase;
        }
    } else if (viz_type == "energy") {
        // Similar a densidad pero podría aplicar alguna transformación
        for (const auto& coord : coord_keys) {
            if (coord.z != 0 || coord.x < 0 || coord.x >= grid_size_ || coord.y < 0 || coord.y >= grid_size_) {
                continue;
            }
            
            torch::Tensor state = matter_map_.get_tensor(coord);
            float energy = torch::sum(torch::abs(state).pow(2)).item<float>();
            viz_tensor[coord.y][coord.x] = energy;
        }
    } else if (viz_type == "holographic") {
        // Holographic: 3 Channels [R, G, B]
        // R: Energy (Normalized)
        // G: Phase (Mapped to 0-1)
        // B: Real Part / Structure
        
        // Output tensor: [H, W, 3] -> But typically we return [H, W, C] or [C, H, W]?
        // Python wrapper expects [H, W, C] usually or we need to align with conventions.
        // Let's create [H, W, 3]
        
        // Re-create viz_tensor with 3 channels
        viz_tensor = torch::zeros({grid_size_, grid_size_, 3}, options);
        
        for (const auto& coord : coord_keys) {
            if (coord.z != 0 || coord.x < 0 || coord.x >= grid_size_ || coord.y < 0 || coord.y >= grid_size_) {
                continue;
            }
            
            torch::Tensor state = matter_map_.get_tensor(coord);
            
            // Channel 1 (Red): Energy / Magnitude (Log scale option?)
            float mag_sq = torch::sum(torch::abs(state).pow(2)).item<float>();
            // Simple normalization attempt (local) - global norm handled by frontend usually
            // but for safety let's clamp
            float r_val = std::min(1.0f, mag_sq * 5.0f); // *5 boost
            
            // Channel 2 (Green): Phase
            float phase = torch::angle(state[0]).item<float>(); 
            // Map [-pi, pi] -> [0, 1]
            float g_val = (phase + M_PI) / (2.0f * M_PI);
            
            // Channel 3 (Blue): Real part dominance or Imag part?
            // Let's use real part of first component
            float real_val = torch::real(state[0]).item<float>();
            float b_val = (real_val + 1.0f) * 0.5f; // [-1,1] -> [0,1]
            b_val = std::max(0.0f, std::min(1.0f, b_val));

            viz_tensor[coord.y][coord.x][0] = r_val;
            viz_tensor[coord.y][coord.x][1] = g_val;
            viz_tensor[coord.y][coord.x][2] = b_val;
        }

    } else if (viz_type == "holographic_bulk") {
        // 1. Compute Base Density (Z=0)
        for (const auto& coord : coord_keys) {
            if (coord.z != 0 || coord.x < 0 || coord.x >= grid_size_ || coord.y < 0 || coord.y >= grid_size_) {
                continue;
            }
            torch::Tensor state = matter_map_.get_tensor(coord);
            // Action Density / Magnitude
            float density = torch::sum(torch::abs(state).pow(2)).item<float>();
            viz_tensor[coord.y][coord.x] = density;
        }
        
        // 2. Generate Bulk (Gaussian Renormalization Flow)
        // Ensure inputs are on device
        if (viz_tensor.device() != device_) viz_tensor = viz_tensor.to(device_);
        
        // Prepare Kernel (3x3 Gaussian)
        auto kernel = torch::tensor({
            {0.0625f, 0.125f, 0.0625f},
            {0.125f, 0.25f,  0.125f},
            {0.0625f, 0.125f, 0.0625f}
        }, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
        kernel = kernel.reshape({1, 1, 3, 3});
        
        // Setup input [1, 1, H, W]
        torch::Tensor current = viz_tensor.unsqueeze(0).unsqueeze(0);
        
        std::vector<torch::Tensor> layers;
        int bulk_depth = 5; // Default depth for native
        layers.reserve(bulk_depth);
        layers.push_back(current);
        
        for (int i = 1; i < bulk_depth; ++i) {
             current = torch::conv2d(current, kernel, {}, {1}, {1}, {1}, 1);
             layers.push_back(current);
        }
        
        // Flatten layers for simple return or return stacked?
        // Python expects: { "data": flat_array, "shape": [D, H, W] }
        // We return Tensor. Python wrapper handles the rest.
        // Return [1, Depth, H, W] --> Squeeze to [Depth, H, W]
        auto bulk_stack = torch::cat(layers, 1).squeeze(0);
        return bulk_stack; // [Depth, H, W]
    }
    
    return viz_tensor;
}

std::vector<Coord3D> Engine::query_radius(const Coord3D& center, int radius) const {
    Coord3D min(center.x - radius, center.y - radius, center.z - radius);
    Coord3D max(center.x + radius, center.y + radius, center.z + radius);
    
    // Use Octree for efficient box query
    // Note: This returns all points in the bounding box.
    // We could filter strictly by radius here if needed, but for "neighborhood" 
    // in this grid context, box is often what we want (Moore neighborhood).
    // If strict Euclidean radius is needed, we can filter the result.
    return octree_.query_box(min, max);
}

} // namespace atheria

