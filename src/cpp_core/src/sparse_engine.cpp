// src/cpp_core/src/sparse_engine.cpp
#include "../include/sparse_engine.h"
#include <algorithm>
#include <stdexcept>
#include <omp.h> // OpenMP re-enabled
#include "../include/morton_utils.h"

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
    // This ensures that particles close in space are processed together,
    // improving cache locality and reducing memory jumps.
    const int64_t OFFSET = 1048576; 
    std::sort(processed_coords.begin(), processed_coords.end(), [OFFSET](const Coord3D& a, const Coord3D& b) {
        // Clamp to valid range for Morton encoding
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
        torch::set_num_threads(1);

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
                            auto output_ivalue = model_.forward(inputs);

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
                            int64_t center_idx = grid_size_ / 2;

                            for (size_t j = 0; j < local_batch_coords.size(); j++) {
                                // Extraer salida del centro del patch
                                torch::Tensor output_center = batch_output[j].select(2, center_idx).select(1, center_idx);

                                // Dividir en real e imag
                                torch::Tensor delta_real = output_center.slice(0, 0, d_state_);
                                torch::Tensor delta_imag = output_center.slice(0, d_state_, 2 * d_state_);
                                torch::Tensor delta_complex = torch::complex(delta_real, delta_imag);

                                // Obtener estado actual
                                torch::Tensor current_state_j = local_batch_states[j];
                                if (!current_state_j.is_complex()) {
                                    current_state_j = torch::complex(current_state_j, torch::zeros_like(current_state_j));
                                }

                                // Aplicar delta
                                torch::Tensor new_state = current_state_j + delta_complex;

                                // Normalizar
                                torch::Tensor abs_squared = torch::abs(new_state).pow(2);
                                float norm = torch::sum(abs_squared).item<float>();
                                if (norm > 1e-6f) {
                                    new_state = new_state / std::sqrt(norm);
                                }

                                // Filtrar estados con energía muy baja
                                if (norm > 0.01f) {
                                    local_next_matter_map.insert_tensor(local_batch_coords[j], new_state);
                                    update_active_region({local_batch_coords[j]}, local_next_active_region);
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
            torch::Tensor batch_input = build_batch_input(local_batch_coords);
            
            torch::NoGradGuard no_grad;
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(batch_input);
            
            torch::Tensor batch_output;
            try {
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
                    int64_t center_idx = grid_size_ / 2;

                    for (size_t j = 0; j < local_batch_coords.size(); j++) {
                        torch::Tensor output_center = batch_output[j].select(2, center_idx).select(1, center_idx);
                        torch::Tensor delta_real = output_center.slice(0, 0, d_state_);
                        torch::Tensor delta_imag = output_center.slice(0, d_state_, 2 * d_state_);
                        torch::Tensor delta_complex = torch::complex(delta_real, delta_imag);
                        
                        torch::Tensor current_state_j = local_batch_states[j];
                        if (!current_state_j.is_complex()) {
                            current_state_j = torch::complex(current_state_j, torch::zeros_like(current_state_j));
                        }
                        
                        torch::Tensor new_state = current_state_j + delta_complex;
                        torch::Tensor abs_squared = torch::abs(new_state).pow(2);
                        float norm = torch::sum(abs_squared).item<float>();
                        if (norm > 1e-6f) {
                            new_state = new_state / std::sqrt(norm);
                        }
                        
                        if (norm > 0.01f) {
                            local_next_matter_map.insert_tensor(local_batch_coords[j], new_state);
                            update_active_region({local_batch_coords[j]}, local_next_active_region);
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
    // IMPORTANTE: El modelo UNet fue entrenado con inputs de tamaño completo del grid.
    // Para mantener compatibilidad con los skip connections, necesitamos usar el mismo
    // tamaño que el modelo espera. El modelo típicamente espera inputs de 64x64 o similar.
    //
    // SOLUCIÓN: Usar el tamaño del grid con el que fue entrenado el modelo.
    // Este tamaño se pasa al constructor del Engine y se almacena en grid_size_.

    int64_t batch_size = static_cast<int64_t>(coords.size());

    // Usar el tamaño del grid con el que fue entrenado el modelo
    int64_t patch_size = grid_size_;  // Tamaño del patch (debe coincidir con el tamaño de entrenamiento)
    int64_t patch_radius = patch_size / 2;  // Radio del patch (para centrar)

    int64_t height = patch_size;
    int64_t width = patch_size;

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_)
        .requires_grad(false);

    // Crear tensor de entrada: [batch, 2*d_state, patch_size, patch_size]
    torch::Tensor batch_input = torch::zeros({batch_size, 2 * d_state_, height, width}, options);

    // Para cada coordenada, construir un patch centrado
    for (int64_t i = 0; i < batch_size; i++) {
        const Coord3D& center = coords[i];

        // Obtener vecindario del tamaño del patch
        for (int64_t py = 0; py < height; py++) {
            for (int64_t px = 0; px < width; px++) {
                // Mapear posición del patch a coordenada global
                // El centro del patch está en (patch_radius, patch_radius)
                int64_t dx = static_cast<int64_t>(px) - patch_radius;
                int64_t dy = static_cast<int64_t>(py) - patch_radius;

                Coord3D patch_coord(center.x + dx, center.y + dy, center.z);

                // Obtener estado en esta coordenada (materia o vacío)
                torch::Tensor state = get_state_at(patch_coord);

                // Convertir estado complejo a [real, imag] concatenado
                torch::Tensor real, imag;
                if (state.is_complex()) {
                    real = torch::real(state);
                    imag = torch::imag(state);
                } else {
                    real = state;
                    imag = torch::zeros_like(state);
                }

                // Copiar a batch_input [batch, channels, y, x]
                if (real.dim() == 1 && real.size(0) == d_state_) {
                    for (int64_t c = 0; c < d_state_; c++) {
                        batch_input[i][c][py][px] = real[c].item<float>();
                        batch_input[i][c + d_state_][py][px] = imag[c].item<float>();
                    }
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

