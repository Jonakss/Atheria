// src/cpp_core/include/sparse_engine.h
#ifndef ATHERIA_SPARSE_ENGINE_H
#define ATHERIA_SPARSE_ENGINE_H

#include "chunk_manager.h"
#include "sparse_map.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <unordered_set>
#include <vector>

namespace atheria {

/**
 * HarmonicVacuum: Generador procedural del vacío cuántico en C++.
 * 
 * En QFT, el vacío no es cero, es un estado de mínima energía con fluctuaciones.
 * Esta clase genera fluctuaciones deterministas usando operaciones tensorales de LibTorch.
 */
class HarmonicVacuum {
public:
    HarmonicVacuum(int64_t d_state, torch::Device device);
    
    /**
     * Genera una fluctuación determinista para una coordenada.
     * Esto asegura que el vacío sea consistente (si vuelves al mismo sitio, el ruido es el mismo).
     */
    torch::Tensor get_fluctuation(const Coord3D& coord, int64_t step_count);
    
private:
    int64_t d_state_;
    torch::Device device_;
};

class Engine {
public:
    Engine(int64_t d_state, const std::string& device_str = "cpu", int64_t grid_size = 64);
    ~Engine();
    
    // ... (public methods remain unchanged)
    
    bool load_model(const std::string& model_path);
    void add_particle(const Coord3D& coord, const torch::Tensor& state);
    torch::Tensor get_state_at(const Coord3D& coord);
    int64_t step_native();
    int64_t get_matter_count() const;
    int64_t get_step_count() const;
    void clear();
    void activate_neighborhood(const Coord3D& coord, int radius = 1);
    std::vector<Coord3D> get_active_coords() const;
    std::string get_last_error() const;
    
private:
    // Configuración
    int64_t d_state_;
    int64_t grid_size_;
    torch::Device device_;
    torch::jit::script::Module model_;
    bool model_loaded_;
    
    // Almacenamiento
    SparseMap matter_map_;
    HarmonicVacuum vacuum_;
    ChunkManager chunk_manager_; // [NEW] Gestor de chunks
    
    // Estado de simulación
    std::unordered_set<Coord3D, Coord3DHash> active_region_;
    int64_t step_count_;
    
    // Manejo de errores
    std::string last_error_;
    
    // Helpers internos
    std::vector<Coord3D> get_neighbors(const Coord3D& center, int radius = 1) const;
    // build_batch_input removed/replaced by chunk logic inside step_native
    void update_active_region(const std::vector<Coord3D>& coords, 
                              std::unordered_set<Coord3D, Coord3DHash>& region);
};

} // namespace atheria

#endif // ATHERIA_SPARSE_ENGINE_H

