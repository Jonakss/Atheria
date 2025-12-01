// src/cpp_core/include/sparse_engine.h
#ifndef ATHERIA_SPARSE_ENGINE_H
#define ATHERIA_SPARSE_ENGINE_H

#include "sparse_map.h"
#include "octree.h"
#include "tensor_pool.h" // Added
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

/**
 * Engine: Motor de simulación disperso de alto rendimiento.
 *
 * Esta clase ejecuta completamente en C++:
 * - Almacenamiento de tensores en SparseMap
 * - Generación de vacío cuántico
 * - Inferencia neuronal (TorchScript)
 * - Evolución física
 *
 * Python solo llama a step_native() y C++ hace todo el trabajo.
 */
class Engine {
public:
    Engine(int64_t d_state, const std::string& device_str = "cpu", int64_t grid_size = 64);
    ~Engine();
    
    /**
     * Carga un modelo TorchScript desde un archivo .pt
     */
    bool load_model(const std::string& model_path);

    /**
     * Agrega una partícula en las coordenadas dadas
     */
    void add_particle(const Coord3D& coord, const torch::Tensor& state);

    /**
     * Obtiene el estado en una coordenada (materia o vacío)
     */
    torch::Tensor get_state_at(const Coord3D& coord);

    /**
     * Ejecuta un paso completo de la simulación en C++.
     * Este método hace TODO el trabajo pesado sin volver a Python:
     * - Identifica regiones activas
     * - Genera vacío para vecinos
     * - Construye batches de entrada
     * - Ejecuta inferencia neuronal
     * - Actualiza el mapa
     *
     * @return Número de partículas activas después del paso
     */
    int64_t step_native();

    /**
     * Obtiene información del estado del motor
     */
    int64_t get_matter_count() const;
    int64_t get_step_count() const;

    /**
     * Limpia toda la materia
     */
    void clear();

    /**
     * Activa el vecindario de una coordenada
     */
    void activate_neighborhood(const Coord3D& coord, int radius = 1);
    
    /**
     * Obtiene todas las coordenadas activas en el motor
     */
    std::vector<Coord3D> get_active_coords() const;
    
    /**
     * Obtiene el último mensaje de error (si hubo un error al cargar el modelo)
     */
    std::string get_last_error() const;

    /**
     * Query particles within a radius
     */
    std::vector<Coord3D> query_radius(const Coord3D& center, int radius) const;

    /**
     * Calcula la visualización directamente en C++ para evitar conversiones costosas.
     * 
     * @param viz_type Tipo de visualización ("density", "phase", "energy")
     * @return Tensor denso [H, W] con los valores calculados
     */
    torch::Tensor compute_visualization(const std::string& viz_type);
    
private:
    // Configuración
    int64_t d_state_;
    int64_t grid_size_;  // Tamaño del grid para construir inputs del modelo
    torch::Device device_;
    torch::jit::script::Module model_;
    bool model_loaded_;
    
    // Almacenamiento
    SparseMap matter_map_;
    HarmonicVacuum vacuum_;
    TensorPool pool_; // Memory pool for tensors
    OctreeIndex octree_;
    
    // Estado de simulación
    std::unordered_set<Coord3D, Coord3DHash> active_region_;
    int64_t step_count_;
    
    // Manejo de errores
    std::string last_error_;
    
    // Helpers internos
    std::vector<Coord3D> get_neighbors(const Coord3D& center, int radius = 1) const;
    torch::Tensor build_batch_input(const std::vector<Coord3D>& coords);
    void update_active_region(const std::vector<Coord3D>& coords, 
                              std::unordered_set<Coord3D, Coord3DHash>& region);
};

} // namespace atheria

#endif // ATHERIA_SPARSE_ENGINE_H

