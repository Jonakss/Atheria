#ifndef ATHERIA_BASE_ENGINE_H
#define ATHERIA_BASE_ENGINE_H

#include <torch/torch.h>
#include <string>
#include <map>

namespace atheria {

/**
 * BaseEngine: Interface abstracta para todos los motores nativos.
 * Permite polimorfismo en el wrapper de Python.
 */
class BaseEngine {
public:
    virtual ~BaseEngine() = default;

    /**
     * Avanza la simulación un paso.
     * @return int64_t Número de elementos activos (o -1 si no aplica)
     */
    virtual int64_t step() = 0;

    /**
     * Obtiene el estado actual como tensor.
     * @return Tensor [B, C, H, W] o similar
     */
    virtual torch::Tensor get_state() = 0;
    
    /**
     * Carga un modelo TorchScript.
     */
    virtual bool load_model(const std::string& model_path) = 0;

    /**
     * Aplica una herramienta (tool) al estado.
     */
    virtual bool apply_tool(const std::string& action, const std::map<std::string, float>& params) = 0;
};

} // namespace atheria

#endif // ATHERIA_BASE_ENGINE_H
