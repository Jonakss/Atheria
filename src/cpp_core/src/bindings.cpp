// src/cpp_core/src/bindings.cpp
// IMPORTANTE: Incluir torch/extension.h ANTES de pybind11 para habilitar los type casters
#ifdef TORCH_FOUND
#include <torch/extension.h>
#include <torch/torch.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/sparse_map.h"
#include "../include/sparse_engine.h"
#include "../include/version.h"

namespace py = pybind11;
using namespace atheria;

// Función simple de prueba
int add(int i, int j) {
    return i + j;
}

// Módulo PyBind11
PYBIND11_MODULE(atheria_core, m) {
    m.doc() = "Atheria Core: Motor nativo de alto rendimiento para simulaciones cuánticas";
    
    // Exponer información de versión
    m.attr("__version__") = ATHERIA_NATIVE_VERSION_STRING;
    m.attr("VERSION_MAJOR") = ATHERIA_NATIVE_VERSION_MAJOR;
    m.attr("VERSION_MINOR") = ATHERIA_NATIVE_VERSION_MINOR;
    m.attr("VERSION_PATCH") = ATHERIA_NATIVE_VERSION_PATCH;
    m.def("get_version", []() {
        return std::string(ATHERIA_NATIVE_VERSION_STRING);
    }, "Retorna la versión del motor nativo en formato SemVer");
    
    // Función de prueba simple
    m.def("add", &add, "Suma dos enteros",
          py::arg("i"), py::arg("j"));
    
    // Verificar soporte de LibTorch
#ifdef TORCH_FOUND
    m.def("has_torch_support", []() { return true; }, 
          "Verifica si el soporte para tensores PyTorch está disponible");
#else
    m.def("has_torch_support", []() { return false; }, 
          "Verifica si el soporte para tensores PyTorch está disponible");
#endif
    
    // Binding para Coord3D
    py::class_<Coord3D>(m, "Coord3D")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t, int64_t>(), 
             "Constructor con coordenadas", 
             py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &Coord3D::x)
        .def_readwrite("y", &Coord3D::y)
        .def_readwrite("z", &Coord3D::z)
        .def("__repr__", [](const Coord3D& c) {
            return "Coord3D(" + std::to_string(c.x) + ", " + 
                   std::to_string(c.y) + ", " + std::to_string(c.z) + ")";
        });
    
    // Clase SparseMap
    auto sparse_map_class = py::class_<SparseMap>(m, "SparseMap")
        .def(py::init<>(), "Constructor por defecto")
        
        // Métodos básicos con valores numéricos (compatibilidad hacia atrás)
        .def("insert", &SparseMap::insert, 
             "Inserta o actualiza un valor numérico en el mapa",
             py::arg("key"), py::arg("value"))
        .def("contains", &SparseMap::contains,
             "Verifica si una clave existe en el mapa",
             py::arg("key"))
        .def("get", &SparseMap::get,
             "Obtiene un valor numérico del mapa con valor por defecto opcional",
             py::arg("key"), py::arg("default_value") = 0.0)
        .def("remove", &SparseMap::remove,
             "Elimina una clave del mapa",
             py::arg("key"))
        .def("clear", &SparseMap::clear,
             "Limpia todos los elementos del mapa")
        .def("size", &SparseMap::size,
             "Retorna el número de elementos en el mapa")
        .def("empty", &SparseMap::empty,
             "Verifica si el mapa está vacío")
        .def("keys", &SparseMap::keys,
             "Retorna una lista con todas las claves numéricas")
        .def("values", &SparseMap::values,
             "Retorna una lista con todos los valores numéricos")
        .def("__len__", &SparseMap::size)
        .def("__contains__", [](const SparseMap& self, int64_t key) {
            return self.contains(key);
        })
        .def("__setitem__", [](SparseMap& self, int64_t key, double value) {
            self.insert(key, value);
        })
        .def("__getitem__", [](const SparseMap& self, int64_t key) {
            if (!self.contains(key)) {
                throw py::key_error("Key not found");
            }
            return self.get(key);
        })
        .def("__delitem__", &SparseMap::remove)
        .def("__repr__", [](const SparseMap& self) {
            return "<SparseMap with " + std::to_string(self.size()) + " elements>";
        });
    
#ifdef TORCH_FOUND
    // Métodos para tensores si LibTorch está disponible
    sparse_map_class
        .def("insert_tensor", &SparseMap::insert_tensor,
             "Inserta o actualiza un tensor en el mapa usando coordenadas 3D",
             py::arg("coord"), py::arg("tensor"))
        .def("contains_coord", &SparseMap::contains_coord,
             "Verifica si existe un tensor en las coordenadas dadas",
             py::arg("coord"))
        .def("get_tensor", &SparseMap::get_tensor,
             "Obtiene un tensor del mapa usando coordenadas 3D",
             py::arg("coord"))
        .def("remove_coord", &SparseMap::remove_coord,
             "Elimina un tensor del mapa usando coordenadas 3D",
             py::arg("coord"))
        .def("coord_keys", &SparseMap::coord_keys,
             "Retorna una lista con todas las coordenadas 3D que tienen tensores");
    
    // Clase Engine (Motor de alto rendimiento nativo)
    py::class_<Engine>(m, "Engine")
        .def(py::init<int64_t, const std::string&, int64_t>(),
             "Constructor del motor de simulación",
             py::arg("d_state"), py::arg("device") = "cpu", py::arg("grid_size") = 64)
        .def("load_model", &Engine::load_model,
             "Carga un modelo TorchScript desde un archivo .pt",
             py::arg("model_path"))
        .def("add_particle", &Engine::add_particle,
             "Agrega una partícula en las coordenadas dadas",
             py::arg("coord"), py::arg("state"))
        .def("get_state_at", &Engine::get_state_at,
             "Obtiene el estado en una coordenada (materia o vacío)",
             py::arg("coord"))
        .def("step_native", &Engine::step_native,
             "Ejecuta un paso completo de simulación en C++ (todo el trabajo pesado)",
             "Retorna el número de partículas activas")
        .def("get_matter_count", &Engine::get_matter_count,
             "Retorna el número de partículas de materia")
        .def("get_step_count", &Engine::get_step_count,
             "Retorna el número de pasos ejecutados")
        .def("clear", &Engine::clear,
             "Limpia toda la materia del universo")
        .def("activate_neighborhood", &Engine::activate_neighborhood,
             "Activa el vecindario de una coordenada",
             py::arg("coord"), py::arg("radius") = 1)
        .def("get_active_coords", &Engine::get_active_coords,
             "Retorna una lista de todas las coordenadas activas")
        .def("get_last_error", &Engine::get_last_error,
             "Obtiene el último mensaje de error (si hubo un error al cargar el modelo)");
#endif
}
