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

namespace py = pybind11;
using namespace atheria;

// Función simple de prueba
int add(int i, int j) {
    return i + j;
}

// Módulo PyBind11
PYBIND11_MODULE(atheria_core, m) {
    m.doc() = "Atheria Core: Motor nativo de alto rendimiento para simulaciones cuánticas";
    
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
#endif
}
