// src/cpp_core/src/bindings.cpp
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
    
    // Clase SparseMap
    py::class_<SparseMap>(m, "SparseMap")
        .def(py::init<>(), "Constructor por defecto")
        .def("insert", &SparseMap::insert, 
             "Inserta o actualiza un valor en el mapa",
             py::arg("key"), py::arg("value"))
        .def("contains", &SparseMap::contains,
             "Verifica si una clave existe en el mapa",
             py::arg("key"))
        .def("get", &SparseMap::get,
             "Obtiene un valor del mapa con valor por defecto opcional",
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
             "Retorna una lista con todas las claves")
        .def("values", &SparseMap::values,
             "Retorna una lista con todos los valores")
        .def("__len__", &SparseMap::size)
        .def("__contains__", &SparseMap::contains)
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
}

