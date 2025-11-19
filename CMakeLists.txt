cmake_minimum_required(VERSION 3.15)
project(atheria_core)

# Configuración de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configuración de build tipo Release por defecto
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configuración de flags de compilación
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
else()
    add_compile_options(-g -Wall -Wextra)
endif()

# Buscar Python3 (necesario para PyBind11)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Buscar PyBind11
find_package(pybind11 REQUIRED)

# Incluir directorios
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp_core/include
    ${pybind11_INCLUDE_DIRS}
)

# Fuentes del módulo
set(CPP_SOURCES
    src/cpp_core/src/sparse_map.cpp
    src/cpp_core/src/bindings.cpp
)

# Crear el módulo de Python
pybind11_add_module(
    atheria_core
    ${CPP_SOURCES}
)

# Configuración específica de PyBind11
target_link_libraries(atheria_core PRIVATE pybind11::module)

# Configuración de instalación
set_target_properties(atheria_core PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""
    SUFFIX "${Python3_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}"
)

# Instalación del módulo (será manejado por setup.py)
# No instalamos aquí, dejamos que setuptools maneje la instalación

# Mensaje informativo
message(STATUS "Building atheria_core module")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python version: ${Python3_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

