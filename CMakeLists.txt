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

# Buscar PyBind11 usando el método oficial
# PyBind11 se instala como paquete Python, así que obtenemos su ubicación desde Python
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE pybind11_result
)

if(NOT pybind11_result AND pybind11_CMAKE_DIR)
    # Agregar el directorio CMake de PyBind11 al path
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_CMAKE_DIR}")
    message(STATUS "Found PyBind11 CMake directory: ${pybind11_CMAKE_DIR}")
else()
    message(FATAL_ERROR 
        "PyBind11 not found. Please install it first:\n"
        "  pip install pybind11>=2.10.0"
    )
endif()

# Buscar PyBind11 con CMake (ahora debería encontrarlo)
find_package(pybind11 REQUIRED)

# Incluir directorios
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp_core/include
)

# Obtener directorio de headers de PyBind11 si no está definido
if(NOT pybind11_INCLUDE_DIRS)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE pybind11_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_INCLUDE_DIRS)
        include_directories(${pybind11_INCLUDE_DIRS})
    endif()
endif()

# Fuentes del módulo
set(CPP_SOURCES
    src/cpp_core/src/sparse_map.cpp
    src/cpp_core/src/bindings.cpp
)

# Crear el módulo de Python usando pybind11_add_module
pybind11_add_module(
    atheria_core
    ${CPP_SOURCES}
)

# Configuración específica de PyBind11
target_link_libraries(atheria_core PRIVATE pybind11::module)

# Configuración de instalación
# PyBind11 maneja el nombre del módulo, pero necesitamos asegurarnos de que tenga el formato correcto
set_target_properties(atheria_core PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX ""
    # PyBind11 añadirá automáticamente el sufijo correcto, pero podemos especificarlo explícitamente
    OUTPUT_NAME "atheria_core"
)

# Instalación del módulo (será manejado por setup.py)
# No instalamos aquí, dejamos que setuptools maneje la instalación

# Mensaje informativo
message(STATUS "Building atheria_core module")
message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python version: ${Python3_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PyBind11 CMake dir: ${pybind11_CMAKE_DIR}")
