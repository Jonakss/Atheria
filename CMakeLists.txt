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

# Definir TORCH_FOUND si LibTorch está disponible (se definirá más adelante)

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

# Buscar LibTorch
# LibTorch viene con PyTorch instalado, así que obtenemos su ubicación desde Python
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE torch_result
)

if(NOT torch_result AND TORCH_PYTHON_DIR)
    # LibTorch instala archivos CMake en share/cmake/Torch
    set(TORCH_CMAKE_DIR "${TORCH_PYTHON_DIR}/share/cmake/Torch")
    get_filename_component(TORCH_CMAKE_DIR "${TORCH_CMAKE_DIR}" ABSOLUTE)
    
    if(EXISTS "${TORCH_CMAKE_DIR}/TorchConfig.cmake")
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_DIR}")
        message(STATUS "Found LibTorch CMake directory: ${TORCH_CMAKE_DIR}")
    else()
        # Intentar configuración manual
        set(TORCH_INSTALL_PREFIX "${TORCH_PYTHON_DIR}")
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_INSTALL_PREFIX}")
        message(STATUS "Setting TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
    endif()
endif()

# Buscar Torch
find_package(Torch QUIET)

if(NOT Torch_FOUND)
    # Si no se encuentra, intentar configuración manual
    if(TORCH_PYTHON_DIR)
        set(TORCH_INCLUDE_DIRS "${TORCH_PYTHON_DIR}/include")
        set(TORCH_LIBRARIES "${TORCH_PYTHON_DIR}/lib/libtorch.so;${TORCH_PYTHON_DIR}/lib/libtorch_cpu.so;${TORCH_PYTHON_DIR}/lib/libc10.so")
        
        if(EXISTS "${TORCH_INCLUDE_DIRS}/torch/torch.h")
            message(STATUS "Found LibTorch headers: ${TORCH_INCLUDE_DIRS}")
            message(STATUS "Found LibTorch libraries: ${TORCH_LIBRARIES}")
            set(Torch_FOUND TRUE)
        else()
            message(WARNING "LibTorch headers not found at ${TORCH_INCLUDE_DIRS}")
            message(FATAL_ERROR "LibTorch not found. Please ensure PyTorch is installed.")
        endif()
    else()
        message(FATAL_ERROR "LibTorch not found. Please install PyTorch first:\n  pip install torch")
    endif()
endif()

# Incluir directorios
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp_core/include
)

if(Torch_FOUND)
    if(TARGET torch)
        message(STATUS "Using LibTorch target: torch")
        add_definitions(-DTORCH_FOUND)
    else()
        message(STATUS "LibTorch found but target not available, using manual configuration")
        add_definitions(-DTORCH_FOUND)
    endif()
    set(TORCH_FOUND TRUE)
else()
    set(TORCH_FOUND FALSE)
endif()

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
    src/cpp_core/src/harmonic_vacuum.cpp
    src/cpp_core/src/sparse_engine.cpp
    src/cpp_core/src/bindings.cpp
)

# Crear el módulo de Python usando pybind11_add_module
pybind11_add_module(
    atheria_core
    ${CPP_SOURCES}
)

# Configuración específica de PyBind11 y LibTorch
target_link_libraries(atheria_core PRIVATE pybind11::module)

# Enlazar LibTorch si está disponible
if(TORCH_FOUND)
    if(TARGET torch)
        # Si encontramos el target torch, también necesitamos torch_python para los bindings
        target_link_directories(atheria_core PRIVATE ${TORCH_PYTHON_DIR}/lib)
        target_link_libraries(atheria_core PRIVATE torch torch_python)
        target_compile_definitions(atheria_core PRIVATE TORCH_FOUND)
    else()
        # Configuración manual - obtener rutas desde Python
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))"
            OUTPUT_VARIABLE TORCH_INCLUDE
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
            OUTPUT_VARIABLE TORCH_LIB_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        
        if(TORCH_INCLUDE)
            target_include_directories(atheria_core PRIVATE ${TORCH_INCLUDE})
            message(STATUS "LibTorch include: ${TORCH_INCLUDE}")
        endif()
        if(TORCH_LIB_DIR)
            target_link_directories(atheria_core PRIVATE ${TORCH_LIB_DIR})
            # Enlazar bibliotecas necesarias de PyTorch
            # torch_python contiene los bindings de PyBind11 para tensores
            target_link_libraries(atheria_core PRIVATE 
                torch_python
                torch 
                torch_cpu 
                c10
            )
            message(STATUS "LibTorch lib dir: ${TORCH_LIB_DIR}")
        endif()
        target_compile_definitions(atheria_core PRIVATE TORCH_FOUND)
    endif()
    message(STATUS "LibTorch linked to atheria_core")
else()
    message(WARNING "LibTorch not found, tensor support will be limited")
endif()

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
