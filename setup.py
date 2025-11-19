#!/usr/bin/env python3
"""
Setup script para Atheria 4 con soporte para extensiones C++ usando PyBind11.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    """
    Clase personalizada para compilar extensiones C++ usando CMake y PyBind11.
    """
    
    def build_extensions(self):
        # Asegurarse de que CMake esté disponible
        cmake = self._find_cmake()
        
        # Directorios
        build_temp = Path(self.build_temp).resolve()
        build_temp.mkdir(parents=True, exist_ok=True)
        
        ext_dir = Path(self.get_ext_fullpath("atheria_core")).resolve().parent
        ext_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
        ]
        
        # Configuración específica de plataforma
        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(
                    "DEBUG" if self.debug else "RELEASE", ext_dir
                )
            ]
        
        # Configuración de build
        build_args = ["--config", "Release" if not self.debug else "Debug"]
        
        if platform.system() == "Windows":
            build_args += ["--", "/m"]
        else:
            build_args += ["--", "-j4"]  # Usar 4 cores por defecto
        
        # Ejecutar CMake
        print(f"Running CMake in {build_temp}...")
        subprocess.check_call(
            ["cmake", str(Path(__file__).parent)] + cmake_args,
            cwd=build_temp,
        )
        
        # Ejecutar build
        print("Building extension...")
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
        )
        
        # Mover el módulo compilado a la ubicación correcta
        self._move_built_module(build_temp, ext_dir)
    
    def _find_cmake(self):
        """Encuentra el ejecutable de CMake."""
        for cmake in ["cmake3", "cmake"]:
            try:
                subprocess.check_call([cmake, "--version"], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL)
                return cmake
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        raise RuntimeError(
            "CMake no está instalado. "
            "Por favor instala CMake (>= 3.15): "
            "https://cmake.org/download/"
        )
    
    def _move_built_module(self, build_temp, ext_dir):
        """Mueve el módulo compilado a la ubicación correcta."""
        import shutil
        
        # Buscar el módulo compilado en build_temp
        module_exts = [".so", ".pyd", ".dylib"]
        module_name = None
        src = None
        
        for ext in module_exts:
            for file in build_temp.glob(f"*atheria_core*{ext}"):
                module_name = file.name
                src = file
                break
            if module_name:
                break
        
        if module_name and src and src.exists():
            # ext_dir es donde setuptools espera encontrar el módulo
            # El módulo debe estar directamente en ext_dir con su nombre completo
            dst = ext_dir / module_name
            
            # Crear directorio de destino si no existe
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Copiar el módulo
            shutil.copy2(str(src), str(dst))
            print(f"✅ Module installed to {dst}")
            
            # También crear un symlink con nombre simple si es posible (opcional)
            if not platform.system() == "Windows":
                simple_name = f"atheria_core{module_exts[0]}"
                simple_dst = ext_dir / simple_name
                if not simple_dst.exists():
                    try:
                        os.symlink(module_name, str(simple_dst))
                    except:
                        pass  # Ignorar si no se puede crear symlink
        else:
            print("⚠️  Warning: Could not find compiled module")
            print(f"   Searched in: {build_temp}")
            print(f"   Pattern: *atheria_core*")


# Configuración del setup
setup(
    name="atheria",
    version="4.0.0",
    description="Atheria 4: Simulador de universo infinito con núcleo C++ de alto rendimiento",
    long_description=open("README.md").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Atheria Team",
    python_requires=">=3.8",
    packages=[
        "src",
        "src.engines",
        "src.models",
        "src.physics",
        "src.physics.analysis",
        "src.trainers",
    ],
    package_dir={"": "."},
    ext_modules=[Extension("atheria_core", [])],  # Placeholder, CMake lo maneja
    cmdclass={"build_ext": CMakeBuildExt},
    zip_safe=False,
    install_requires=[
        "torch",
        "numpy",
        "pybind11>=2.10.0",
    ],
    setup_requires=[
        "pybind11>=2.10.0",
        "setuptools>=45",
        "wheel",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
    ],
)

