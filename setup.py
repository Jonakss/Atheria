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
        
        # Obtener el directorio donde debe estar el módulo compilado
        # ext_dir será donde setuptools espera encontrar el módulo
        ext_fullpath = Path(self.get_ext_fullpath("atheria_core")).resolve()
        ext_dir = ext_fullpath.parent
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
        """Verifica que el módulo compilado esté en la ubicación correcta."""
        import shutil
        
        # MODIFICACIÓN: Colocar el módulo en src/ para mejor organización
        project_root = Path(__file__).parent.resolve()
        target_dir = project_root / "src"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # El módulo ya debería estar en ext_dir porque CMake lo configuramos así
        # Buscar el módulo compilado en ext_dir (donde CMake lo colocó)
        module_exts = [".so", ".pyd", ".dylib"]
        module_name = None
        src = None
        
        # Buscar en ext_dir (donde CMake debería haberlo colocado)
        for ext in module_exts:
            for file in ext_dir.glob(f"*atheria_core*{ext}"):
                module_name = file.name
                src = file
                break
            if module_name:
                break
        
        # Si no se encuentra en ext_dir, buscar en build_temp
        if not src or not src.exists():
            for ext in module_exts:
                for file in build_temp.glob(f"*atheria_core*{ext}"):
                    module_name = file.name
                    src = file
                    break
                if module_name:
                    break
        
        # Si se encuentra, mover/copiar a src/
        if module_name and src and src.exists():
            target_path = target_dir / module_name
            if src != target_path:
                # Mover o copiar a src/
                if target_path.exists():
                    target_path.unlink()  # Eliminar si existe
                shutil.copy2(str(src), str(target_path))
                print(f"✅ Módulo copiado a src/: {target_path}")
        
        # Si se encuentra, crear symlink con nombre que setuptools espera (si es necesario)
        if module_name and src and src.exists():
            # PyBind11 genera el nombre como: atheria_corecpython-...
            # Pero setuptools espera: atheria_core.cpython-...
            # Crear un symlink o copia con el nombre correcto si es necesario
            expected_name = module_name.replace("atheria_corec", "atheria_core.c", 1) if "atheria_corec" in module_name else module_name
            
            # Si el nombre es diferente, crear un symlink o copia
            if expected_name != module_name:
                expected_path = ext_dir / expected_name
                if not expected_path.exists():
                    try:
                        # Intentar crear symlink primero (más eficiente)
                        if not platform.system() == "Windows":
                            os.symlink(src.name, str(expected_path))
                            print(f"✅ Created symlink: {expected_path} -> {src.name}")
                        else:
                            # En Windows, copiar el archivo
                            shutil.copy2(str(src), str(expected_path))
                            print(f"✅ Copied module with correct name: {expected_path}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not create symlink/copy: {e}")
                        print(f"   Source: {src}")
                        print(f"   Expected: {expected_path}")
            
            print(f"✅ Module found: {src}")
        else:
            print("⚠️  Warning: Could not find compiled module")
            print(f"   Searched in: {build_temp}")
            print(f"   Also searched in: {ext_dir}")
            print(f"   Pattern: *atheria_core*")


# Leer versión desde src/__version__.py
def get_version():
    version_path = Path(__file__).parent / "src" / "__version__.py"
    if version_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("version", version_path)
        version_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(version_module)
        return version_module.__version__
    return "0.0.0"

# Configuración del setup
setup(
    name="atheria",
    version=get_version(),
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
        "redis>=4.0.0",
        "zstandard>=0.19.0",
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
    entry_points={
        'console_scripts': [
            'atheria=src.cli:main',
            'ath=src.cli:main',  # Alias corto
        ],
    },
)

