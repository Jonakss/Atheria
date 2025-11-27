"""
Utilidad para detectar y descargar binarios pre-compilados del motor nativo atheria_core
desde GitHub Releases.

Uso en notebooks (Kaggle/Colab):
    from src.utils.binary_loader import try_load_native_engine
    
    if not try_load_native_engine(fallback_to_download=True):
        print("Motor nativo no disponible, usando motor Python")
"""

import os
import sys
import platform
import importlib.util
from pathlib import Path
from typing import Optional, Dict
import urllib.request
import json


def get_platform_info() -> Dict[str, str]:
    """
    Detecta información de la plataforma actual.
    
    Returns:
        dict con keys: 'os', 'arch', 'python_version', 'ext'
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    
    # Normalizar nombres de OS
    if system == "darwin":
        os_name = "macos"
    elif system == "linux":
        os_name = "linux"
    elif system == "windows":
        os_name = "windows"
    else:
        os_name = system
    
    # Normalizar arquitectura
    if machine in ["x86_64", "amd64"]:
        arch = "x86_64" if os_name != "windows" else "x64"
    elif machine in ["arm64", "aarch64"]:
        arch = "arm64"
    else:
        arch = machine
    
    # Extensión del binario
    if os_name == "windows":
        ext = "pyd"
    else:
        ext = "so"
    
    return {
        "os": os_name,
        "arch": arch,
        "python_version": py_version,
        "ext": ext,
        "platform": f"{os_name}-{arch}"
    }


def get_atheria_version() -> str:
    """
    Obtiene la versión actual de Atheria desde src/__version__.py
    
    Returns:
        Version string (ej: "4.2.6")
    """
    try:
        version_file = Path(__file__).parent.parent / "__version__.py"
        spec = importlib.util.spec_from_file_location("version", version_file)
        if spec and spec.loader:
            version_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(version_module)
            return version_module.__version__
    except Exception:
        pass
    
    # Fallback: intentar importar directamente
    try:
        from src import __version__
        return __version__.__version__
    except Exception:
        pass
    
    return "4.2.6"  # Fallback hardcoded


def construct_download_url(version: str, platform_info: Dict[str, str]) -> str:
    """
    Construye la URL de descarga del binario desde GitHub Releases.
    
    Args:
        version: Version tag (ej: "4.2.6")
        platform_info: Dict de get_platform_info()
    
    Returns:
        URL completa de descarga
    """
    repo_owner = "Jonakss"
    repo_name = "Atheria"
    
    # Nombre del archivo: atheria_core-{platform}-py{version}.{ext}
    filename = f"atheria_core-{platform_info['platform']}-py{platform_info['python_version']}.{platform_info['ext']}"
    
    # URL del release
    url = f"https://github.com/{repo_owner}/{repo_name}/releases/download/v{version}/{filename}"
    
    return url


def download_prebuilt_binary(
    version: Optional[str] = None,
    target_dir: Optional[Path] = None,
    verbose: bool = True
) -> Optional[Path]:
    """
    Descarga el binario pre-compilado desde GitHub Releases.
    
    Args:
        version: Version tag (None = auto-detect)
        target_dir: Directorio destino (None = src/)
        verbose: Mostrar mensajes de progreso
    
    Returns:
        Path al binario descargado, o None si falla
    """
    # Auto-detectar versión si no se provee
    if version is None:
        version = get_atheria_version()
    
    # Detectar plataforma
    platform_info = get_platform_info()
    
    if verbose:
        print(f"🔍 Plataforma detectada: {platform_info['platform']} (Python {platform_info['python_version']})")
    
    # Construir URL
    download_url = construct_download_url(version, platform_info)
    
    if verbose:
        print(f"📥 Descargando desde: {download_url}")
    
    # Determinar directorio destino
    if target_dir is None:
        # Intentar encontrar src/ relativo a este archivo
        src_dir = Path(__file__).parent.parent
        if not src_dir.name == "src":
            # Fallback: buscar src/ en el proyecto
            project_root = Path.cwd()
            src_dir = project_root / "src"
            if not src_dir.exists():
                # Último fallback: usar directorio actual
                src_dir = Path.cwd()
    else:
        src_dir = Path(target_dir)
    
    src_dir.mkdir(parents=True, exist_ok=True)
    
    # Nombre esperado del módulo (formato CPython)
    # Ejemplo: atheria_core.cpython-311-x86_64-linux-gnu.so
    python_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    
    if platform_info['os'] == 'linux':
        module_name = f"atheria_core.{python_tag}-{platform_info['arch']}-linux-gnu.{platform_info['ext']}"
    elif platform_info['os'] == 'macos':
        module_name = f"atheria_core.{python_tag}-darwin.{platform_info['ext']}"
    elif platform_info['os'] == 'windows':
        module_name = f"atheria_core.{python_tag}-win_amd64.{platform_info['ext']}"
    else:
        # Fallback genérico
        module_name = f"atheria_core.{platform_info['ext']}"
    
    target_path = src_dir / module_name
    
    try:
        # Descargar
        urllib.request.urlretrieve(download_url, target_path)
        
        if verbose:
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"✅ Descargado: {module_name} ({size_mb:.2f} MB)")
        
        return target_path
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            if verbose:
                print(f"⚠️  Binary no encontrado en GitHub Releases (404)")
                print(f"   URL: {download_url}")
                print(f"   Esto puede suceder si:")
                print(f"   - La versión v{version} no tiene releases publicados aún")
                print(f"   - Tu plataforma ({platform_info['platform']}) no está soportada")
        else:
            if verbose:
                print(f"❌ Error descargando: HTTP {e.code}")
        return None
        
    except Exception as e:
        if verbose:
            print(f"❌ Error descargando binario: {e}")
        return None


def try_load_native_engine(fallback_to_download: bool = True, verbose: bool = True) -> bool:
    """
    Intenta importar atheria_core. Si falla y fallback_to_download=True,
    intenta descargar automáticamente desde GitHub Releases.
    
    Args:
        fallback_to_download: Si True, descarga automáticamente si no está disponible
        verbose: Mostrar mensajes de progreso
    
    Returns:
        True si el motor nativo está disponible (importado exitosamente)
    """
    # Intentar importar primero
    try:
        import atheria_core
        if verbose:
            print("✅ Motor nativo atheria_core disponible")
        return True
    except ImportError:
        pass
    
    # Si no está disponible y queremos fallback
    if not fallback_to_download:
        if verbose:
            print("⚠️  Motor nativo no disponible")
        return False
    
    # Intentar descargar
    if verbose:
        print("🔧 Motor nativo no encontrado. Intentando descargar desde GitHub Releases...")
    
    downloaded = download_prebuilt_binary(verbose=verbose)
    
    if downloaded is None:
        if verbose:
            print("⚠️  No se pudo descargar el motor nativo")
            print("   El entrenamiento usará el motor Python (más lento pero funcional)")
        return False
    
    # Intentar importar de nuevo
    try:
        # Agregar src/ al path si no está
        src_dir = downloaded.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        
        import atheria_core
        if verbose:
            print("✅ Motor nativo descargado e importado exitosamente")
        return True
        
    except ImportError as e:
        if verbose:
            print(f"⚠️  Motor nativo descargado pero no se pudo importar: {e}")
            print("   Esto puede deberse a incompatibilidad de versión de Python o dependencias")
        return False


if __name__ == "__main__":
    # Test/demo
    print("=" * 60)
    print("Atheria Binary Loader - Test")
    print("=" * 60)
    
    info = get_platform_info()
    print(f"\n📊 Platform Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    version = get_atheria_version()
    print(f"\n📌 Atheria Version: {version}")
    
    url = construct_download_url(version, info)
    print(f"\n🔗 Download URL:")
    print(f"  {url}")
    
    print(f"\n🔧 Attempting to load native engine...")
    success = try_load_native_engine(fallback_to_download=True, verbose=True)
    
    if success:
        print("\n✅ SUCCESS: Native engine is available!")
    else:
        print("\n⚠️  Native engine not available (will use Python engine)")
