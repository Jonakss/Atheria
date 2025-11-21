#!/usr/bin/env python3
"""
CLI simple para Atheria 4.
Facilita comandos comunes: build, install, dev, run, clean
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_project_root():
    """Obtiene el directorio raíz del proyecto."""
    # Este script está en src/, el proyecto está un nivel arriba
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    return project_root


def run_command(cmd, cwd=None, check=True):
    """Ejecuta un comando y muestra su output."""
    if cwd is None:
        cwd = get_project_root()
    
    print(f"📦 Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check)
    return result.returncode == 0


def build():
    """Compila las extensiones C++."""
    print("🔨 Compilando extensiones C++...")
    project_root = get_project_root()
    success = run_command([
        sys.executable, "setup.py", "build_ext", "--inplace"
    ], cwd=project_root)
    
    if success:
        print("✅ Compilación completada exitosamente")
    else:
        print("❌ Error en la compilación")
        sys.exit(1)


def install():
    """Instala el paquete en modo desarrollo."""
    print("📥 Instalando paquete en modo desarrollo...")
    project_root = get_project_root()
    success = run_command([
        sys.executable, "-m", "pip", "install", "-e", "."
    ], cwd=project_root)
    
    if success:
        print("✅ Instalación completada exitosamente")
    else:
        print("❌ Error en la instalación")
        sys.exit(1)


def run_server(no_frontend=True, port=None, host=None):
    """Ejecuta el servidor."""
    print("🚀 Iniciando servidor Atheria...")
    project_root = get_project_root()
    
    cmd = [sys.executable, "run_server.py"]
    
    if no_frontend:
        cmd.append("--no-frontend")
    
    if port:
        cmd.extend(["--port", str(port)])
    
    if host:
        cmd.extend(["--host", host])
    
    # Establecer variable de entorno según no_frontend
    # CRÍTICO: Limpiar la variable si queremos frontend, establecerla si no queremos
    env = os.environ.copy()
    if no_frontend:
        env['ATHERIA_NO_FRONTEND'] = '1'
    else:
        # Asegurar que la variable NO esté establecida si queremos frontend
        # Esto es importante porque puede estar establecida desde una ejecución anterior
        if 'ATHERIA_NO_FRONTEND' in env:
            del env['ATHERIA_NO_FRONTEND']
    
    print(f"🌐 Servidor iniciándose {'(sin frontend)' if no_frontend else '(con frontend)'}...")
    subprocess.run(cmd, cwd=project_root, env=env)


def run_frontend_dev(port=None):
    """Ejecuta el frontend en modo desarrollo (npm run dev)."""
    print("🎨 Iniciando frontend en modo desarrollo...")
    project_root = get_project_root()
    frontend_dir = project_root / "frontend"
    
    if not frontend_dir.exists():
        print("❌ Error: Directorio 'frontend' no encontrado")
        sys.exit(1)
    
    # Verificar si node_modules existe
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("⚠️  node_modules no encontrado. Ejecutando 'npm install' primero...")
        install_success = run_command(
            ["npm", "install"],
            cwd=frontend_dir,
            check=False
        )
        if not install_success:
            print("❌ Error instalando dependencias de npm")
            sys.exit(1)
        print()
    
    # Ejecutar npm run dev
    cmd = ["npm", "run", "dev"]
    
    # Si se especifica puerto, agregarlo como variable de entorno
    env = os.environ.copy()
    if port:
        env['PORT'] = str(port)
        print(f"🌐 Frontend iniciándose en puerto {port}...")
    else:
        print(f"🌐 Frontend iniciándose en puerto por defecto (5173)...")
    
    subprocess.run(cmd, cwd=frontend_dir, env=env)


def clean():
    """Limpia archivos de build y cache."""
    print("🧹 Limpiando archivos de build...")
    project_root = get_project_root()
    
    to_remove = [
        "build/",
        "dist/",
        "*.egg-info/",
        "**/__pycache__/",
        "**/*.pyc",
        "**/*.pyo",
        "*.so",
        "**/*.so",
        ".pytest_cache/",
        ".mypy_cache/",
    ]
    
    import shutil
    from glob import glob
    
    removed_count = 0
    project_path = Path(project_root)
    for pattern in to_remove:
        matches = list(project_path.glob(pattern))
        for path in matches:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"  🗑️  Removido directorio: {path.relative_to(project_path)}")
                else:
                    path.unlink()
                    print(f"  🗑️  Removido archivo: {path.relative_to(project_path)}")
                removed_count += 1
    
    if removed_count == 0:
        print("✨ Ya está limpio, no hay nada que limpiar")
    else:
        print(f"✅ Limpieza completada ({removed_count} items removidos)")


def dev(no_frontend=True, port=None, host=None):
    """Build + Install + Run (workflow completo de desarrollo)."""
    print("🔧 Modo desarrollo: Build + Install + Run")
    print("=" * 60)
    
    try:
        # Build
        build()
        print()
        
        # Install
        install()
        print()
        
        # Run
        run_server(no_frontend=no_frontend, port=port, host=host)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def main():
    """Función principal del CLI."""
    parser = argparse.ArgumentParser(
        description="CLI simple para Atheria 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  atheria dev                  # Build + Install + Run (sin frontend)
  atheria dev --frontend       # Build + Install + Run (con frontend)
  atheria frontend-dev         # Solo frontend en modo desarrollo (npm run dev)
  atheria frontend-dev --port 3000  # Frontend en puerto 3000
  atheria build                # Solo compilar
  atheria install              # Solo instalar
  atheria run                  # Solo ejecutar servidor
  atheria clean                # Limpiar builds
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: dev
    dev_parser = subparsers.add_parser('dev', help='Build + Install + Run (modo desarrollo)')
    dev_parser.add_argument('--frontend', action='store_true', 
                           help='Incluir frontend (por defecto: sin frontend)')
    dev_parser.add_argument('--port', type=int, default=None,
                           help='Puerto del servidor')
    dev_parser.add_argument('--host', type=str, default=None,
                           help='Host del servidor')
    
    # Comando: build
    subparsers.add_parser('build', help='Compilar extensiones C++')
    
    # Comando: install
    subparsers.add_parser('install', help='Instalar paquete en modo desarrollo')
    
    # Comando: run
    run_parser = subparsers.add_parser('run', help='Ejecutar servidor')
    run_parser.add_argument('--frontend', action='store_true',
                           help='Incluir frontend (por defecto: sin frontend)')
    run_parser.add_argument('--port', type=int, default=None,
                           help='Puerto del servidor')
    run_parser.add_argument('--host', type=str, default=None,
                           help='Host del servidor')
    
    # Comando: clean
    subparsers.add_parser('clean', help='Limpiar archivos de build')
    
    # Comando: frontend-dev
    frontend_parser = subparsers.add_parser('frontend-dev', 
                                           help='Ejecutar frontend en modo desarrollo (npm run dev)')
    frontend_parser.add_argument('--port', type=int, default=None,
                                help='Puerto del frontend (por defecto: 5173)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Ejecutar comando
    if args.command == 'dev':
        dev(no_frontend=not args.frontend, port=args.port, host=args.host)
    elif args.command == 'build':
        build()
    elif args.command == 'install':
        install()
    elif args.command == 'run':
        run_server(no_frontend=not args.frontend, port=args.port, host=args.host)
    elif args.command == 'clean':
        clean()
    elif args.command == 'frontend-dev':
        run_frontend_dev(port=args.port if hasattr(args, 'port') else None)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

