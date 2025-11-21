#!/usr/bin/env python3
"""
Script helper para bump de versiones manual.
Útil para desarrollo local o cuando GitHub Actions no está disponible.

Uso:
    python scripts/bump_version.py --type patch
    python scripts/bump_version.py --type minor
    python scripts/bump_version.py --type major
    python scripts/bump_version.py --version 4.2.0  # Versión específica
"""

import argparse
import re
import sys
from pathlib import Path

def get_current_version():
    """Lee la versión actual desde src/__version__.py (fuente de verdad)."""
    version_file = Path("src/__version__.py")
    if not version_file.exists():
        print(f"❌ Error: {version_file} no existe")
        sys.exit(1)
    
    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        print(f"❌ Error: No se pudo leer versión de {version_file}")
        sys.exit(1)
    
    return match.group(1)

def parse_version(version_str):
    """Parsea versión string a tuple (major, minor, patch)."""
    parts = version_str.split('.')
    if len(parts) != 3:
        raise ValueError(f"Versión inválida: {version_str} (debe ser MAJOR.MINOR.PATCH)")
    
    return tuple(int(p) for p in parts)

def bump_version(current_version, bump_type):
    """Calcula nueva versión según bump type."""
    major, minor, patch = parse_version(current_version)
    
    if bump_type == 'major':
        return f"{major + 1}.0.0", (major + 1, 0, 0)
    elif bump_type == 'minor':
        return f"{major}.{minor + 1}.0", (major, minor + 1, 0)
    elif bump_type == 'patch':
        return f"{major}.{minor}.{patch + 1}", (major, minor, patch + 1)
    else:
        raise ValueError(f"Tipo de bump inválido: {bump_type} (debe ser major/minor/patch)")

def update_file_version(file_path, pattern, replacement):
    """Actualiza versión en un archivo usando regex."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"  ⚠️  {file_path} no existe, saltando...")
        return False
    
    content = file_path.read_text()
    new_content = re.sub(pattern, replacement, content)
    
    if content != new_content:
        file_path.write_text(new_content)
        return True
    return False

def update_all_versions(new_version, new_major, new_minor, new_patch):
    """Actualiza versiones en todos los archivos del proyecto."""
    print(f"📝 Actualizando versiones a {new_version}...")
    
    updates = []
    
    # 1. src/__version__.py
    if update_file_version(
        "src/__version__.py",
        r'__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{new_version}"'
    ):
        updates.append("src/__version__.py")
    
    if update_file_version(
        "src/__version__.py",
        r'__version_info__\s*=\s*\([^)]+\)',
        f'__version_info__ = ({new_major}, {new_minor}, {new_patch})'
    ):
        pass  # Ya actualizado arriba
    
    # 2. src/engines/__version__.py
    if update_file_version(
        "src/engines/__version__.py",
        r'ENGINE_VERSION\s*=\s*["\'][^"\']+["\']',
        f'ENGINE_VERSION = "{new_version}"'
    ):
        updates.append("src/engines/__version__.py")
    
    # 3. src/cpp_core/include/version.h
    cpp_updated = False
    cpp_file = Path("src/cpp_core/include/version.h")
    if cpp_file.exists():
        content = cpp_file.read_text()
        new_content = content
        
        new_content = re.sub(
            r'#define ATHERIA_NATIVE_VERSION_MAJOR \d+',
            f'#define ATHERIA_NATIVE_VERSION_MAJOR {new_major}',
            new_content
        )
        new_content = re.sub(
            r'#define ATHERIA_NATIVE_VERSION_MINOR \d+',
            f'#define ATHERIA_NATIVE_VERSION_MINOR {new_minor}',
            new_content
        )
        new_content = re.sub(
            r'#define ATHERIA_NATIVE_VERSION_PATCH \d+',
            f'#define ATHERIA_NATIVE_VERSION_PATCH {new_patch}',
            new_content
        )
        new_content = re.sub(
            r'#define ATHERIA_NATIVE_VERSION_STRING "[^"]+"',
            f'#define ATHERIA_NATIVE_VERSION_STRING "{new_version}"',
            new_content
        )
        
        if content != new_content:
            cpp_file.write_text(new_content)
            updates.append("src/cpp_core/include/version.h")
            cpp_updated = True
    
    # 4. frontend/package.json
    package_json = Path("frontend/package.json")
    if package_json.exists():
        import json
        data = json.loads(package_json.read_text())
        if data.get('version') != new_version:
            data['version'] = new_version
            package_json.write_text(json.dumps(data, indent=2) + '\n')
            updates.append("frontend/package.json")
    
    if updates:
        print(f"✅ Archivos actualizados: {', '.join(updates)}")
        return True
    else:
        print("⚠️  No se actualizaron archivos")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Bump versiones en todos los componentes de Atheria 4"
    )
    parser.add_argument(
        '--type',
        choices=['major', 'minor', 'patch'],
        help='Tipo de bump: major, minor, o patch'
    )
    parser.add_argument(
        '--version',
        type=str,
        help='Versión específica (ej: 4.2.0). Ignora --type si se especifica.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Solo mostrar cambios sin aplicar'
    )
    
    args = parser.parse_args()
    
    if not args.type and not args.version:
        parser.error("Debe especificar --type o --version")
    
    current_version = get_current_version()
    print(f"📌 Versión actual: {current_version}")
    
    if args.version:
        # Usar versión específica
        try:
            major, minor, patch = parse_version(args.version)
            new_version = args.version
            new_tuple = (major, minor, patch)
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Calcular nueva versión según bump type
        new_version, new_tuple = bump_version(current_version, args.type)
        print(f"🚀 Nueva versión: {new_version} (bump: {args.type})")
    
    if args.dry_run:
        print("🔍 DRY RUN: Mostrando cambios que se aplicarían...")
        print(f"  - Versión actual: {current_version}")
        print(f"  - Versión nueva: {new_version}")
        print("  - Archivos a actualizar:")
        print("    - src/__version__.py")
        print("    - src/engines/__version__.py")
        print("    - src/cpp_core/include/version.h")
        print("    - frontend/package.json")
        sys.exit(0)
    
    # Actualizar archivos
    new_major, new_minor, new_patch = new_tuple
    if update_all_versions(new_version, new_major, new_minor, new_patch):
        print(f"\n✅ Versiones actualizadas exitosamente a {new_version}")
        print(f"\n📝 Próximos pasos:")
        print(f"  1. Revisar cambios: git diff")
        print(f"  2. Crear commit: git commit -am 'chore: bump version to {new_version}'")
        print(f"  3. Crear tag: git tag -a v{new_version} -m 'Release version {new_version}'")
        print(f"  4. Push: git push && git push --tags")
    else:
        print("❌ Error actualizando versiones")
        sys.exit(1)

if __name__ == "__main__":
    main()

