#!/usr/bin/env python3
"""
Ejemplo práctico de uso del módulo atheria_core C++.

Este script muestra cómo usar las funcionalidades del núcleo C++
de alto rendimiento de Atheria 4.
"""

import atheria_core
import time

def ejemplo_funcion_add():
    """Ejemplo básico de la función add()"""
    print("=" * 60)
    print("Ejemplo 1: Función add()")
    print("=" * 60)
    
    resultado = atheria_core.add(10, 20)
    print(f"add(10, 20) = {resultado}")
    
    resultado2 = atheria_core.add(-5, 15)
    print(f"add(-5, 15) = {resultado2}")
    print()

def ejemplo_sparsemap_basico():
    """Ejemplo básico de uso de SparseMap"""
    print("=" * 60)
    print("Ejemplo 2: SparseMap - Uso Básico")
    print("=" * 60)
    
    # Crear un mapa disperso
    smap = atheria_core.SparseMap()
    
    # Insertar valores
    smap.insert(1, 10.5)
    smap.insert(2, 20.3)
    smap.insert(3, 30.7)
    
    print(f"Tamaño del mapa: {smap.size()}")
    print(f"¿Contiene la clave 2?: {smap.contains(2)}")
    print(f"Valor en clave 1: {smap.get(1)}")
    print(f"Valor en clave 999 (default): {smap.get(999, -1.0)}")
    print()

def ejemplo_sparsemap_operadores():
    """Ejemplo de uso de operadores Python en SparseMap"""
    print("=" * 60)
    print("Ejemplo 3: SparseMap - Operadores Python")
    print("=" * 60)
    
    smap = atheria_core.SparseMap()
    
    # Usar operadores [] para asignar
    smap[10] = 100.5
    smap[20] = 200.3
    smap[30] = 300.7
    
    # Usar operador in (__contains__)
    if 10 in smap:
        print(f"Clave 10 existe: {smap[10]}")
    
    # Usar len()
    print(f"Tamaño con len(): {len(smap)}")
    
    # Acceder con []
    print(f"smap[20] = {smap[20]}")
    print()

def ejemplo_sparsemap_rendimiento():
    """Ejemplo de rendimiento de SparseMap"""
    print("=" * 60)
    print("Ejemplo 4: SparseMap - Prueba de Rendimiento")
    print("=" * 60)
    
    smap = atheria_core.SparseMap()
    
    # Insertar muchos valores
    num_elementos = 10000
    print(f"Insertando {num_elementos} elementos...")
    
    start = time.time()
    for i in range(num_elementos):
        smap[i] = i * 1.5
    insert_time = time.time() - start
    
    print(f"Tiempo de inserción: {insert_time:.4f} segundos")
    print(f"Elementos insertados: {smap.size()}")
    
    # Acceso aleatorio
    print("\nProbando acceso aleatorio...")
    start = time.time()
    for _ in range(1000):
        key = (i * 7) % num_elementos
        _ = smap.get(key)
    access_time = time.time() - start
    
    print(f"Tiempo de 1000 accesos: {access_time:.4f} segundos")
    print(f"Promedio por acceso: {access_time/1000*1000:.4f} microsegundos")
    
    # Obtener todas las claves y valores
    print("\nObteniendo todas las claves y valores...")
    start = time.time()
    keys = smap.keys()
    values = smap.values()
    extract_time = time.time() - start
    
    print(f"Claves obtenidas: {len(keys)}")
    print(f"Valores obtenidos: {len(values)}")
    print(f"Tiempo de extracción: {extract_time:.4f} segundos")
    print()

def ejemplo_sparsemap_simulacion():
    """Ejemplo de uso en contexto de simulación"""
    print("=" * 60)
    print("Ejemplo 5: SparseMap - Simulación de Estado Esparcido")
    print("=" * 60)
    
    # Simular un estado disperso de partículas en un espacio 3D
    # Usando hash de coordenadas como clave
    def coord_to_key(x, y, z):
        """Convierte coordenadas 3D a una clave única"""
        return (x << 20) | (y << 10) | z
    
    estado = atheria_core.SparseMap()
    
    # Agregar algunas partículas dispersas
    particulas = [
        (10, 20, 30, 1.5),  # (x, y, z, energía)
        (50, 60, 70, 2.3),
        (100, 200, 300, 3.7),
        (15, 25, 35, 1.2),
        (55, 65, 75, 2.8),
    ]
    
    for x, y, z, energia in particulas:
        key = coord_to_key(x, y, z)
        estado[key] = energia
    
    print(f"Estado inicial: {len(estado)} partículas")
    
    # Consultar una partícula específica
    key = coord_to_key(50, 60, 70)
    if key in estado:
        print(f"Partícula en (50, 60, 70): energía = {estado[key]}")
    
    # Simular evolución: actualizar energía de una partícula
    key = coord_to_key(10, 20, 30)
    estado[key] = estado.get(key, 0) * 1.1  # Aumentar energía 10%
    print(f"Partícula en (10, 20, 30) después de evolución: {estado[key]}")
    
    # Limpiar partículas de baja energía
    keys_to_remove = []
    for key in estado.keys():
        if estado[key] < 1.5:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del estado[key]
    
    print(f"Estado después de limpieza: {len(estado)} partículas")
    print()

def main():
    """Función principal"""
    print("\n" + "=" * 60)
    print("🚀 Ejemplos de Uso de atheria_core (Núcleo C++)")
    print("=" * 60)
    print()
    
    try:
        ejemplo_funcion_add()
        ejemplo_sparsemap_basico()
        ejemplo_sparsemap_operadores()
        ejemplo_sparsemap_rendimiento()
        ejemplo_sparsemap_simulacion()
        
        print("=" * 60)
        print("✅ Todos los ejemplos ejecutados exitosamente!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

