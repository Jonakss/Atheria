#!/bin/bash
# Script para ejecutar tests de Fase 2 con configuración de entorno correcta

set -e

echo "🚀 Configurando entorno para tests de Fase 2..."

# Configurar LD_LIBRARY_PATH para CUDA
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:$LD_LIBRARY_PATH

echo "✅ LD_LIBRARY_PATH configurado"
echo "   LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Ejecutar tests
echo ""
echo "🧪 Ejecutando tests..."
python3 scripts/test_cpp_binding.py

