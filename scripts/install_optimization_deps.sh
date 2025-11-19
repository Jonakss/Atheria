#!/bin/bash
# Script para instalar dependencias de optimización de transferencia de datos

echo "📦 Instalando dependencias para optimización de transferencia de datos..."

# Activar entorno virtual si existe
if [ -d "ath_venv" ]; then
    echo "Activando entorno virtual..."
    source ath_venv/bin/activate
fi

# Instalar LZ4 (compresión rápida)
echo "Instalando lz4..."
pip install lz4

# Instalar CBOR2 (formato binario eficiente)
echo "Instalando cbor2..."
pip install cbor2

echo "✅ Dependencias instaladas:"
echo "   - lz4: Compresión rápida (2-3x más rápido que zlib)"
echo "   - cbor2: Formato binario eficiente (mejor que JSON)"
echo ""
echo "Para usar las optimizaciones, importa data_transfer_optimized en lugar de data_compression"

