// frontend/src/utils/dataDecompression.ts
/**
 * Utilidades para descomprimir datos recibidos del servidor.
 * El servidor envía arrays comprimidos en formato base64 + zlib.
 * 
 * Nota: Para descompresión zlib en el navegador, necesitamos usar pako.
 * Si no está disponible, los datos se envían sin comprimir como fallback.
 */

import pako from 'pako';

interface CompressedArray {
    data: string; // base64
    shape: number[];
    dtype: string;
    compressed: boolean;
}

export function decompressArray(compressed: CompressedArray): number[][] {
    /**
     * Descomprime un array comprimido desde el servidor.
     * 
     * @param compressed - Objeto con data (base64), shape, dtype
     * @returns Array 2D descomprimido
     */
    try {
        // Decodificar base64 a bytes
        const binaryString = atob(compressed.data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        
        // Descomprimir con pako
        let decompressed: Uint8Array;
        try {
            decompressed = pako.inflate(bytes);
        } catch (e) {
            // Si falla la descompresión, intentar usar datos directamente
            console.warn('Error descomprimiendo con pako, usando datos sin descomprimir:', e);
            decompressed = bytes;
        }
        
        // Convertir bytes a array según dtype
        let values: Float32Array | Float64Array;
        if (compressed.dtype.includes('float32')) {
            // Asegurar que el buffer tiene el tamaño correcto
            const byteLength = decompressed.length;
            const float32Length = Math.floor(byteLength / 4);
            const buffer = decompressed.buffer.slice(0, float32Length * 4);
            values = new Float32Array(buffer);
        } else if (compressed.dtype.includes('float64')) {
            const byteLength = decompressed.length;
            const float64Length = Math.floor(byteLength / 8);
            const buffer = decompressed.buffer.slice(0, float64Length * 8);
            values = new Float64Array(buffer);
        } else {
            // Default a float32
            const byteLength = decompressed.length;
            const float32Length = Math.floor(byteLength / 4);
            const buffer = decompressed.buffer.slice(0, float32Length * 4);
            values = new Float32Array(buffer);
        }
        
        // Reshape a 2D
        const [H, W] = compressed.shape;
        const result: number[][] = [];
        for (let y = 0; y < H; y++) {
            const row: number[] = [];
            for (let x = 0; x < W; x++) {
                const idx = y * W + x;
                if (idx < values.length) {
                    row.push(values[idx]);
                } else {
                    row.push(0);
                }
            }
            result.push(row);
        }
        
        return result;
    } catch (error) {
        console.error('Error descomprimiendo array:', error);
        // Retornar array vacío como fallback
        return [];
    }
}

export function isCompressed(data: any): data is CompressedArray {
    /**
     * Verifica si un dato es un array comprimido.
     */
    return (
        typeof data === 'object' &&
        data !== null &&
        'compressed' in data &&
        data.compressed === true &&
        'data' in data &&
        'shape' in data &&
        'dtype' in data
    );
}

export function decompressIfNeeded(data: any): any {
    /**
     * Descomprime un dato si está comprimido, sino lo retorna tal cual.
     */
    if (isCompressed(data)) {
        return decompressArray(data);
    }
    return data;
}

