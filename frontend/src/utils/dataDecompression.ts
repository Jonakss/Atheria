// frontend/src/utils/dataDecompression.ts
/**
 * Utilidades para descomprimir datos recibidos del servidor.
 *
 * Soporta múltiples formatos:
 * 1. Formato antiguo: base64 + zlib (con pako)
 * 2. Formato nuevo binario: CBOR + LZ4 + quantización uint8
 *
 * El frontend detecta automáticamente el formato y descomprime apropiadamente.
 */

import { decode as cborDecode } from "@msgpack/msgpack"; // msgpack también soporta CBOR
import pako from "pako";

// Intentar importar LZ4 (si está disponible)
const lz4: any = null;
// LZ4 support removed to avoid unused dependency

interface CompressedArray {
  data: string; // base64 (formato antiguo)
  shape: number[];
  dtype: string;
  compressed: boolean;
}

interface BinaryCompressedArray {
  data: Uint8Array | string; // bytes raw o base64
  shape: number[];
  dtype: string;
  compressed: boolean;
  quantized?: boolean;
  min_val?: number;
  max_val?: number;
  format?: string; // 'binary' para formato nuevo
  is_differential?: boolean;
}

/**
 * Descuantiza un array uint8 a float32.
 */
function dequantizeFromUint8(
  data: Uint8Array,
  shape: number[],
  minVal: number,
  maxVal: number
): Float32Array {
  // Normalizar de [0, 255] a [0, 1]
  const normalized = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    normalized[i] = data[i] / 255.0;
  }

  // Desnormalizar de [0, 1] a [minVal, maxVal]
  const denormalized = new Float32Array(data.length);
  if (maxVal !== minVal) {
    for (let i = 0; i < data.length; i++) {
      denormalized[i] = normalized[i] * (maxVal - minVal) + minVal;
    }
  } else {
    for (let i = 0; i < data.length; i++) {
      denormalized[i] = minVal;
    }
  }

  return denormalized;
}

/**
 * Descomprime bytes con LZ4 o zlib (fallback).
 */
function decompressBytes(compressed: Uint8Array): Uint8Array {
  // Intentar LZ4 primero
  if (lz4) {
    try {
      return new Uint8Array(lz4.decompress(compressed));
    } catch (e) {
      // Fallback a pako (zlib)
    }
  }

  // Usar pako (zlib) como fallback
  try {
    return pako.inflate(compressed);
  } catch (e) {
    console.warn(
      "Error descomprimiendo, retornando datos sin descomprimir:",
      e
    );
    return compressed;
  }
}

/**
 * Decodifica un array binario (formato nuevo).
 */
function decodeArrayBinary(
  encoded: BinaryCompressedArray
): Float32Array | Float64Array {
  let dataBytes: Uint8Array;

  // Convertir data a Uint8Array si es necesario
  if (typeof encoded.data === "string") {
    // Base64 string (puede venir de JSON fallback)
    const binaryString = atob(encoded.data);
    dataBytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      dataBytes[i] = binaryString.charCodeAt(i);
    }
  } else {
    dataBytes = encoded.data;
  }

  // Descomprimir si es necesario
  if (encoded.compressed) {
    dataBytes = decompressBytes(dataBytes);
  }

  // Descuantizar si es necesario
  if (
    encoded.quantized &&
    encoded.min_val !== undefined &&
    encoded.max_val !== undefined
  ) {
    const shape = encoded.shape;
    const totalSize = shape.reduce((a, b) => a * b, 1);
    const quantized = dataBytes.slice(0, totalSize);
    const denormalized = dequantizeFromUint8(
      quantized,
      shape,
      encoded.min_val,
      encoded.max_val
    );

    // Reshape
    const result = new Float32Array(totalSize);
    result.set(denormalized);
    return result;
  } else {
    // No cuantizado, convertir según dtype
    if (encoded.dtype.includes("float32")) {
      const float32Length = Math.floor(dataBytes.length / 4);
      const buffer = dataBytes.buffer.slice(0, float32Length * 4);
      return new Float32Array(buffer);
    } else if (encoded.dtype.includes("float64")) {
      const float64Length = Math.floor(dataBytes.length / 8);
      const buffer = dataBytes.buffer.slice(0, float64Length * 8);
      return new Float64Array(buffer);
    } else if (encoded.dtype.includes("uint8")) {
      // uint8 sin cuantización (caso especial)
      return new Float32Array(dataBytes);
    } else {
      // Default a float32
      const float32Length = Math.floor(dataBytes.length / 4);
      const buffer = dataBytes.buffer.slice(0, float32Length * 4);
      return new Float32Array(buffer);
    }
  }
}

/**
 * Convierte un TypedArray a array 2D de números.
 */
function typedArrayTo2D(
  arr: Float32Array | Float64Array,
  shape: number[]
): number[][] {
  const [H, W] = shape;
  const result: number[][] = [];

  for (let y = 0; y < H; y++) {
    const row: number[] = [];
    for (let x = 0; x < W; x++) {
      const idx = y * W + x;
      if (idx < arr.length) {
        row.push(arr[idx]);
      } else {
        row.push(0);
      }
    }
    result.push(row);
  }

  return result;
}

export function decompressArray(
  compressed: CompressedArray | BinaryCompressedArray
): number[][] {
  /**
   * Descomprime un array comprimido desde el servidor.
   * Soporta formato antiguo (base64 + zlib) y formato nuevo (binario + LZ4 + quantización).
   *
   * @param compressed - Objeto con data, shape, dtype, etc.
   * @returns Array 2D descomprimido
   */
  try {
    // Detectar formato nuevo (binario)
    if (
      ("format" in compressed && (compressed as any).format === "binary") ||
      "quantized" in compressed
    ) {
      const binaryCompressed = compressed as BinaryCompressedArray;
      const typedArray = decodeArrayBinary(binaryCompressed);
      return typedArrayTo2D(typedArray, binaryCompressed.shape);
    }

    // Formato antiguo (base64 + zlib)
    const oldFormat = compressed as CompressedArray;

    // Decodificar base64 a bytes
    const binaryString = atob(oldFormat.data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    // Descomprimir con pako
    let decompressed: Uint8Array;
    try {
      decompressed = pako.inflate(bytes);
    } catch (e) {
      console.warn(
        "Error descomprimiendo con pako, usando datos sin descomprimir:",
        e
      );
      decompressed = bytes;
    }

    // Convertir bytes a array según dtype
    let values: Float32Array | Float64Array;
    if (oldFormat.dtype.includes("float32")) {
      const byteLength = decompressed.length;
      const float32Length = Math.floor(byteLength / 4);
      const buffer = decompressed.buffer.slice(0, float32Length * 4);
      values = new Float32Array(buffer);
    } else if (oldFormat.dtype.includes("float64")) {
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
    return typedArrayTo2D(values, oldFormat.shape);
  } catch (error) {
    console.error("Error descomprimiendo array:", error);
    // Retornar array vacío como fallback
    return [];
  }
}

export function isCompressed(
  data: any
): data is CompressedArray | BinaryCompressedArray {
  /**
   * Verifica si un dato es un array comprimido (formato antiguo o nuevo).
   */
  return (
    typeof data === "object" &&
    data !== null &&
    "compressed" in data &&
    data.compressed === true &&
    "data" in data &&
    "shape" in data &&
    "dtype" in data
  );
}

/**
 * Decodifica un frame binario completo (MessagePack, CBOR o JSON).
 *
 * Soporta:
 * - MessagePack: Más eficiente para arrays numéricos (preferido)
 * - CBOR: Formato binario alternativo
 * - JSON: Fallback para compatibilidad
 */
export async function decodeBinaryFrame(
  data: ArrayBuffer | Uint8Array | string,
  format?: string
): Promise<any> {
  try {
    let bytes: Uint8Array;

    // Convertir a Uint8Array si es necesario
    if (typeof data === "string") {
      // String JSON (fallback)
      return JSON.parse(data);
    } else if (data instanceof ArrayBuffer) {
      bytes = new Uint8Array(data);
    } else {
      bytes = data;
    }

    // Si el formato está especificado, intentar ese formato primero
    if (format === "msgpack" || format === "cbor") {
      try {
        // @msgpack/msgpack puede decodificar tanto MessagePack como CBOR
        const decoded = cborDecode(bytes);
        if (decoded && typeof decoded === "object") {
          return decoded;
        }
      } catch (decodeError) {
        // Si el formato está especificado y falla, continuar con fallbacks
        console.warn(
          `Error decodificando ${format}, intentando auto-detección:`,
          decodeError
        );
      }
    }

    // Auto-detección: intentar detectar formato por primer byte
    if (bytes[0] === 0x7b || bytes[0] === 0x5b) {
      // '{' o '['
      // Probablemente JSON
      const text = new TextDecoder("utf-8").decode(bytes);
      return JSON.parse(text);
    }

    // Intentar MessagePack/CBOR usando @msgpack/msgpack (puede decodificar ambos)
    if (!format || format === "msgpack" || format === "cbor") {
      try {
        const decoded = cborDecode(bytes);
        if (decoded && typeof decoded === "object") {
          return decoded;
        }
      } catch (decodeError) {
        // No es MessagePack/CBOR válido, continuar con JSON
      }
    }

    // Fallback: intentar como JSON
    const text = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
    if (text) {
      return JSON.parse(text);
    }

    throw new Error(
      "No se pudo decodificar el frame (no es JSON ni CBOR válido)"
    );
  } catch (error) {
    console.error("Error decodificando frame binario:", error);
    throw error;
  }
}

/**
 * Procesa un payload decodificado y descomprime arrays binarios.
 */
export function processDecodedPayload(
  frameData: any,
  previousFrame?: any
): any {
  try {
    const payload: any = { ...(frameData.metadata || {}) };
    const arrays = frameData.arrays || {};

    // Decodificar arrays binarios
    for (const [key, encoded] of Object.entries(arrays)) {
      const encodedArray = encoded as BinaryCompressedArray;

      if (encodedArray.format === "binary") {
        // Convertir data a Uint8Array si viene como base64 string (JSON fallback)
        if (typeof encodedArray.data === "string") {
          const binaryString = atob(encodedArray.data);
          encodedArray.data = new Uint8Array(binaryString.length);
          for (let i = 0; i < binaryString.length; i++) {
            (encodedArray.data as Uint8Array)[i] = binaryString.charCodeAt(i);
          }
        }

        const typedArray = decodeArrayBinary(encodedArray);
        const array2D = typedArrayTo2D(typedArray, encodedArray.shape);

        // Si es differential, reconstruir desde frame anterior
        if (
          encodedArray.is_differential &&
          previousFrame &&
          previousFrame[key]
        ) {
          const prev = Array.isArray(previousFrame[key])
            ? previousFrame[key]
            : [[0]];
          // Sumar diff al frame anterior
          const reconstructed: number[][] = [];
          for (let y = 0; y < array2D.length; y++) {
            const row: number[] = [];
            for (let x = 0; x < array2D[y].length; x++) {
              const prevVal = prev[y] && prev[y][x] ? prev[y][x] : 0;
              row.push(prevVal + array2D[y][x]);
            }
            reconstructed.push(row);
          }
          payload[key] = reconstructed;
        } else {
          payload[key] = array2D;
        }
      } else {
        // Otros datos (no binarios)
        payload[key] = encoded;
      }
    }

    return payload;
  } catch (error) {
    console.error("Error procesando payload decodificado:", error);
    return frameData;
  }
}

export function decompressIfNeeded(data: any): any {
  /**
   * Descomprime un dato si está comprimido, sino lo retorna tal cual.
   * Soporta formato antiguo y nuevo.
   */
  if (isCompressed(data)) {
    return decompressArray(data);
  }
  return data;
}
