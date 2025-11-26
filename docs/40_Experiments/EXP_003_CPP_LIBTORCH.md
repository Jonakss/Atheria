Experimento 003: Integración de LibTorch en Motor C++

Fecha: 2025-11-19
Estado: En Progreso / Planeado
Objetivo: Lograr que C++ almacene y manipule tensores de PyTorch nativamente para eliminar el cuello de botella de conversión de datos.

Hipótesis

Al mover el almacenamiento de tensores al std::unordered_map de C++, eliminaremos la duplicidad de datos en Python. Aunque la llamada individual (add_particle) puede seguir teniendo overhead, operaciones masivas como step() (cuando se migren) serán órdenes de magnitud más rápidas al no tener que cruzar la barrera Python-C++ por cada partícula.

Configuración

Librerías: PyBind11 + LibTorch (ABI compatible con PyTorch Python).

Estructura de Datos: std::unordered_map<Coord, torch::Tensor>.

Resultados Esperados

Compilación exitosa de atheria_core con links a Torch.

Script de Python capaz de enviar un Tensor a C++ y recibirlo de vuelta sin corrupción de memoria.

Benchmark de memoria: El uso de RAM debería ser eficiente.

Bitácora de Ejecución

[ ] Configuración de CMake para encontrar LibTorch.

[ ] Implementación de SparseMap con Tensores.

[ ] Pruebas de integridad de datos.