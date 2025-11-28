# Guía de Cambio de Motor (Engine Switching)

Atheria 4 soporta dos motores de simulación principales:

1.  **Motor Python (Harmónico/Standard):**
    *   **Ventajas:** Flexible, fácil de depurar, soporta inyección de energía compleja.
    *   **Desventajas:** Menor rendimiento en grids grandes.
    *   **Uso:** Ideal para experimentación rápida y desarrollo.

2.  **Motor Nativo (C++):**
    *   **Ventajas:** Alto rendimiento, paralelismo real, soporta grids masivos.
    *   **Desventajas:** Menos flexible, requiere compilación.
    *   **Uso:** Ideal para simulaciones de larga duración y alta resolución.

## ¿Cómo cambiar de motor?

Actualmente, el cambio de motor se realiza mediante comandos en la consola de desarrollo o recargando el experimento.

### Opción 1: Recargar Experimento (Recomendado)

Cuando cargas un experimento, puedes forzar el motor que deseas usar.

**Comando:**
```json
{
  "type": "command",
  "content": "/load_experiment nombre_del_experimento --engine=native"
}
```

Opciones para `--engine`:
*   `native`: Fuerza el uso del motor C++.
*   `python`: Fuerza el uso del motor Python.
*   `harmonic`: Fuerza el uso del motor Harmónico (variante de Python).
*   `auto`: Deja que el sistema decida (por defecto usa Nativo si está disponible).

### Opción 2: Comando `switch_engine`

Si ya tienes un experimento cargado, puedes intentar cambiar el motor "en caliente" (esto recargará el experimento internamente).

**Comando:**
```json
{
  "type": "command",
  "content": "/switch_engine native"
}
```

### Verificación

Para verificar qué motor estás usando, observa los logs del servidor o la notificación de inicio:
*   "Inicializando motor nativo..."
*   "Inicializando motor Python..."

## Solución de Problemas

*   **Error "atheria_core no encontrado":** Significa que el motor nativo no está compilado o instalado. Ejecuta `ath dev` para recompilar.
*   **Crash al cambiar a Nativo:** Asegúrate de que tu GPU tenga suficiente memoria. El motor nativo puede ser más agresivo con el uso de memoria.
