# Mejoras Sugeridas para Cursor Agents

## Análisis Basado en Correcciones del Proyecto Atheria 4

### 1. **Detección de Redundancias y Duplicación**
**Problema Observado:**
- El agente creó botones duplicados (`EJECUTAR`, `REINICIAR`) en `Toolbar` y `LabSider` sin verificar si ya existían.
- Funcionalidad duplicada causó confusión de UI.

**Mejora Sugerida:**
- Antes de crear un componente/acción, buscar si ya existe en el codebase.
- Usar codebase_search para encontrar funcionalidades similares antes de implementar.
- Crear una función de validación que detecte duplicados funcionales.

**Ejemplo de Checklist:**
```typescript
// Antes de agregar un botón "Ejecutar":
// 1. Buscar: "ejecutar|run|play|start" en el código
// 2. Verificar si ya existe una función conectada
// 3. Si existe, reutilizar en lugar de duplicar
```

---

### 2. **Gestión de Estados Desconectados**
**Problema Observado:**
- El agente mostraba FPS y métricas incluso cuando `connectionStatus === 'disconnected'`.
- Variables mostraban datos obsoletos o incorrectos cuando no había conexión activa.

**Mejora Sugerida:**
- Siempre validar el estado de conexión antes de mostrar datos dinámicos.
- Implementar "guards" de estado: `if (!isConnected) return <DisconnectedState />`
- Limpiar datos obsoletos cuando se pierde la conexión.

**Patrón Recomendado:**
```typescript
// ❌ MAL: Mostrar datos siempre
<div>{simData.fps} FPS</div>

// ✅ BIEN: Validar estado
{connectionStatus === 'connected' ? (
  <div>{simData.fps} FPS</div>
) : (
  <div className="text-gray-500">-- FPS</div>
)}
```

---

### 3. **Migraciones Completas vs. Parciales**
**Problema Observado:**
- El agente dejó componentes comentados (`CheckpointManager`, `TransferLearningWizard`) con `// TODO: Migrar a Tailwind`.
- Esto generó confusión sobre qué está funcional y qué no.

**Mejora Sugerida:**
- **Opción A (Completa)**: Completar la migración de TODOS los componentes necesarios.
- **Opción B (Temporal)**: Si se comentan componentes, documentar claramente:
  - Por qué están comentados
  - Cuándo se restaurarán
  - Qué funcionalidad se pierde temporalmente
- Crear un archivo `MIGRATION_STATUS.md` para rastrear el progreso.

**Template de Comentario:**
```typescript
// TEMPORALMENTE DESHABILITADO: CheckpointManager
// Razón: Requiere migración completa de Mantine a Tailwind
// Impacto: No se pueden gestionar checkpoints desde la UI
// Plan de Restauración: Fase 3.1 (ver ROADMAP_PHASE_1.md)
// Fecha de Comentado: 2024-01-XX
```

---

### 4. **Validación de Sistema de Diseño (Design System)**
**Problema Observado:**
- El agente no siguió estrictamente el Design System (`DESIGN_SYSTEM.md`).
- Colores hardcodeados (`#1a1b1e`, etc.) en lugar de tokens del sistema.

**Mejora Sugerida:**
- Leer `DESIGN_SYSTEM.md` ANTES de cualquier cambio de UI.
- Validar que todos los colores usen los tokens definidos.
- Crear una función de validación que verifique:
  - Colores: Solo usar `#020202`, `#050505`, `#080808`, `#0a0a0a`
  - Espaciado: Solo usar valores del sistema (4px, 8px, 16px, etc.)
  - Tipografía: Solo usar `font-mono`, `font-sans`, tamaños definidos

**Checklist Pre-UI:**
```markdown
1. Leer DESIGN_SYSTEM.md
2. Identificar componentes base (GlassPanel, MetricItem, etc.)
3. Usar solo tokens de color definidos
4. Validar espaciado y tipografía
```

---

### 5. **Verificación de Conexión Funcional**
**Problema Observado:**
- Botones que no funcionaban (botón de configuración en header).
- Configuraciones que no se aplicaban correctamente.

**Mejora Sugerida:**
- Después de agregar un botón/acción, verificar:
  1. ¿Está conectado a una función?
  2. ¿La función existe en el contexto/hook correcto?
  3. ¿Los parámetros se pasan correctamente?
- Crear tests básicos para acciones críticas:
  ```typescript
  // Verificar que el botón "Config" abre el panel
  expect(settingsPanelOpen).toBe(true);
  ```

---

### 6. **Manejo de Contexto y Prop Drilling**
**Problema Observado:**
- Props que no se pasaban correctamente entre componentes.
- Estados duplicados en lugar de usar contexto compartido.

**Mejora Sugerida:**
- Antes de crear un nuevo estado, verificar si ya existe en un contexto (ej: `WebSocketContext`).
- Usar el hook/contexto apropiado en lugar de pasar props manualmente.
- Documentar qué estados son globales vs. locales.

**Ejemplo:**
```typescript
// ❌ MAL: Prop drilling
<Parent activeTab={activeTab} onTabChange={setActiveTab} />
  <Child activeTab={activeTab} onTabChange={setActiveTab} />

// ✅ BIEN: Contexto compartido
const { activeTab, setActiveTab } = useDashboardContext();
```

---

### 7. **Atomicidad de Cambios**
**Problema Observado:**
- Cambios grandes que rompen múltiples cosas a la vez.
- Difícil de revertir o debuggear.

**Mejora Sugerida:**
- Hacer cambios en pasos pequeños y verificables.
- Probar cada paso antes de continuar.
- Usar `todo_write` para planificar y rastrear cambios complejos.

**Estrategia:**
1. **Paso 1**: Crear estructura base
2. **Paso 2**: Conectar datos
3. **Paso 3**: Aplicar estilos
4. **Paso 4**: Validar funcionalidad

---

### 8. **Documentación de Decisiones**
**Problema Observado:**
- El agente no documentaba por qué tomó ciertas decisiones.
- Difícil entender el razonamiento después.

**Mejora Sugerida:**
- Agregar comentarios explicativos para decisiones no obvias:
  ```typescript
  // Usamos debounce (500ms) para evitar actualizaciones
  // excesivas de ROI durante pan/zoom. El throttle (300ms)
  // previene actualizaciones demasiado frecuentes.
  ```
- Actualizar `AI_DEV_LOG.md` con decisiones importantes.

---

## Prioridades

### 🔴 Alta Prioridad
1. **Detección de redundancias** - Ahorra tiempo y mejora UX
2. **Validación de estados desconectados** - Crítico para funcionalidad
3. **Verificación de conexión funcional** - Evita bugs obvios

### 🟡 Media Prioridad
4. **Migraciones completas** - Mejora mantenibilidad
5. **Validación de Design System** - Mejora consistencia visual
6. **Atomicidad de cambios** - Reduce riesgo

### 🟢 Baja Prioridad
7. **Manejo de contexto** - Optimización
8. **Documentación de decisiones** - Mejora legibilidad

---

## Implementación en Cursor

### Prompt Mejorado para Agentes

```
Antes de hacer cambios de UI:

1. Buscar componentes/acciones similares existentes
2. Validar estado de conexión si es necesario
3. Leer DESIGN_SYSTEM.md para colores/espaciado
4. Verificar que funciones estén conectadas correctamente
5. Si comentas algo, documentar por qué y cuándo restaurarlo
6. Hacer cambios en pasos pequeños y verificables
7. Probar cada cambio antes de continuar

Después de cambios:
- Verificar que no haya redundancias
- Probar con conexión desconectada
- Validar que sigue el Design System
```

---

## Conclusión

Las mejoras más impactantes serían:
1. **Validación proactiva** de redundancias y estados
2. **Migraciones completas** en lugar de parciales
3. **Verificación automática** de conexión funcional

Estas mejoras reducirían significativamente el tiempo de corrección y mejorarán la calidad del código generado.

