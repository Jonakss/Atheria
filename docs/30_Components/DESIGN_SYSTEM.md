🎨 Atheria 4: Design System (Scientific Dark)

Filosofía: "Instrumento Científico de Precisión".
Interfaz de alta densidad de datos, distracciones mínimas, jerarquía clara y estética de "cristal oscuro".

1. Paleta de Colores (Tailwind Tokens)

Fondos (Backgrounds)

Canvas / Base: #020202 (Negro casi puro) - bg-[#020202]

Paneles (Surface): #050505 con 90% opacidad - bg-[#050505]/90

Paneles Secundarios: #080808 - bg-[#080808]

Glass (Overlay): #0a0a0a con 80% opacidad + Blur - bg-[#0a0a0a]/80 backdrop-blur-md

Bordes (Borders)

Sutil: Blanco al 5% - border-white/5

Medio: Blanco al 10% - border-white/10

Activo/Foco: Azul al 40% - border-blue-500/40

Tipografía & Texto

Headings (H1-H3): text-gray-100 (Casi blanco, no #FFF puro para evitar fatiga)

Body: text-gray-300

Muted / Labels: text-gray-500

Data / Numbers: font-mono (Monospace obligatoria para datos).

Semántica (Status Colors)

Paleta Principal: Teal & Pink (Inspirada en infografía moderna)

Primary (Acción/Selección): Teal (verde azulado)

Texto: text-teal-400

Fondo sutil: bg-teal-500/10

Borde activo: border-teal-500/40

Glow: shadow-glow-teal o shadow-[0_0_15px_rgba(20,184,166,0.4)]

Secondary (Energía/Vacío): Pink/Rosa

Texto: text-pink-400

Fondo sutil: bg-pink-500/10

Borde activo: border-pink-500/40

Glow: shadow-glow-pink o shadow-[0_0_15px_rgba(236,72,153,0.4)]

Gradientes: bg-gradient-teal-pink (para efectos especiales)

Success (Estable): Teal más oscuro

Texto: text-teal-600

Indicador: bg-teal-500

Warning (Inestable/Transición): Amber (mantener para transiciones)

Texto: text-amber-400

Critical (Error/Colapso): Pink más intenso

Texto: text-pink-600

Physics (Energía/Vacío): Gradiente Teal→Pink

Texto: text-teal-400 (con efectos de gradiente opcionales)

Glow combinado: shadow-glow-gradient

2. Componentes Core (Building Blocks)

A. GlassPanel (Contenedor Estándar)

El bloque fundamental de la UI. Todo contenido debe vivir aquí.

<div className="bg-[#0a0a0a]/90 backdrop-blur-md border border-white/10 shadow-lg rounded-lg">
  {children}
</div>


B. MetricItem (Dato Científico)

Para mostrar valores numéricos. Siempre usa borde izquierdo.

<div className="flex flex-col border-l-2 border-white/5 pl-3 py-1">
  <span className="text-[10px] uppercase tracking-widest font-semibold text-gray-500 mb-1">
    LABEL
  </span>
  <div className="flex items-baseline gap-1.5">
    <span className="text-lg font-mono font-medium text-gray-100">VALUE</span>
    <span className="text-[10px] text-gray-600 font-mono uppercase">UNIT</span>
  </div>
</div>


C. IconButton (Navegación)

Botones cuadrados para barras laterales.

Normal: text-gray-600 hover:text-gray-300 hover:bg-white/5

Activo: bg-teal-500/10 text-teal-400 + Indicador de borde izquierdo (border-l-2 border-teal-500) + shadow-glow-teal.

D. EpochBadge (Estado del Sistema)

Etiquetas pequeñas tipo "pill".

Estilo: text-[10px] font-mono font-medium tracking-wider px-3 py-1 rounded border.

Activo: bg-teal-500/10 border-teal-500/40 text-teal-400 + shadow-glow-teal.

Inactivo: bg-white/5 border-white/5 text-gray-600.

3. Layout & Espaciado

Grid Principal: Flexbox Column (Header + Body).

Body: Flexbox Row (SidebarIzq + Main + SidebarDer).

Sidebar Izquierdo: Ancho fijo w-12 (48px) o w-16 (64px). Iconos centrados.

Sidebar Derecho: Ancho fijo w-72 (288px) o w-80 (320px). Scrollable.

Main Viewport: flex-1 (Ocupa todo el espacio restante). relative para permitir overlays absolutos.

4. Efectos Especiales

Glow: Usar shadow-[color] para simular luz emitida por pantallas.

Ej: shadow-[0_0_10px_cyan] para partículas.

Scanlines (Opcional): Una textura de fondo muy sutil (opacity-5) para dar textura.

backgroundImage: 'linear-gradient(#222 1px, transparent 1px)'