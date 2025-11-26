import {
  Activity,
  AlertCircle,
  Aperture,
  ChevronRight,
  Database,
  Microscope,
  MoreHorizontal,
  Pause,
  Play,
  Settings,
  Share2,
  Terminal,
  Zap
} from 'lucide-react';
import React, { useState } from 'react';

// --- SUBCOMPONENTES DE UI (REFINADOS) ---

const GlassPanel = ({ children, className = "" }: { children: React.ReactNode, className?: string }) => (
  <div className={`bg-[#0a0a0a]/90 backdrop-blur-md border border-white/10 shadow-lg rounded-lg ${className}`}>
    {children}
  </div>
);

const MetricItem = ({ label, value, unit, status = "neutral" }: { label: string, value: string, unit?: string, status?: "neutral" | "good" | "warning" | "critical" }) => {
  const statusColor = 
    status === "good" ? "text-emerald-400" :
    status === "warning" ? "text-amber-400" :
    status === "critical" ? "text-rose-400" : "text-gray-300";

  return (
    <div className="flex flex-col border-l-2 border-white/5 pl-3 py-1">
      <span className="text-[10px] uppercase tracking-widest font-semibold text-gray-500 mb-1">{label}</span>
      <div className="flex items-baseline gap-1.5">
        <span className={`text-lg font-mono font-medium ${statusColor}`}>{value}</span>
        {unit && <span className="text-[10px] text-gray-600 font-mono uppercase">{unit}</span>}
      </div>
    </div>
  );
};

const EpochBadge = ({ era, current }: { era: number, current: number }) => {
  const labels = ["VACÍO", "CUÁNTICA", "PARTÍCULAS", "QUÍMICA", "GRAVEDAD", "BIOLOGÍA"];
  const isActive = era === current;
  const isPast = era < current;
  
  return (
    <div className={`flex items-center gap-2 px-3 py-1 rounded border text-[10px] font-mono font-medium tracking-wider transition-all ${
      isActive 
        ? 'bg-blue-500/10 border-blue-500/40 text-blue-400' 
        : isPast 
          ? 'bg-white/5 border-white/5 text-gray-600' 
          : 'bg-transparent border-transparent text-gray-800'
    }`}>
      <div className={`w-1 h-1 rounded-full ${isActive ? 'bg-blue-400 shadow-[0_0_5px_cyan]' : isPast ? 'bg-gray-600' : 'bg-gray-800'}`} />
      {labels[era]}
    </div>
  );
};

// --- DASHBOARD PRINCIPAL ---

const DashboardMockup: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'lab' | 'analysis' | 'history' | 'logs'>('lab');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(2); // Era de Partículas

  return (
    <div className="h-screen bg-[#020202] text-gray-300 font-sans selection:bg-blue-500/30 overflow-hidden flex flex-col">
      
      {/* --- HEADER: BARRA DE COMANDO TÉCNICA --- */}
      <header className="h-12 border-b border-white/10 bg-[#050505] flex items-center justify-between px-4 z-50 shrink-0">
        <div className="flex items-center gap-6">
          {/* Identidad */}
          <div className="flex items-center gap-3 group cursor-pointer opacity-90 hover:opacity-100 transition-opacity">
            <div className="relative w-6 h-6 flex items-center justify-center border border-white/10 rounded bg-white/5">
              <Aperture size={14} className="text-gray-300" />
            </div>
            <div className="flex flex-col leading-none">
              <span className="text-xs font-bold text-gray-200 tracking-wide">ATHERIA<span className="text-blue-500">_LAB</span></span>
              <span className="text-[8px] text-gray-600 font-mono uppercase mt-0.5">Ver. 4.0.2-RC</span>
            </div>
          </div>

          {/* Separador Vertical */}
          <div className="h-4 w-px bg-white/10" />

          {/* Timeline de Épocas (Compacto) */}
          <div className="flex items-center gap-0.5">
            {[0, 1, 2, 3, 4, 5].map(e => <EpochBadge key={e} era={e} current={currentEpoch} />)}
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Indicador de Estado del Sistema */}
          <div className="flex items-center gap-2 px-2 py-1 bg-white/5 rounded border border-white/5">
             <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.4)]" />
             <span className="text-[10px] font-mono text-emerald-500 tracking-wide">SPARSE_ENGINE::READY</span>
          </div>
          
          <div className="h-4 w-px bg-white/10" />

          {/* Perfil / Config */}
          <button className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors"><Settings size={16} /></button>
          <div className="w-6 h-6 rounded bg-gray-800 border border-gray-700 flex items-center justify-center text-[10px] text-gray-400 font-bold">JS</div>
        </div>
      </header>

      {/* --- CONTENEDOR PRINCIPAL --- */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* 1. SIDEBAR DE NAVEGACIÓN (Iconos Minimalistas) */}
        <aside className="w-12 border-r border-white/5 bg-[#050505] flex flex-col items-center py-3 gap-2 z-40 shrink-0">
          {[
            { id: 'lab', icon: Microscope, label: 'Lab' },
            { id: 'analysis', icon: Activity, label: 'Analytics' },
            { id: 'history', icon: Database, label: 'Data' },
            { id: 'logs', icon: Terminal, label: 'Console' },
          ].map((item) => (
            <button 
              key={item.id}
              onClick={() => setActiveTab(item.id as any)}
              className={`w-8 h-8 rounded flex items-center justify-center transition-all relative group ${
                activeTab === item.id 
                  ? 'bg-blue-500/10 text-blue-400' 
                  : 'text-gray-600 hover:text-gray-300 hover:bg-white/5'
              }`}
            >
              <item.icon size={16} strokeWidth={2} />
              {/* Indicador Activo */}
              {activeTab === item.id && <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-blue-500 rounded-r" />}
            </button>
          ))}
        </aside>

        {/* 2. ÁREA DE TRABAJO (Viewport + Paneles Flotantes) */}
        <main className="flex-1 relative bg-black flex flex-col overflow-hidden">
          
          {/* Barra de Herramientas Superior (Flotante) */}
          <div className="absolute top-4 left-4 z-30 flex gap-2 pointer-events-none">
            <GlassPanel className="pointer-events-auto flex items-center p-1 gap-1">
               <button 
                 onClick={() => setIsPlaying(!isPlaying)}
                 className={`px-4 py-1.5 rounded text-xs font-bold flex items-center gap-2 transition-all border ${
                   isPlaying 
                     ? 'bg-amber-500/10 text-amber-500 border-amber-500/30 hover:bg-amber-500/20' 
                     : 'bg-white/5 text-gray-200 border-white/10 hover:bg-white/10'
                 }`}
               >
                 {isPlaying ? <Pause size={12} fill="currentColor" /> : <Play size={12} fill="currentColor" />}
                 {isPlaying ? 'PAUSAR' : 'EJECUTAR'}
               </button>
               
               <div className="w-px h-4 bg-white/10 mx-2" />
               
               <div className="flex flex-col px-2">
                 <span className="text-[8px] text-gray-500 uppercase font-bold">Paso Actual</span>
                 <span className="text-xs font-mono text-gray-200">2,405,100</span>
               </div>
            </GlassPanel>

            <GlassPanel className="pointer-events-auto flex items-center px-3 py-1 gap-3">
                <div className="flex items-center gap-2">
                   <span className="text-[10px] font-bold text-gray-500">FPS</span>
                   <span className="text-xs font-mono text-emerald-400">118</span>
                </div>
                <div className="w-px h-3 bg-white/10" />
                <div className="flex items-center gap-2">
                   <span className="text-[10px] font-bold text-gray-500">PARTÍCULAS</span>
                   <span className="text-xs font-mono text-blue-400">14.2K</span>
                </div>
            </GlassPanel>
          </div>

          {/* VIEWPORT 3D (Fondo) */}
          <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black overflow-hidden">
            {/* Grid de Fondo Estático para el mockup */}
            <div 
              className="absolute inset-0 opacity-10" 
              style={{ 
                backgroundImage: 'linear-gradient(#222 1px, transparent 1px), linear-gradient(90deg, #222 1px, transparent 1px)', 
                backgroundSize: '50px 50px',
                transform: 'perspective(500px) rotateX(60deg) translateY(-100px) scale(2)',
                transformOrigin: 'top center'
              }} 
            />
            
            {/* Partículas Simuladas (Centro) */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px]">
               <div className="absolute inset-0 bg-blue-500/5 blur-[100px] rounded-full animate-pulse" />
               {/* Puntos dispersos simulando materia */}
               <div className="absolute top-[30%] left-[40%] w-1 h-1 bg-blue-400 rounded-full shadow-[0_0_10px_cyan]" />
               <div className="absolute top-[50%] left-[60%] w-2 h-2 bg-white rounded-full shadow-[0_0_15px_white]" />
               <div className="absolute bottom-[40%] right-[30%] w-1.5 h-1.5 bg-purple-400 rounded-full shadow-[0_0_10px_purple]" />
            </div>

            <div className="absolute bottom-4 right-4 text-[9px] font-mono text-gray-700 pointer-events-none text-right">
              VIEWPORT: ORTHOGRAPHIC<br/>
              RENDER: WEBGL2 / HIGH_PRECISION
            </div>
          </div>

          {/* PANEL INFERIOR (Métricas Críticas) */}
          <div className="mt-auto z-30 border-t border-white/10 bg-[#050505]/95 backdrop-blur-sm p-4">
            <div className="max-w-6xl mx-auto grid grid-cols-5 gap-6">
              <MetricItem label="Energía de Vacío" value="0.0042" unit="eV" status="good" />
              <MetricItem label="Entropía Local" value="12.50" unit="bits" status="neutral" />
              <MetricItem label="Simetría (IonQ)" value="0.98" unit="idx" status="good" />
              <MetricItem label="Decaimiento" value="1.2e-4" unit="rad/s" status="warning" />
              
              {/* Log Miniatura */}
              <div className="col-span-1 pl-4 border-l border-white/5 flex flex-col justify-center gap-1">
                 <div className="flex items-center gap-2 text-[10px] font-mono text-emerald-500/80">
                    <span className="w-1 h-1 rounded-full bg-emerald-500" />
                    Nucleación exitosa en sector [12, 4]
                 </div>
                 <div className="flex items-center gap-2 text-[10px] font-mono text-blue-500/80">
                    <span className="w-1 h-1 rounded-full bg-blue-500" />
                    Optimizando chunks inactivos...
                 </div>
              </div>
            </div>
          </div>
        </main>

        {/* 3. PANEL LATERAL DERECHO (Inspector y Controles) */}
        <aside className="w-72 border-l border-white/10 bg-[#080808] flex flex-col z-40 shrink-0">
          {/* Título Panel */}
          <div className="h-10 border-b border-white/5 flex items-center justify-between px-4 bg-[#0a0a0a]">
             <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Inspector Físico</span>
             <MoreHorizontal size={14} className="text-gray-600 cursor-pointer hover:text-gray-400" />
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-8 custom-scrollbar">
            
            {/* Sección: Parámetros Globales */}
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
                 <Settings size={12} className="text-blue-500" /> Variables de Entorno
              </div>
              
              {/* Control Slider Refinado */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[10px]">
                  <span className="text-gray-400">Gamma (Disipación)</span>
                  <span className="font-mono text-gray-200 bg-white/5 px-1.5 rounded">0.015</span>
                </div>
                <div className="h-1 w-full bg-gray-800 rounded-full overflow-hidden cursor-pointer hover:bg-gray-700 transition-colors">
                   <div className="h-full bg-gray-500 w-[30%]" />
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex justify-between text-[10px]">
                  <span className="text-gray-400">Ruido Térmico</span>
                  <span className="font-mono text-amber-400/80 bg-amber-900/10 px-1.5 rounded border border-amber-500/10">0.002</span>
                </div>
                <div className="h-1 w-full bg-gray-800 rounded-full overflow-hidden cursor-pointer hover:bg-gray-700 transition-colors">
                   <div className="h-full bg-amber-600 w-[10%]" />
                </div>
              </div>
            </div>

            {/* Sección: Inyección (Génesis) */}
            <div className="space-y-3 pt-4 border-t border-white/5">
               <div className="flex items-center gap-2 text-gray-200 text-xs font-bold uppercase tracking-wider mb-2">
                 <Zap size={12} className="text-yellow-500" /> Inyección de Energía
              </div>
              
              <div className="grid grid-cols-1 gap-2">
                <button className="flex items-center justify-between px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/5 rounded text-xs text-gray-300 transition-all group">
                   <span>Sopa Primordial</span>
                   <ChevronRight size={12} className="text-gray-600 group-hover:text-white" />
                </button>
                <button className="flex items-center justify-between px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/5 rounded text-xs text-gray-300 transition-all group">
                   <span>Monolito Denso</span>
                   <ChevronRight size={12} className="text-gray-600 group-hover:text-white" />
                </button>
                <button className="flex items-center justify-between px-3 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 rounded text-xs text-blue-200 transition-all group">
                   <span className="flex items-center gap-2"><Microscope size={12}/> Semilla Simétrica</span>
                   <ChevronRight size={12} className="text-blue-500" />
                </button>
              </div>
            </div>

             {/* Sección: Debug de Campos */}
             <div className="space-y-3 pt-4 border-t border-white/5">
               <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">Visualización de Campos</span>
               <div className="space-y-1">
                 {['Densidad (Scalar)', 'Fase (Complex)', 'Flujo (Vector)', 'Chunks (Meta)'].map((layer, i) => (
                   <label key={layer} className="flex items-center gap-2 cursor-pointer p-1.5 rounded hover:bg-white/5 transition-colors">
                      <div className={`w-3 h-3 rounded-sm border ${i === 0 ? 'bg-blue-500 border-blue-500' : 'border-gray-600 bg-transparent'} flex items-center justify-center`}>
                        {i === 0 && <div className="w-1.5 h-1.5 bg-white rounded-[1px]" />}
                      </div>
                      <span className={`text-xs ${i === 0 ? 'text-gray-200 font-medium' : 'text-gray-500'}`}>{layer}</span>
                   </label>
                 ))}
               </div>
            </div>
            
            {/* Mensaje de Alerta (Contextual) */}
            <div className="mt-4 p-3 rounded bg-red-500/5 border border-red-500/10 flex gap-2 items-start">
              <AlertCircle size={14} className="text-red-500 shrink-0 mt-0.5" />
              <div className="flex flex-col">
                <span className="text-[10px] font-bold text-red-400 uppercase">Inestabilidad Detectada</span>
                <span className="text-[10px] text-red-300/70 leading-tight">El vacío armónico muestra picos de energía anómalos en el cuadrante negativo.</span>
              </div>
            </div>

          </div>

          {/* Footer del Sidebar */}
          <div className="p-3 border-t border-white/5 bg-[#080808]">
             <button className="w-full py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-bold rounded border border-white/5 transition-all flex items-center justify-center gap-2">
               <Share2 size={14} /> CAPTURAR ESTADO
             </button>
          </div>
        </aside>

      </div>
    </div>
  );
};

export default DashboardMockup;