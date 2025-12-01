// frontend/src/modules/Dashboard/components/HistoryView.tsx
import React, { useRef, useState } from 'react';
import { Database, Clock, Download, Trash2, Upload, RefreshCw, Search, Loader2 } from 'lucide-react';
import { useWebSocket } from '../../../hooks/useWebSocket';
import { GlassPanel } from './GlassPanel';
import { Alert } from './Alert';
import { Modal } from './Modal';
import { API_ENDPOINTS } from '../../../utils/serverConfig';

export const HistoryView: React.FC = () => {
  const { experimentsData, connectionStatus, sendCommand } = useWebSocket();
  const isConnected = connectionStatus === 'connected';

  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ type: 'success' | 'error' | 'info', message: string } | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [experimentToDelete, setExperimentToDelete] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Filtrar experimentos
  const filteredExperiments = experimentsData?.filter(exp =>
    exp.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (exp.model_architecture && exp.model_architecture.toLowerCase().includes(searchTerm.toLowerCase()))
  ).sort((a, b) => {
      // Sort by creation date descending if available
      if (a.created_at && b.created_at) {
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }
      return 0;
  }) || [];

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleRefresh = () => {
    sendCommand('system', 'refresh_experiments');
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset status
    setUploadStatus(null);

    // Validate extension
    const validExtensions = ['.zip', '.pth', '.pt'];
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();

    if (!validExtensions.includes(extension)) {
        setUploadStatus({
            type: 'error',
            message: 'Formato no soportado. Use .zip para experimentos o .pth/.pt para modelos.'
        });
        return;
    }

    setIsUploading(true);
    setUploadStatus({ type: 'info', message: `Subiendo ${file.name}...` });

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(API_ENDPOINTS.UPLOAD_MODEL, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (response.ok) {
            setUploadStatus({
                type: 'success',
                message: data.message || `Archivo ${file.name} importado correctamente.`
            });
            // Refresh list
            handleRefresh();
        } else {
            throw new Error(data.error || 'Error en la subida');
        }
    } catch (error: any) {
        setUploadStatus({
            type: 'error',
            message: error.message || 'Error al conectar con el servidor.'
        });
    } finally {
        setIsUploading(false);
        // Clear input so same file can be selected again if needed
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    }
  };

  const handleDeleteClick = (expName: string) => {
      setExperimentToDelete(expName);
      setDeleteConfirmOpen(true);
  };

  const confirmDelete = () => {
      if (experimentToDelete) {
          sendCommand('experiment', 'delete', { EXPERIMENT_NAME: experimentToDelete });
          setDeleteConfirmOpen(false);
          setExperimentToDelete(null);
      }
  };

  const handleExportClick = async (expName: string) => {
      // Use window.location to trigger download
      window.open(`${API_ENDPOINTS.EXPORT_EXPERIMENT}?name=${encodeURIComponent(expName)}`, '_blank');
  };

  if (!isConnected) {
    return (
      <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black flex items-center justify-center">
        <div className="text-gray-600 text-sm">Conectando al servidor...</div>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#050505] to-black overflow-auto custom-scrollbar">
      <div className="p-6 max-w-7xl mx-auto space-y-6">
        {/* Header with Search and Actions */}
        <div className="flex flex-col gap-4 mb-6">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Database size={20} className="text-blue-400" />
                    <h2 className="text-lg font-bold text-gray-200">Historial de Experimentos</h2>
                    <span className="bg-white/5 px-2 py-0.5 rounded-full text-xs text-gray-500 font-mono">
                        {filteredExperiments.length}
                    </span>
                </div>
                <div className="flex items-center gap-2">
                     <button
                        onClick={handleRefresh}
                        className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded transition-all"
                        title="Actualizar lista"
                    >
                        <RefreshCw size={16} />
                    </button>
                    <div className="h-4 w-px bg-white/10 mx-1"></div>
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                        accept=".zip,.pth,.pt"
                    />
                    <button
                        onClick={handleImportClick}
                        disabled={isUploading}
                        className="flex items-center gap-2 px-3 py-2 bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/30 text-blue-400 text-xs font-bold rounded transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isUploading ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
                        {isUploading ? 'Subiendo...' : 'Importar'}
                    </button>
                </div>
            </div>

            {/* Search Bar */}
            <div className="relative">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input
                    type="text"
                    placeholder="Buscar experimento..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-lg py-2 pl-9 pr-4 text-sm text-gray-300 placeholder-gray-600 focus:outline-none focus:border-blue-500/50 focus:bg-white/10 transition-all"
                />
            </div>

            {/* Upload Status Alert */}
            {uploadStatus && (
                <Alert
                    color={uploadStatus.type === 'error' ? 'red' : uploadStatus.type === 'success' ? 'green' : 'blue'}
                    onClose={() => setUploadStatus(null)}
                    withCloseButton
                >
                    {uploadStatus.message}
                </Alert>
            )}
        </div>

        {/* Lista de Experimentos */}
        {filteredExperiments.length > 0 ? (
          <div className="space-y-3">
            {filteredExperiments.map((exp) => (
              <GlassPanel key={exp.name} className="p-4 hover:bg-white/5 transition-colors group">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-sm font-bold text-gray-200 group-hover:text-white transition-colors">{exp.name}</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        exp.has_checkpoint 
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' 
                          : 'bg-gray-500/10 text-gray-400 border border-gray-500/20'
                      }`}>
                        {exp.has_checkpoint ? '✓ Entrenado' : '○ Sin entrenar'}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-xs text-gray-400 mt-3">
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Arquitectura</div>
                        <div className="font-mono text-gray-300">{exp.model_architecture || 'N/A'}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Grid Size</div>
                        <div className="font-mono text-gray-300">{exp.grid_size_training || 'N/A'}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-gray-500 uppercase mb-1">Episodios</div>
                        <div className="font-mono text-gray-300">{exp.total_episodes || 0}</div>
                      </div>
                    </div>
                    {exp.created_at && (
                      <div className="flex items-center gap-2 mt-3 text-[10px] text-gray-600">
                        <Clock size={12} />
                        <span>Creado: {new Date(exp.created_at).toLocaleString()}</span>
                      </div>
                    )}
                  </div>
                  <div className="flex items-center gap-2 ml-4">
                    <button 
                      onClick={() => handleExportClick(exp.name)}
                      className="p-2 text-gray-400 hover:text-gray-200 hover:bg-white/10 rounded transition-all"
                      title="Exportar"
                    >
                      <Download size={14} />
                    </button>
                    <button 
                      onClick={() => handleDeleteClick(exp.name)}
                      className="p-2 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded transition-all"
                      title="Eliminar"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              </GlassPanel>
            ))}
          </div>
        ) : (
          <GlassPanel className="p-8">
            <div className="text-center text-gray-600 text-sm">
              {searchTerm ? 'No se encontraron experimentos que coincidan con tu búsqueda' : 'No hay experimentos guardados aún'}
            </div>
            {searchTerm && (
                 <div className="flex justify-center mt-4">
                     <button
                        onClick={() => setSearchTerm('')}
                        className="text-blue-400 text-xs hover:underline"
                     >
                         Limpiar búsqueda
                     </button>
                 </div>
            )}
          </GlassPanel>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      <Modal
        opened={deleteConfirmOpen}
        onClose={() => setDeleteConfirmOpen(false)}
        title="Confirmar eliminación"
        size="sm"
      >
          <div className="space-y-4">
              <p className="text-sm text-gray-300">
                  ¿Estás seguro de que deseas eliminar el experimento <span className="font-bold text-white">{experimentToDelete}</span>?
                  <br /><br />
                  <span className="text-red-400">Esta acción no se puede deshacer.</span>
              </p>
              <div className="flex justify-end gap-3 pt-2">
                  <button
                      onClick={() => setDeleteConfirmOpen(false)}
                      className="px-3 py-2 text-xs font-bold text-gray-400 hover:text-white transition-colors"
                  >
                      Cancelar
                  </button>
                  <button
                      onClick={confirmDelete}
                      className="px-3 py-2 bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/30 rounded text-xs font-bold transition-all"
                  >
                      Eliminar
                  </button>
              </div>
          </div>
      </Modal>
    </div>
  );
};
