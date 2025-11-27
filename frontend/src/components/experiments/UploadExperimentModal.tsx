// frontend/src/components/experiments/UploadExperimentModal.tsx
import { useState, useRef } from 'react';
import { UploadCloud, X, FileArchive, FileCode, CheckCircle, AlertTriangle } from 'lucide-react';
import { GlassPanel } from '../../modules/Dashboard/components/GlassPanel';

interface UploadExperimentModalProps {
    onClose: () => void;
    onUploadSuccess: () => void;
}

const ARCHITECTURES = [
    { value: 'UNET', label: 'U-Net Standard' },
    { value: 'UNET_UNITARY', label: 'U-Net Unitary (Recommended)' },
    { value: 'SNN_UNET', label: 'SNN U-Net (Spiking)' },
    { value: 'DEEP_QCA', label: 'Deep QCA' },
    { value: 'MLP', label: 'MLP (Basic)' },
];

export function UploadExperimentModal({ onClose, onUploadSuccess }: UploadExperimentModalProps) {
    const [mode, setMode] = useState<'zip' | 'pth'>('zip');
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Form states for .pth mode
    const [expName, setExpName] = useState('');
    const [architecture, setArchitecture] = useState('UNET_UNITARY');
    const [dState, setDState] = useState(8);
    const [hiddenChannels, setHiddenChannels] = useState(32);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedFile = e.target.files[0];
            setFile(selectedFile);
            setError(null);

            // Auto-detect mode based on extension
            if (selectedFile.name.endsWith('.zip')) {
                setMode('zip');
            } else if (selectedFile.name.endsWith('.pth') || selectedFile.name.endsWith('.pt')) {
                setMode('pth');
                // Auto-fill name if empty
                if (!expName) {
                    const name = selectedFile.name.split('.')[0];
                    setExpName('Imported_' + name.replace(/[^a-zA-Z0-9_-]/g, ''));
                }
            }
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) {
            setError('Por favor seleccione un archivo.');
            return;
        }

        setLoading(true);
        setError(null);
        setSuccess(null);

        const formData = new FormData();
        formData.append('file', file);

        if (mode === 'pth') {
            if (!expName) {
                setError('El nombre del experimento es requerido.');
                setLoading(false);
                return;
            }
            formData.append('experiment_name', expName);
            formData.append('model_architecture', architecture);
            formData.append('d_state', dState.toString());
            formData.append('hidden_channels', hiddenChannels.toString());
        }

        try {
            const response = await fetch('/api/upload_model', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Error en la subida');
            }

            setSuccess(data.message || 'Carga completada exitosamente.');
            setTimeout(() => {
                onUploadSuccess();
                onClose();
            }, 1500);

        } catch (err: any) {
            setError(err.message || 'Error de conexión');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center pointer-events-none">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/80 backdrop-blur-sm pointer-events-auto"
                onClick={onClose}
            />

            <GlassPanel className="relative w-[500px] pointer-events-auto flex flex-col overflow-hidden max-h-[90vh]">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-white/10 bg-white/5">
                    <div className="flex items-center gap-2">
                        <UploadCloud className="text-blue-400" size={18} />
                        <h2 className="text-sm font-bold text-gray-200">Importar Modelo / Experimento</h2>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
                        <X size={18} />
                    </button>
                </div>

                {/* Content */}
                <div className="p-4 space-y-4 overflow-y-auto">

                    {/* Mode Selection */}
                    <div className="flex p-1 bg-black/20 rounded-lg">
                        <button
                            type="button"
                            onClick={() => setMode('zip')}
                            className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-bold rounded-md transition-all ${
                                mode === 'zip'
                                    ? 'bg-blue-500/20 text-blue-400 shadow-sm border border-blue-500/30'
                                    : 'text-gray-400 hover:text-gray-300'
                            }`}
                        >
                            <FileArchive size={14} />
                            Paquete ZIP
                        </button>
                        <button
                            type="button"
                            onClick={() => setMode('pth')}
                            className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs font-bold rounded-md transition-all ${
                                mode === 'pth'
                                    ? 'bg-purple-500/20 text-purple-400 shadow-sm border border-purple-500/30'
                                    : 'text-gray-400 hover:text-gray-300'
                            }`}
                        >
                            <FileCode size={14} />
                            Archivo .PTH
                        </button>
                    </div>

                    <div className="text-xs text-gray-400 px-1">
                        {mode === 'zip'
                            ? "Sube un archivo .zip que contenga 'config.json' y la carpeta de checkpoints. Ideal para mover experimentos completos."
                            : "Sube un archivo de checkpoint (.pth) suelto. Se creará un nuevo experimento con la configuración que indiques abajo."
                        }
                    </div>

                    {/* File Input */}
                    <div
                        className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center gap-2 transition-colors cursor-pointer ${
                            file ? 'border-green-500/30 bg-green-500/5' : 'border-white/10 hover:border-blue-500/30 hover:bg-white/5'
                        }`}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            className="hidden"
                            accept={mode === 'zip' ? ".zip" : ".pth,.pt"}
                            onChange={handleFileChange}
                        />
                        {file ? (
                            <>
                                <CheckCircle className="text-green-400" size={24} />
                                <div className="text-sm font-bold text-gray-200">{file.name}</div>
                                <div className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                            </>
                        ) : (
                            <>
                                <UploadCloud className="text-gray-500" size={24} />
                                <div className="text-xs font-bold text-gray-400">Click para seleccionar archivo</div>
                                <div className="text-[10px] text-gray-500 uppercase font-bold">
                                    {mode === 'zip' ? 'MAX 500MB' : 'MAX 200MB'}
                                </div>
                            </>
                        )}
                    </div>

                    {/* Config Form (Only for .pth) */}
                    {mode === 'pth' && (
                        <div className="space-y-3 pt-2 border-t border-white/10">
                            <h3 className="text-xs font-bold text-purple-400 uppercase tracking-wider">Configuración del Experimento</h3>

                            <div className="space-y-1">
                                <label className="text-[10px] text-gray-400 font-bold uppercase">Nombre del Experimento</label>
                                <input
                                    type="text"
                                    value={expName}
                                    onChange={(e) => setExpName(e.target.value)}
                                    className="w-full bg-black/20 border border-white/10 rounded px-2 py-1.5 text-xs text-gray-200 focus:border-purple-500/50 outline-none"
                                    placeholder="Ej: Imported_MyModel"
                                />
                            </div>

                            <div className="space-y-1">
                                <label className="text-[10px] text-gray-400 font-bold uppercase">Arquitectura</label>
                                <select
                                    value={architecture}
                                    onChange={(e) => setArchitecture(e.target.value)}
                                    className="w-full bg-black/20 border border-white/10 rounded px-2 py-1.5 text-xs text-gray-200 focus:border-purple-500/50 outline-none"
                                >
                                    {ARCHITECTURES.map(arch => (
                                        <option key={arch.value} value={arch.value}>{arch.label}</option>
                                    ))}
                                </select>
                            </div>

                            <div className="grid grid-cols-2 gap-3">
                                <div className="space-y-1">
                                    <label className="text-[10px] text-gray-400 font-bold uppercase">d_state</label>
                                    <input
                                        type="number"
                                        value={dState}
                                        onChange={(e) => setDState(parseInt(e.target.value) || 8)}
                                        className="w-full bg-black/20 border border-white/10 rounded px-2 py-1.5 text-xs text-gray-200 focus:border-purple-500/50 outline-none"
                                    />
                                </div>
                                <div className="space-y-1">
                                    <label className="text-[10px] text-gray-400 font-bold uppercase">Hidden Channels</label>
                                    <input
                                        type="number"
                                        value={hiddenChannels}
                                        onChange={(e) => setHiddenChannels(parseInt(e.target.value) || 32)}
                                        className="w-full bg-black/20 border border-white/10 rounded px-2 py-1.5 text-xs text-gray-200 focus:border-purple-500/50 outline-none"
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Messages */}
                    {error && (
                        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded flex items-start gap-2 text-red-400 text-xs">
                            <AlertTriangle size={14} className="shrink-0 mt-0.5" />
                            <span>{error}</span>
                        </div>
                    )}

                    {success && (
                        <div className="p-3 bg-green-500/10 border border-green-500/30 rounded flex items-center gap-2 text-green-400 text-xs font-bold">
                            <CheckCircle size={14} />
                            <span>{success}</span>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-white/10 bg-white/5 flex justify-end gap-2">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-xs font-bold text-gray-400 hover:text-white transition-colors"
                        disabled={loading}
                    >
                        Cancelar
                    </button>
                    <button
                        onClick={handleSubmit}
                        disabled={loading || !file}
                        className={`px-4 py-2 text-xs font-bold rounded flex items-center gap-2 transition-all ${
                            loading || !file
                                ? 'bg-gray-500/20 text-gray-500 cursor-not-allowed'
                                : 'bg-blue-500 text-white hover:bg-blue-600 shadow-lg shadow-blue-500/20'
                        }`}
                    >
                        {loading ? (
                            <>
                                <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Subiendo...
                            </>
                        ) : (
                            <>
                                <UploadCloud size={14} />
                                Importar
                            </>
                        )}
                    </button>
                </div>
            </GlassPanel>
        </div>
    );
}
