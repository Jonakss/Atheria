// frontend/src/components/MainTabs.tsx
import { Tabs, Box } from '@mantine/core';
import { useState } from 'react';
import { IconChartBar, IconDatabase, IconBook, IconChartScatter, IconHistory, IconSettings } from '@tabler/icons-react';
import { motion, AnimatePresence } from 'framer-motion';
import { VisualizationTab } from '../visualization/VisualizationTab';
import { DataTab } from '../training/DataTab';
import { LogbookTab } from '../training/LogbookTab';
import { AnalysisTab } from '../training/AnalysisTab';
import { HistoryViewer } from '../visualization/HistoryViewer';
import { InferenceConfigTab } from '../controls/InferenceConfigTab';
import { fadeIn } from '../../utils/animations';

export function MainTabs() {
    const [activeTab, setActiveTab] = useState<string>('visualization');

    return (
        <Tabs 
            value={activeTab} 
            onChange={(value) => value && setActiveTab(value)}
            style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
        >
            <Tabs.List>
                <Tabs.Tab value="visualization" leftSection={<IconChartBar size={16} />}>
                    Visualización
                </Tabs.Tab>
                <Tabs.Tab value="analysis" leftSection={<IconChartScatter size={16} />}>
                    Análisis t-SNE
                </Tabs.Tab>
                <Tabs.Tab value="data" leftSection={<IconDatabase size={16} />}>
                    Datos y Estadísticas
                </Tabs.Tab>
                <Tabs.Tab value="logbook" leftSection={<IconBook size={16} />}>
                    Bitácora
                </Tabs.Tab>
                <Tabs.Tab value="history" leftSection={<IconHistory size={16} />}>
                    Historia
                </Tabs.Tab>
                <Tabs.Tab value="inference" leftSection={<IconSettings size={16} />}>
                    Configuración
                </Tabs.Tab>
            </Tabs.List>

            <AnimatePresence mode="wait">
                {activeTab === 'visualization' && (
                    <Tabs.Panel value="visualization" key="visualization" style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
                        <motion.div
                            variants={fadeIn}
                            initial="hidden"
                            animate="visible"
                            exit="hidden"
                            style={{ height: '100%', width: '100%' }}
                        >
                            <VisualizationTab />
                        </motion.div>
                    </Tabs.Panel>
                )}

                {activeTab === 'analysis' && (
                    <Tabs.Panel value="analysis" key="analysis" style={{ flex: 1, position: 'relative', overflow: 'auto' }}>
                        <motion.div
                            variants={fadeIn}
                            initial="hidden"
                            animate="visible"
                            exit="hidden"
                            style={{ height: '100%', width: '100%' }}
                        >
                            <AnalysisTab />
                        </motion.div>
                    </Tabs.Panel>
                )}

                {activeTab === 'data' && (
                    <Tabs.Panel value="data" key="data" style={{ flex: 1, position: 'relative', overflow: 'auto' }}>
                        <motion.div
                            variants={fadeIn}
                            initial="hidden"
                            animate="visible"
                            exit="hidden"
                            style={{ height: '100%', width: '100%' }}
                        >
                            <DataTab />
                        </motion.div>
                    </Tabs.Panel>
                )}

                {activeTab === 'logbook' && (
                    <Tabs.Panel value="logbook" key="logbook" style={{ flex: 1, position: 'relative', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                        <motion.div
                            variants={fadeIn}
                            initial="hidden"
                            animate="visible"
                            exit="hidden"
                            style={{ height: '100%', width: '100%' }}
                        >
                            <LogbookTab />
                        </motion.div>
                    </Tabs.Panel>
                )}

                {activeTab === 'history' && (
                    <Tabs.Panel value="history" key="history" style={{ flex: 1, position: 'relative', overflow: 'auto' }}>
                        <motion.div
                            variants={fadeIn}
                            initial="hidden"
                            animate="visible"
                            exit="hidden"
                            style={{ height: '100%', width: '100%' }}
                        >
                            <HistoryViewer />
                        </motion.div>
                    </Tabs.Panel>
                )}

                {activeTab === 'inference' && (
                    <Tabs.Panel value="inference" key="inference" style={{ flex: 1, position: 'relative', overflow: 'auto' }}>
                        <motion.div
                            variants={fadeIn}
                            initial="hidden"
                            animate="visible"
                            exit="hidden"
                            style={{ height: '100%', width: '100%' }}
                        >
                            <InferenceConfigTab />
                        </motion.div>
                    </Tabs.Panel>
                )}
            </AnimatePresence>
        </Tabs>
    );
}

