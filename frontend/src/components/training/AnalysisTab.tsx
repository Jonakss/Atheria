// frontend/src/components/AnalysisTab.tsx
import { Box, Stack, Grid } from '@mantine/core';
import { UniverseAtlasViewer } from '../visualization/UniverseAtlasViewer';
import { CellChemistryViewer } from '../visualization/CellChemistryViewer';

export function AnalysisTab() {
    return (
        <Box style={{ height: '100%', overflow: 'auto', padding: 'var(--mantine-spacing-md)' }}>
            <Stack gap="md">
                <Grid gutter="md">
                    <Grid.Col span={{ base: 12, md: 6 }}>
                        <UniverseAtlasViewer />
                    </Grid.Col>
                    <Grid.Col span={{ base: 12, md: 6 }}>
                        <CellChemistryViewer />
                    </Grid.Col>
                </Grid>
            </Stack>
        </Box>
    );
}

