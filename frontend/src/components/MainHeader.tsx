import { Group, Title, Burger, ActionIcon, Text, MantineColorScheme } from '@mantine/core';
import { 
  IconDeviceFloppy, 
  IconSettings, 
  IconSun, 
  IconMoon, 
  IconLayoutSidebarRightCollapse, 
  IconLayoutSidebarRightExpand 
} from '@tabler/icons-react';

interface MainHeaderProps {
  navOpened: boolean;
  toggleNav: () => void;
  colorScheme: MantineColorScheme;
  toggleColorScheme: () => void;
  currentProject: string;
  panelsVisible: boolean;
  togglePanels: () => void;
}

const MainHeader: React.FC<MainHeaderProps> = ({
  navOpened,
  toggleNav,
  colorScheme,
  toggleColorScheme,
  currentProject,
  panelsVisible,
  togglePanels,
}) => {
  return (
    <Group h="100%" px="md" justify="space-between">
      <Group>
        <Burger opened={navOpened} onClick={toggleNav} hiddenFrom="sm" size="sm" />
        <Title order={3}>Atheria</Title>
      </Group>
      
      <Group>
        <Text fw={500}>{currentProject}</Text>
      </Group>

      <Group>
        <ActionIcon variant="default" size="lg" aria-label="Guardar">
          <IconDeviceFloppy size={18} />
        </ActionIcon>
        <ActionIcon variant="default" size="lg" aria-label="Configuración">
          <IconSettings size={18} />
        </ActionIcon>
        <ActionIcon variant="default" onClick={toggleColorScheme} size="lg" aria-label="Cambiar Tema">
          {colorScheme === 'dark' ? <IconSun size={18} /> : <IconMoon size={18} />}
        </ActionIcon>
        <ActionIcon variant="default" onClick={togglePanels} size="lg" aria-label="Ocultar Paneles">
          {panelsVisible ? <IconLayoutSidebarRightCollapse size={18} /> : <IconLayoutSidebarRightExpand size={18} />}
        </ActionIcon>
      </Group>
    </Group>
  );
};

export default MainHeader;
