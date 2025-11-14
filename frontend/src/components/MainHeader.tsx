// frontend/src/components/MainHeader.tsx
import { Group, Burger, Text } from '@mantine/core';
import classes from './MainHeader.module.css';

interface HeaderProps {
    mobileOpened: boolean;
    desktopOpened: boolean;
    toggleMobile: () => void;
    toggleDesktop: () => void;
}

export function MainHeader({ mobileOpened, desktopOpened, toggleMobile, toggleDesktop }: HeaderProps) {
    return (
        <Group h="100%" px="md" justify="space-between">
            <Group>
                <Burger opened={mobileOpened} onClick={toggleMobile} hiddenFrom="sm" size="sm" />
                <Burger opened={desktopOpened} onClick={toggleDesktop} visibleFrom="sm" size="sm" />
                <Text fw={500}>Aetheria Simulation Lab</Text>
            </Group>
        </Group>
    );
}
