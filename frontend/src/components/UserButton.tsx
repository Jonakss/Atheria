import { UnstyledButton, Group, Avatar, Text } from '@mantine/core';
import classes from './UserButton.module.css';

export function UserButton() {
  return (
    <UnstyledButton className={classes.user}>
      <Group>
        <Avatar
          src="https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/avatars/avatar-8.png"
          radius="xl"
        />

        <div style={{ flex: 1 }}>
          <Text size="sm" fw={500}>
            Jonathan Correa
          </Text>

          <Text c="dimmed" size="xs">
            jonathan.correa@example.com
          </Text>
        </div>
      </Group>
    </UnstyledButton>
  );
}
