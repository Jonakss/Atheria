import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import '@mantine/core/styles.css';
import { MantineProvider } from '@mantine/core';

ReactDOM.render(
  <React.StrictMode>
    <MantineProvider>
      <App />
    </MantineProvider>
  </React.StrictMode>,
  document.getElementById('root')
);