declare module '*.module.css' {
  const classes: { [key: string]: string };
  export default classes;
}

declare module 'react-dom/client';

interface ImportMetaEnv {
  readonly APP_VERSION?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}