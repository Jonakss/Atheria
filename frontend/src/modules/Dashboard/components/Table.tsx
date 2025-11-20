import React from 'react';

interface TableProps {
  children: React.ReactNode;
  highlightOnHover?: boolean;
}

interface TableHeadProps {
  children: React.ReactNode;
}

interface TableBodyProps {
  children: React.ReactNode;
}

interface TableRowProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

interface TableCellProps {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Table: Componente de tabla según Design System.
 */
export const Table: React.FC<TableProps> = ({ children, highlightOnHover = false }) => {
  return (
    <div className="overflow-x-auto">
      <table className={`w-full border-collapse ${highlightOnHover ? '' : ''}`}>
        {children}
      </table>
    </div>
  );
};

export const TableHead: React.FC<TableHeadProps> = ({ children }) => {
  return <thead className="border-b border-white/10">{children}</thead>;
};

export const TableBody: React.FC<TableBodyProps> = ({ children }) => {
  return <tbody>{children}</tbody>;
};

export const TableRow: React.FC<TableRowProps> = ({ children, className = '', style = {} }) => {
  return (
    <tr 
      className={`border-b border-white/5 hover:bg-white/5 transition-colors ${className}`}
      style={style}
    >
      {children}
    </tr>
  );
};

export const TableCell: React.FC<TableCellProps> = ({ children, className = '', style }) => {
  return (
    <td className={`px-3 py-2 text-xs text-gray-300 ${className}`} style={style}>
      {children}
    </td>
  );
};

// Alias para compatibilidad con Mantine
export const TableThead = TableHead;
export const TableTbody = TableBody;
export const TableTr = TableRow;
export const TableTh = ({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) => (
  <th className="px-3 py-2 text-[10px] font-bold text-gray-400 uppercase tracking-wider text-left border-b border-white/10" style={style}>
    {children}
  </th>
);
export const TableTd = TableCell;

