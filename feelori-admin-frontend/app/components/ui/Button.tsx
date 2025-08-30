import React from 'react';

export const Button = ({ children, onClick, className = '', disabled = false, variant = 'primary', ...props }: { children: React.ReactNode; onClick?: () => void; className?: string; disabled?: boolean; variant?: 'primary' | 'secondary' | 'ghost' | 'success'; [key: string]: any; }) => {
  const baseClasses = `inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none px-4 py-2`;

  const variantClasses = {
    primary: 'bg-[#ff4d6d] text-white hover:bg-[#e6395b]',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 border border-gray-200',
    ghost: 'hover:bg-gray-100 hover:text-gray-900',
    success: 'bg-green-600 text-white hover:bg-green-700',
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      suppressHydrationWarning={true}
      {...props}
    >
      {children}
    </button>
  );
};
