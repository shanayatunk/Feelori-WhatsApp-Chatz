import React from 'react';

// Define the component's specific props
type CustomButtonProps = {
  variant?: 'primary' | 'secondary' | 'ghost' | 'success';
  children: React.ReactNode;
}

// Combine your custom props with all the standard HTML button props
type ButtonProps = CustomButtonProps & React.ButtonHTMLAttributes<HTMLButtonElement>;

export const Button = ({ children, className = '', variant = 'primary', ...props }: ButtonProps) => {
  const baseClasses = `inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none px-4 py-2`;

  const variantClasses = {
    primary: 'bg-[#ff4d6d] text-white hover:bg-[#e6395b]',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 border border-gray-200',
    ghost: 'hover:bg-gray-100 hover:text-gray-900',
    success: 'bg-green-600 text-white hover:bg-green-700',
  };

  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      suppressHydrationWarning={true}
      {...props}
    >
      {children}
    </button>
  );
};