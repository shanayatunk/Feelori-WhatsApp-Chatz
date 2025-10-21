import React from 'react';

// Define the component's specific props
// FIX: Add the optional size prop
type CustomButtonProps = {
  variant?: 'primary' | 'secondary' | 'ghost' | 'success';
  size?: 'sm' | 'default' | 'lg'; // Added size prop
  children: React.ReactNode;
}

// Combine your custom props with all the standard HTML button props
type ButtonProps = CustomButtonProps & React.ButtonHTMLAttributes<HTMLButtonElement>;

export const Button = ({
    children,
    className = '',
    variant = 'primary',
    size = 'default', // Default size
    ...props
}: ButtonProps) => {

  // Base classes remain the same
  const baseClasses = `inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none`;

  // Variant classes remain the same
  const variantClasses = {
    primary: 'bg-[#ff4d6d] text-white hover:bg-[#e6395b]',
    secondary: 'bg-gray-100 text-gray-900 hover:bg-gray-200 border border-gray-200',
    ghost: 'hover:bg-gray-100 hover:text-gray-900',
    success: 'bg-green-600 text-white hover:bg-green-700',
  };

  // FIX: Add size classes
  const sizeClasses = {
      sm: 'px-3 py-1.5 text-xs', // Example small size
      default: 'px-4 py-2 text-sm', // Your original/default size
      lg: 'px-6 py-3 text-base', // Example large size
  };

  return (
    <button
      // Apply base, variant, size, and any custom className
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${className}`}
      suppressHydrationWarning={true}
      {...props}
    >
      {children}
    </button>
  );
};