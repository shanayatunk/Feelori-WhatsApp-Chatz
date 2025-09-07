import React from 'react';

// Combine your custom className prop with all the standard HTML input props
type InputProps = {
  className?: string;
} & React.InputHTMLAttributes<HTMLInputElement>;

export const Input = ({ className = '', ...props }: InputProps) => (
  <input
    className={`flex h-10 w-full rounded-md border border-gray-300 bg-transparent px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-[#ff4d6d] focus:ring-offset-2 ${className}`}
    suppressHydrationWarning={true}
    {...props}
  />
);