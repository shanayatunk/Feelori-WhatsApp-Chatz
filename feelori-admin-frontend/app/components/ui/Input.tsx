import React from 'react';

export const Input = ({ className = '', ...props }: { className?: string; [key: string]: any; }) => (
  <input
    className={`flex h-10 w-full rounded-md border border-gray-300 bg-transparent px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-[#ff4d6d] focus:ring-offset-2 ${className}`}
    {...props}
  />
);
