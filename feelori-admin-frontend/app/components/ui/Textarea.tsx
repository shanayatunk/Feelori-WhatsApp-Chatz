import React from 'react';

// Combine your custom className prop with all the standard HTML textarea props
type TextareaProps = {
  className?: string;
} & React.TextareaHTMLAttributes<HTMLTextAreaElement>;


export const Textarea = ({ className = '', ...props }: TextareaProps) => (
  <textarea
    className={`flex min-h-[80px] w-full rounded-md border border-gray-300 bg-transparent px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-[#ff4d6d] focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
    suppressHydrationWarning={true}
    {...props}
  />
);