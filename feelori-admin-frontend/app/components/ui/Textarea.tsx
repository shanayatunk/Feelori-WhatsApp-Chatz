import React from 'react';

export const Textarea = ({ className = '', ...props }: { className?: string; [key: string]: any; }) => (
    <textarea
        className={`w-full rounded-md border border-gray-300 p-2 font-mono text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-[#ff4d6d] focus:ring-offset-2 ${className}`}
        {...props}
    />
);
