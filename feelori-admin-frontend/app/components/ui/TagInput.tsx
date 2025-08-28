import React from 'react';

export const TagInput = ({ tags, setTags, placeholder }: { tags: string[]; setTags: (tags: string[]) => void; placeholder: string; }) => {
    const [inputValue, setInputValue] = React.useState('');

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && inputValue.trim()) {
            e.preventDefault();
            if (!tags.includes(inputValue.trim().toLowerCase())) {
                setTags([...tags, inputValue.trim().toLowerCase()]);
            }
            setInputValue('');
        }
    };

    const removeTag = (tagToRemove: string) => {
        setTags(tags.filter(tag => tag !== tagToRemove));
    };

    return (
        <div className="flex flex-wrap items-center gap-2 p-2 border border-gray-300 rounded-md bg-white">
            {tags.map(tag => (
                <span key={tag} className="flex items-center gap-1 bg-[#ff4d6d] text-white text-sm font-medium px-2 py-1 rounded-full">
                    {tag}
                    <button onClick={() => removeTag(tag)} className="text-white hover:text-gray-200">&times;</button>
                </span>
            ))}
            <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={placeholder}
                className="flex-grow bg-transparent outline-none text-sm text-gray-900"
            />
        </div>
    );
};
