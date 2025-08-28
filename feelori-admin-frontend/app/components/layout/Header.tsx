import React from 'react';
import { Button } from '../ui/Button';
import { LogOutIcon } from '../ui/Icons';

export const Header = ({ onLogout }: { onLogout: () => void; }) => (
  <header className="flex h-14 items-center gap-4 border-b bg-white px-6 lg:h-[60px]">
    <div className="w-full flex-1">
      {/* Can add a global search bar here in the future */}
    </div>
    <Button onClick={onLogout} variant="secondary">
      <LogOutIcon className="h-5 w-5 mr-2" /> Logout
    </Button>
  </header>
);
