import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

export const DashboardLayout = ({ children, activePage, setActivePage, onLogout }: { children: React.ReactNode; activePage: string; setActivePage: (page: string) => void; onLogout: () => void; }) => (
  <div className="grid min-h-screen w-full lg:grid-cols-[256px_1fr]">
    <Sidebar activePage={activePage} setActivePage={setActivePage} />
    <div className="flex flex-col">
      <Header onLogout={onLogout} />
      <main className="flex-1 bg-gray-100 p-4 sm:p-6">{children}</main>
    </div>
  </div>
);
