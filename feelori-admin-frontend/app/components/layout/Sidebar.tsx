import React from 'react';
import { HomeIcon, UsersIcon, MessageSquareIcon, HeartPulseIcon, ShieldCheckIcon, SendIcon, SettingsIcon, FileTextIcon } from '../ui/Icons';

export const Sidebar = ({ activePage, setActivePage }: { activePage: string; setActivePage: (page: string) => void; }) => {
  const navItems = [
    { type: 'header', label: 'Analytics' },
    { id: 'dashboard', label: 'Dashboard', icon: <HomeIcon className="h-5 w-5" /> },
    { id: 'conversations', label: 'Conversations', icon: <MessageSquareIcon className="h-5 w-5" /> },
    { id: 'performance', label: 'Packer Performance', icon: <UsersIcon className="h-5 w-5" /> },
    { type: 'divider' as const },
    { type: 'header', label: 'Administration' },
    { id: 'health', label: 'System Health', icon: <HeartPulseIcon className="h-5 w-5" /> },
    { id: 'security', label: 'Security', icon: <ShieldCheckIcon className="h-5 w-5" /> },
    { id: 'broadcast', label: 'Broadcast', icon: <SendIcon className="h-5 w-5" /> },
    { id: 'triage', label: 'Triage Tickets', icon: <FileTextIcon className="h-5 w-5" /> },
    { type: 'divider' as const },
    { type: 'header', label: 'Configuration' },
    { id: 'rules', label: 'Rules Engine', icon: <SettingsIcon className="h-5 w-5" /> },
    { id: 'strings', label: 'Strings Manager', icon: <FileTextIcon className="h-5 w-5" /> },
  ];

  return (
    <aside className="hidden w-64 flex-col border-r bg-gray-50 lg:flex">
      <div className="flex h-16 items-center border-b px-6">
        <a href="#" className="flex items-center gap-2 font-semibold">
          <span className="text-black">FeelOri Admin</span>
        </a>
      </div>
      <nav className="flex-1 overflow-auto py-4">
        <ul className="grid items-start px-4 text-sm font-medium">
          {navItems.map((item, index) => {
            if ('type' in item && item.type === 'header') {
                return <li key={index} className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">{item.label}</li>
            }
            if ('type' in item && item.type === 'divider') {
                return <li key={index}><hr className="my-2 border-gray-200" /></li>
            }
            // The item must be a clickable link if it's not a header or divider
            const linkItem = item as { id: string; label: string; icon: React.ReactNode; };
            return (
            <li key={linkItem.id}>
              <a
                href="#"
                onClick={(e) => { e.preventDefault(); setActivePage(linkItem.id); }}
                className={`flex items-center gap-3 rounded-lg px-3 py-2 transition-all ${activePage === linkItem.id ? 'bg-[#ff4d6d] text-white' : 'text-gray-700 hover:bg-gray-100'}`}
              >
                {linkItem.icon}
                {linkItem.label}
              </a>
            </li>
            )
          })}
        </ul>
      </nav>
    </aside>
  );
};