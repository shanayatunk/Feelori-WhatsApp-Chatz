"use client";
import React from 'react';
import { LoginPage } from './features/auth/LoginPage';
import { DashboardLayout } from './components/layout/DashboardLayout';
import { DashboardPage } from './features/dashboard/DashboardPage';
import { ConversationsPage } from './features/conversations/ConversationsPage';
import { PerformancePage } from './features/performance/PerformancePage';
import { HealthPage } from './features/health/HealthPage';
import { SecurityPage } from './features/security/SecurityPage';
import { BroadcastPage } from './features/broadcast/BroadcastPage';
import { RulesEnginePage } from './features/rules/RulesEnginePage';
import { StringsManagerPage } from './features/strings/StringsManagerPage';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false);
  const [activePage, setActivePage] = React.useState('dashboard');

  React.useEffect(() => {
    const token = localStorage.getItem('feelori_admin_token');
    if (token) {
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = (token: string) => {
    localStorage.setItem('feelori_admin_token', token);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('feelori_admin_token');
    setIsAuthenticated(false);
  };

  if (!isAuthenticated) {
    return <LoginPage onLogin={handleLogin} />;
  }

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return <DashboardPage setPage={setActivePage} />;
      case 'conversations':
        return <ConversationsPage />;
      case 'performance':
        return <PerformancePage />;
      case 'health':
        return <HealthPage />;
      case 'security':
        return <SecurityPage />;
      case 'broadcast':
        return <BroadcastPage />;
      case 'rules':
        return <RulesEnginePage />;
      case 'strings':
        return <StringsManagerPage />;
      default:
        return <DashboardPage setPage={setActivePage} />;
    }
  };

  return (
    <DashboardLayout activePage={activePage} setActivePage={setActivePage} onLogout={handleLogout}>
      {renderPage()}
    </DashboardLayout>
  );
}