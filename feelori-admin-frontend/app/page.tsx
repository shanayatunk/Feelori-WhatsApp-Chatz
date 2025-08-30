"use client";
import React from 'react';
import { LoginPage } from './features/auth/LoginPage';
import { DashboardLayout } from './components/layout/DashboardLayout';
import { DashboardPage } from './features/dashboard/DashboardPage';
import { ConversationsPage } from './features/conversations/ConversationsPage';
import CustomerChatPage from './features/customer/[customerId]/page';
import { PerformancePage } from './features/performance/PerformancePage';
import { HealthPage } from './features/health/HealthPage';
import { SecurityPage } from './features/security/SecurityPage';
import { BroadcastPage } from './features/broadcast/BroadcastPage';
import { RulesEnginePage } from './features/rules/RulesEnginePage';
import { StringsManagerPage } from './features/strings/StringsManagerPage';

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false);
  const [activePage, setActivePage] = React.useState('dashboard');
  const [viewingCustomerId, setViewingCustomerId] = React.useState<string | null>(null);

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

  const handleViewCustomer = (customerId: string) => {
    setActivePage('conversations');
    setViewingCustomerId(customerId);
  };

  const handleBackToConversations = () => {
    setViewingCustomerId(null);
  };

  // FIX: This new function handles all sidebar navigation clicks.
  // It ensures that if you're viewing a customer, you're taken out of that view first.
  const handleSetActivePage = (page: string) => {
    setViewingCustomerId(null); // Exit customer view
    setActivePage(page);      // Switch to the new page
  };


  if (!isAuthenticated) {
    return <LoginPage onLogin={handleLogin} />;
  }

  const renderPage = () => {
    if (viewingCustomerId) {
        return <CustomerChatPage customerId={viewingCustomerId} onBack={handleBackToConversations} />;
    }

    switch (activePage) {
      case 'dashboard':
        return <DashboardPage setPage={setActivePage} onViewCustomer={handleViewCustomer} />;
      case 'conversations':
        return <ConversationsPage onViewCustomer={handleViewCustomer} />;
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
        return <DashboardPage setPage={setActivePage} onViewCustomer={handleViewCustomer} />;
    }
  };

  return (
    <DashboardLayout activePage={activePage} setActivePage={handleSetActivePage} onLogout={handleLogout}>
      {renderPage()}
    </DashboardLayout>
  );
}