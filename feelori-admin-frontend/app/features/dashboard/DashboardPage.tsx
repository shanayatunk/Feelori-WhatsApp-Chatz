import React from 'react';
import { apiService } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';

const StatCard = ({ title, value, icon, color = '#ff4d6d' }: { title: string; value: string | number; icon?: React.ReactNode; color?: string; }) => (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <h3 className="text-sm font-medium text-gray-700">{title}</h3>
        {icon}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold" style={{ color }}>{value}</div>
      </CardContent>
    </Card>
  );

export const DashboardPage = ({ setPage }: { setPage: (page: string) => void; }) => {
  const [stats, setStats] = React.useState<any>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const fetchStats = async () => {
      try {
        setLoading(true);
        const data = await apiService.getStats();
        setStats(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  if (loading) return <div className="text-center p-10 text-gray-900">Loading dashboard...</div>;
  if (error) return <div className="text-center p-10 text-red-500">Error: {error}</div>;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-black">Business Performance</h1>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Total Customers" value={stats?.customers.total ?? '...'} />
        <StatCard title="Active Customers (24h)" value={stats?.customers.active_24h ?? '...'} />
        <StatCard title="Messages (24h)" value={stats?.messages.total_24h ?? '...'} />
        <StatCard title="Queue Size" value={stats?.system.queue_size ?? '...'} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
            <CardHeader><CardTitle>Orders Over Time</CardTitle></CardHeader>
            <CardContent><div className="h-64 bg-gray-200 rounded-md flex items-center justify-center text-gray-400">Chart Placeholder</div></CardContent>
        </Card>
        <Card>
            <CardHeader><CardTitle>Human Support Requests</CardTitle></CardHeader>
            <CardContent>
                <p className="text-sm text-gray-700 mb-4">Conversations that triggered the 'human_escalation' intent.</p>
                <div className="space-y-2 text-sm">
                    <div className="flex justify-between p-2 rounded hover:bg-gray-50 text-gray-900"><span>Priya Sharma (+91...210)</span><a href="#" className="text-[#ff4d6d] font-medium">View</a></div>
                    <div className="flex justify-between p-2 rounded hover:bg-gray-50 text-gray-900"><span>Rohan Mehra (+91...556)</span><a href="#" className="text-[#ff4d6d] font-medium">View</a></div>
                </div>
            </CardContent>
        </Card>
      </div>
    </div>
  );
};
