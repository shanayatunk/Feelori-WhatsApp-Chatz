import React from 'react';
import { apiService } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';

// Define types for your data
interface DashboardStats {
  total_customers: number;
  active_conversations: number;
  human_escalations: number;
  avg_response_time: string;
}

interface Escalation {
  _id: string;
  name: string;
  phone_number: string;
}

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

export const DashboardPage = ({ setPage, onViewCustomer }: { setPage: (page: string) => void; onViewCustomer: (customerId: string) => void; }) => {
  const [stats, setStats] = React.useState<DashboardStats | null>(null);
  const [escalations, setEscalations] = React.useState<Escalation[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [statsData, escalationsData] = await Promise.all([
          apiService.getDashboardStats(),
          apiService.getEscalations()
        ]);
        setStats(statsData);
        setEscalations(escalationsData);
      } catch (err) {
        if (err instanceof Error) {
            setError(err.message);
        } else {
            setError('An unknown error occurred while fetching dashboard data.');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div className="text-center p-8">Loading dashboard...</div>;
  if (error) return <div className="text-center p-8 text-red-600">Error: {error}</div>;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-black">Welcome Back!</h1>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Total Customers" value={stats?.total_customers ?? 'N/A'} />
        <StatCard title="Active Conversations" value={stats?.active_conversations ?? 'N/A'} />
        <StatCard title="Human Escalations" value={stats?.human_escalations ?? 'N/A'} color="#f59e0b" />
        <StatCard title="Avg. Response Time" value={stats?.avg_response_time ?? 'N/A'} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
            <CardHeader><CardTitle>Orders Over Time</CardTitle></CardHeader>
            <CardContent><div className="h-64 bg-gray-200 rounded-md flex items-center justify-center text-gray-400">Chart Placeholder</div></CardContent>
        </Card>
        <Card>
            <CardHeader>
                <CardTitle>Human Support Requests</CardTitle>
                <CardDescription>Recent conversations that triggered the 'human_escalation' intent.</CardDescription>
            </CardHeader>
            <CardContent>
                <div className="space-y-2 text-sm">
                    {escalations.length > 0 ? (
                        escalations.map(e => (
                            <div key={e._id} className="flex justify-between items-center p-2 rounded hover:bg-gray-50 text-gray-900">
                                <span>{e.name} ({e.phone_number})</span>
                                <button onClick={() => onViewCustomer(e._id)} className="text-[#ff4d6d] font-medium hover:underline">
                                    View
                                </button>
                            </div>
                        ))
                    ) : (
                        <p className="text-gray-500 text-center py-4">No recent escalation requests.</p>
                    )}
                </div>
            </CardContent>
        </Card>
      </div>
    </div>
  );
};