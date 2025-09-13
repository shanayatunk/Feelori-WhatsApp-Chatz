import React from 'react';
import { apiService, DashboardStats, Escalation } from '../../../lib/api';
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

const VolumeChart = ({ data }: { data: { _id: string; count: number }[] }) => {
    const maxValue = Math.max(...data.map(d => d.count), 0);
    const chartData = Array.from({ length: 7 }, (_, i) => {
        const d = new Date();
        d.setDate(d.getDate() - i);
        const dateString = d.toISOString().split('T')[0];
        const entry = data.find(item => item._id === dateString);
        return {
            date: d.toLocaleDateString('en-US', { weekday: 'short' }),
            count: entry ? entry.count : 0,
        };
    }).reverse();

    return (
        <div className="h-64 flex items-end justify-around gap-2 pt-4">
            {chartData.map(({ date, count }) => (
                <div key={date} className="flex-1 flex flex-col items-center gap-2">
                    <div className="text-xs text-gray-500">{count}</div>
                    <div className="w-full bg-gray-200 rounded-md" style={{ height: '100%' }}>
                        <div className="bg-[#ff4d6d] rounded-md" style={{ height: `${maxValue > 0 ? (count / maxValue) * 100 : 0}%` }}></div>
                    </div>
                    <div className="text-xs font-medium text-gray-600">{date}</div>
                </div>
            ))}
        </div>
    );
};


export const DashboardPage = ({ onViewCustomer }: { onViewCustomer: (customerId: string) => void; }) => {
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
        <StatCard title="Total Customers" value={stats?.total_customers ?? 0} />
        <StatCard title="Active Conversations (24h)" value={stats?.active_conversations ?? 0} />
        <StatCard title="Human Escalations (24h)" value={stats?.human_escalations ?? 0} color="#f59e0b" />
        <StatCard title="Avg. Response Time" value={stats?.avg_response_time ?? 'N/A'} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
            <CardHeader>
                <CardTitle>Conversation Volume (Last 7 Days)</CardTitle>
            </CardHeader>
            <CardContent>
                {stats?.conversation_volume ? <VolumeChart data={stats.conversation_volume} /> : <div className="text-center text-gray-500">No data</div>}
            </CardContent>
        </Card>
        <Card>
            <CardHeader>
                <CardTitle>Human Support Requests</CardTitle>
                <CardDescription>Recent conversations that triggered the &apos;human_escalation&apos; intent.</CardDescription>
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