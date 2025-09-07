import React from 'react';
import { apiService } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';

// Define a type for your health status data
interface HealthStatus {
  status: string;
  services: {
    [key: string]: string; // e.g., 'database': 'connected'
  };
}

export const HealthPage = () => {
    const [health, setHealth] = React.useState<HealthStatus | null>(null);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    React.useEffect(() => {
        const fetchHealth = async () => {
            try {
                setLoading(true);
                const data = await apiService.getHealth();
                setHealth(data);
            } catch (err) {
                if (err instanceof Error) {
                    setError(err.message);
                } else {
                    setError('An unknown error occurred.');
                }
            } finally {
                setLoading(false);
            }
        };

        fetchHealth();
    }, []);

    const StatusIndicator = ({ status }: { status: string }) => {
        const isOk = status === 'connected' || status === 'configured';
        return (
            <div className={`flex items-center gap-2 text-sm font-medium ${isOk ? 'text-green-600' : 'text-red-600'}`}>
                <span className={`h-2.5 w-2.5 rounded-full ${isOk ? 'bg-green-500' : 'bg-red-500'}`}></span>
                {status.replace('_', ' ')}
            </div>
        );
    };

    if (loading) return <div className="text-center p-8">Checking system health...</div>;
    if (error) return <div className="text-center p-8 text-red-600">Error: {error}</div>;

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold text-black">System Health</h1>
                <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold ${health?.status === 'ok' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                    <span className={`h-2.5 w-2.5 rounded-full ${health?.status === 'ok' ? 'bg-green-500' : 'bg-red-500'}`}></span>
                    {health?.status === 'ok' ? 'All Systems Operational' : 'System Degraded'}
                </div>
            </div>
            <Card>
                <CardHeader>
                    <CardTitle>Service Status</CardTitle>
                    <CardDescription>Real-time status of all critical services.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        {health && Object.entries(health.services).map(([service, status]) => (
                            <div key={service} className="p-4 bg-gray-50 rounded-lg border">
                                <h4 className="font-semibold capitalize text-gray-700">{service}</h4>
                                <StatusIndicator status={status as string} />
                            </div>
                        ))}
                    </div>
                </CardContent>
            </Card>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                    <CardHeader><CardTitle>Circuit Breakers</CardTitle></CardHeader>
                    <CardContent><div className="h-48 bg-gray-200 rounded-md flex items-center justify-center text-gray-400">Circuit Breaker Status Placeholder</div></CardContent>
                </Card>
                <Card>
                    <CardHeader><CardTitle>Recent Critical Alerts</CardTitle></CardHeader>
                    <CardContent><div className="h-48 bg-gray-200 rounded-md flex items-center justify-center text-gray-400">Alerts Feed Placeholder</div></CardContent>
                </Card>
            </div>
        </div>
    );
};