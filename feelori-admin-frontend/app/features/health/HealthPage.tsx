import React from 'react';
import { apiService } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';

export const HealthPage = () => {
    const [health, setHealth] = React.useState<any>(null);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    React.useEffect(() => {
        const fetchHealth = async () => {
            try {
                setLoading(true);
                const data = await apiService.getHealth();
                setHealth(data);
            } catch (err: any) {
                setError(err.message);
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

    if (loading) return <div className="text-center p-10 text-gray-900">Loading system health...</div>;
    if (error) return <div className="text-center p-10 text-red-500">Error: {error}</div>;

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold text-black">System Health</h1>
            <Card>
                <CardHeader>
                    <CardTitle>Live Service Status</CardTitle>
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
