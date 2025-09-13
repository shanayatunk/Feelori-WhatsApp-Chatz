import React from 'react';
import { apiService, BroadcastDetails } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';

const StatCard = ({ title, value, color }: { title: string; value: number; color?: string; }) => (
    <Card>
        <CardHeader><CardTitle className="text-sm font-medium text-gray-500">{title}</CardTitle></CardHeader>
        <CardContent><div className="text-2xl font-bold" style={{ color }}>{value}</div></CardContent>
    </Card>
);

export const BroadcastReportPage = ({ jobId, onBack }: { jobId: string; onBack: () => void; }) => {
    const [details, setDetails] = React.useState<BroadcastDetails | null>(null);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    React.useEffect(() => {
        if (!jobId) return;
        const fetchData = async () => {
            try {
                setLoading(true);
                const data = await apiService.getBroadcastDetails(jobId);
                setDetails(data);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'An error occurred.');
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [jobId]);

    if (loading) return <div>Loading report...</div>;
    if (error) return <div className="text-red-500">Error: {error}</div>;
    if (!details) return <div>No details found for this broadcast.</div>;

    const stats = details.stats;

    return (
        <div className="space-y-6">
            <Button onClick={onBack} variant="secondary">← Back to History</Button>
            <Card>
                <CardHeader>
                    <CardTitle>Broadcast Report</CardTitle>
                    <CardDescription>Sent on {new Date(details.created_at).toLocaleString()}</CardDescription>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-gray-700 border p-3 rounded-md bg-gray-50 mb-6">{details.message}</p>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <StatCard title="Total Recipients" value={stats.total_recipients} />
                        <StatCard title="Sent" value={stats.sent} color="#3b82f6" />
                        <StatCard title="Delivered" value={stats.delivered} color="#22c55e" />
                        <StatCard title="Read" value={stats.read} color="#8b5cf6" />
                        <StatCard title="Failed" value={stats.failed} color="#ef4444" />
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};