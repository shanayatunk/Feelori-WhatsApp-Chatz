import React from 'react';
import { apiService } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle } from '../../components/ui/Card';

// Define an interface for the metrics data structure
interface PackingMetrics {
    status_counts: Record<string, number>;
}

export const PerformancePage = () => {
    const [metrics, setMetrics] = React.useState<PackingMetrics | null>(null);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    React.useEffect(() => {
        const fetchMetrics = async () => {
            try {
                setLoading(true);
                const data = await apiService.getPackingMetrics();
                setMetrics(data);
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

        fetchMetrics();
    }, []);

    if (loading) return <div className="text-center p-10 text-gray-900">Loading performance data...</div>;
    if (error) return <div className="text-center p-10 text-red-500">Error: {error}</div>;

    const statusCounts = metrics?.status_counts || {};

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold text-black">Packing Metrics</h1>
             <Card>
                <CardHeader>
                    <CardTitle>Order Status Overview</CardTitle>
                </CardHeader>
                <CardContent>
                   <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left text-gray-700">
                            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                                <tr>
                                    <th scope="col" className="px-6 py-3">Status</th>
                                    <th scope="col" className="px-6 py-3">Number of Orders</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.keys(statusCounts).length > 0 ? Object.entries(statusCounts).map(([status, count]) => (
                                    <tr key={status} className="bg-white border-b">
                                        <td className="px-6 py-4 font-medium text-gray-900">{status}</td>
                                        <td className="px-6 py-4 font-bold text-[#ff4d6d]">{count}</td>
                                    </tr>
                                )) : (
                                    <tr>
                                        <td colSpan={2} className="text-center p-8 text-gray-500">No performance data available.</td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};