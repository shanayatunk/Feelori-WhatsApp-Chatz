import React from 'react';
import { apiService, PackerPerformanceData } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';

// A simple component for the KPI cards
const KpiCard = ({ title, value, unit = '' }: { title: string, value: string | number, unit?: string }) => (
    <Card>
        <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-500">{title}</CardTitle>
        </CardHeader>
        <CardContent>
            <div className="text-3xl font-bold text-black">
                {value}
                {unit && <span className="text-lg font-normal ml-1">{unit}</span>}
            </div>
        </CardContent>
    </Card>
);

export const PerformancePage = () => {
    const [data, setData] = React.useState<PackerPerformanceData | null>(null);
    const [days, setDays] = React.useState(7);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    React.useEffect(() => {
        const fetchMetrics = async () => {
            try {
                setLoading(true);
                const result = await apiService.getPackerPerformance(days);
                setData(result);
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
    }, [days]); // Re-fetch when 'days' changes

    if (loading) return <div className="text-center p-10 text-gray-900">Loading performance data...</div>;
    if (error) return <div className="text-center p-10 text-red-500">Error: {error}</div>;
    if (!data) return <div className="text-center p-10 text-gray-500">No performance data available.</div>;

    const kpis = data.kpis;

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap justify-between items-center gap-4">
                <h1 className="text-3xl font-bold text-black">Packer Performance</h1>
                <div className="flex items-center gap-2 p-1 bg-gray-100 rounded-lg">
                    {[7, 30, 90].map(d => (
                        <Button
                            key={d}
                            variant={days === d ? 'primary' : 'ghost'}
                            onClick={() => setDays(d)}
                            // --- UI FIX: Added classes to make inactive buttons darker ---
                            className={`px-4 py-2 ${days !== d && 'text-gray-700 font-semibold'}`}
                        >
                            Last {d} Days
                        </Button>
                    ))}
                </div>
            </div>

            {/* KPI Cards Section */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <KpiCard title="Completed Orders" value={kpis.completed_orders} />
                <KpiCard title="Avg. Time to Pack" value={kpis.avg_time_to_pack_minutes} unit="min" />
                <KpiCard title="Orders On Hold" value={kpis.on_hold_orders} />
                <KpiCard title="Hold Rate" value={`${kpis.hold_rate}%`} />
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Packer Leaderboard */}
                <Card className="lg:col-span-2">
                    <CardHeader>
                        <CardTitle>Packer Leaderboard</CardTitle>
                        <CardDescription>Orders packed in the selected period.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <table className="w-full text-sm">
                            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                                <tr>
                                    <th className="px-4 py-3">Packer Name</th>
                                    <th className="px-4 py-3 text-right">Orders Packed</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.packer_leaderboard.map((packer) => (
                                    <tr key={packer._id} className="border-b">
                                        <td className="px-4 py-3 font-medium">{packer._id}</td>
                                        <td className="px-4 py-3 text-right font-bold text-[#ff4d6d]">{packer.orders_packed}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </CardContent>
                </Card>

                {/* Hold Analysis */}
                <Card>
                    <CardHeader>
                        <CardTitle>Hold Analysis</CardTitle>
                        <CardDescription>Top reasons for putting orders on hold.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div>
                            {/* --- UI FIX: Added classes to make subheading darker --- */}
                            <h3 className="text-sm font-medium mb-2 text-gray-900">By Reason</h3>
                            {data.hold_analysis.by_reason.map(reason => (
                                <div key={reason._id} className="flex justify-between text-sm">
                                    <span>{reason._id}</span>
                                    <span className="font-semibold">{reason.count}</span>
                                </div>
                            ))}
                        </div>
                        <div>
                            {/* --- UI FIX: Added classes to make subheading darker --- */}
                            <h3 className="text-sm font-medium mb-2 text-gray-900">Top Problem SKUs</h3>
                             {data.hold_analysis.top_problem_skus.map(sku => (
                                <div key={sku._id} className="flex justify-between text-sm">
                                    <span className="font-mono text-xs">{sku._id}</span>
                                    <span className="font-semibold">{sku.count}</span>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};