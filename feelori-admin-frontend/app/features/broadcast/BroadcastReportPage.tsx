import React from 'react';
import { apiService, BroadcastDetails, Recipient, Pagination } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';
import { Input } from '../../components/ui/Input';

const StatCard = ({ title, value, color }: { title: string; value: number; color?: string; }) => (
    <Card>
        <CardHeader><CardTitle className="text-sm font-medium text-gray-500">{title}</CardTitle></CardHeader>
        <CardContent><div className="text-2xl font-bold" style={{ color }}>{value}</div></CardContent>
    </Card>
);

export const BroadcastReportPage = ({ jobId, onBack }: { jobId: string; onBack: () => void; }) => {
    const [details, setDetails] = React.useState<BroadcastDetails | null>(null);
    const [recipients, setRecipients] = React.useState<Recipient[]>([]);
    const [pagination, setPagination] = React.useState<Pagination | null>(null);
    const [page, setPage] = React.useState(1);
    const [search, setSearch] = React.useState('');
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    React.useEffect(() => {
        if (!jobId) return;
        const fetchData = async () => {
            try {
                setLoading(true);
                const [detailsData, recipientsData] = await Promise.all([
                    apiService.getBroadcastDetails(jobId),
                    apiService.getBroadcastRecipients(jobId, page, search)
                ]);
                setDetails(detailsData);
                setRecipients(recipientsData.recipients);
                setPagination(recipientsData.pagination);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'An error occurred.');
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [jobId, page, search]);

    const handleDownloadCsv = () => {
        const token = localStorage.getItem('feelori_admin_token');
        const url = `${process.env.NEXT_PUBLIC_API_URL}/admin/broadcasts/${jobId}/recipients/csv?token=${token}`;
        window.open(url, '_blank');
    };

    if (loading && !details) return <div>Loading report...</div>;
    if (error) return <div className="text-red-500">Error: {error}</div>;
    if (!details) return <div>No details found for this broadcast.</div>;

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
                        <StatCard title="Total Recipients" value={details.stats.total_recipients} color="#000000" />
                        <StatCard title="Sent" value={details.stats.sent} color="#3b82f6" />
                        <StatCard title="Delivered" value={details.stats.delivered} color="#22c55e" />
                        <StatCard title="Read" value={details.stats.read} color="#8b5cf6" />
                        <StatCard title="Failed" value={details.stats.failed} color="#ef4444" />
                    </div>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <div className="flex justify-between items-center">
                        <div>
                            <CardTitle>Recipients ({pagination?.total})</CardTitle>
                            <CardDescription>List of all users included in this broadcast.</CardDescription>
                        </div>
                        <div className="flex gap-2">
                           <Input placeholder="Search recipients..." value={search} onChange={e => setSearch(e.target.value)} />
                           <Button onClick={handleDownloadCsv} variant="secondary">Download CSV</Button>
                        </div>
                    </div>
                </CardHeader>
                <CardContent>
                    <table className="w-full text-sm">
                         <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                            <tr>
                                <th className="px-4 py-3">Name</th>
                                <th className="px-4 py-3">Phone Number</th>
                                <th className="px-4 py-3">Final Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {recipients.map(r => (
                                <tr key={r._id} className="border-b">
                                    <td className="px-4 py-3 font-medium text-gray-800">{r.customer_info?.name || 'N/A'}</td>
                                    <td className="px-4 py-3 text-gray-800">{r.phone}</td>
                                    <td className="px-4 py-3 capitalize text-gray-800">{r.status}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                     {/* Pagination Controls */}
                    <div className="flex justify-end items-center mt-4 gap-2">
                        <span>Page {pagination?.page} of {pagination?.pages}</span>
                        <Button onClick={() => setPage(p => p - 1)} disabled={pagination?.page === 1}>Previous</Button>
                        <Button onClick={() => setPage(p => p + 1)} disabled={pagination?.page === pagination?.pages}>Next</Button>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};