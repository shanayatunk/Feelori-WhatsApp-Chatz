import React from 'react';
import { apiService, BroadcastJob, Pagination } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Button } from '../../components/ui/Button';

export const BroadcastHistoryPage = ({ onSelectBroadcast }: { onSelectBroadcast: (jobId: string) => void; }) => {
    const [jobs, setJobs] = React.useState<BroadcastJob[]>([]);
    const [pagination, setPagination] = React.useState<Pagination | null>(null);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    React.useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const data = await apiService.getBroadcasts(1, 20);
                setJobs(data.broadcasts);
                setPagination(data.pagination);
            } catch (err) {
                setError(err instanceof Error ? err.message : 'An error occurred.');
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    if (loading) return <div>Loading broadcast history...</div>;
    if (error) return <div className="text-red-500">Error: {error}</div>;

    return (
        <Card>
            <CardHeader>
                <CardTitle>Broadcast History</CardTitle>
                <CardDescription>A log of all past broadcast campaigns.</CardDescription>
            </CardHeader>
            <CardContent>
                <table className="w-full text-sm">
                    <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                        <tr>
                            <th className="px-4 py-3">Date</th>
                            <th className="px-4 py-3">Message</th>
                            <th className="px-4 py-3">Target</th>
                            <th className="px-4 py-3">Status</th>
                            <th className="px-4 py-3"></th>
                        </tr>
                    </thead>
                    <tbody>
                        {jobs.map(job => (
                            <tr key={job._id} className="border-b">
                                <td className="px-4 py-3">{new Date(job.created_at).toLocaleString()}</td>
                                <td className="px-4 py-3 text-gray-600 truncate max-w-xs">{job.message}</td>
                                <td className="px-4 py-3 capitalize">{job.target_type}</td>
                                <td className="px-4 py-3 capitalize">{job.status}</td>
                                <td className="px-4 py-3 text-right">
                                    <Button variant="secondary" size="sm" onClick={() => onSelectBroadcast(job._id)}>
                                        View Report
                                    </Button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </CardContent>
        </Card>
    );
};