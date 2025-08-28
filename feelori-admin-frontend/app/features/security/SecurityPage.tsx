import React from 'react';
import { apiService } from '../../../lib/api';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';

export const SecurityPage = () => {
    const [events, setEvents] = React.useState<any[]>([]);
    const [loading, setLoading] = React.useState(true);
    const [error, setError] = React.useState<string | null>(null);

    React.useEffect(() => {
        const fetchEvents = async () => {
            try {
                setLoading(true);
                const data = await apiService.getSecurityEvents();
                setEvents(data);
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchEvents();
    }, []);

    if (loading) return <div className="text-center p-10 text-gray-900">Loading security events...</div>;
    if (error) return <div className="text-center p-10 text-red-500">Error: {error}</div>;

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold text-black">Security Log</h1>
             <Card>
                <CardHeader>
                    <CardTitle>Recent Security Events</CardTitle>
                    <CardDescription>A log of important security-related activities.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left text-gray-700">
                            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
                                <tr>
                                    <th scope="col" className="px-6 py-3">Event</th>
                                    <th scope="col" className="px-6 py-3">IP Address</th>
                                    <th scope="col" className="px-6 py-3">Timestamp</th>
                                    <th scope="col" className="px-6 py-3">Details</th>
                                </tr>
                            </thead>
                            <tbody>
                                {events.map(event => (
                                    <tr key={event._id} className="bg-white border-b">
                                        <td className="px-6 py-4 font-medium text-gray-900 capitalize">{event.event_type.replace(/_/g, ' ')}</td>
                                        <td className="px-6 py-4 text-gray-900">{event.ip_address}</td>
                                        <td className="px-6 py-4 text-gray-900">{new Date(event.timestamp).toLocaleString()}</td>
                                        <td className="px-6 py-4 text-xs text-gray-900"><pre>{JSON.stringify(event.details, null, 2)}</pre></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};
