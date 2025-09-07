import React from 'react';
import { apiService } from '../../../lib/api';
import { Button } from '../../components/ui/Button';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Input } from '../../components/ui/Input';
import { Textarea } from '../../components/ui/Textarea';

export const BroadcastPage = () => {
    const [message, setMessage] = React.useState('');
    const [imageUrl, setImageUrl] = React.useState('');
    const [target, setTarget] = React.useState('active');
    const [isSending, setIsSending] = React.useState(false);
    const [result, setResult] = React.useState<{success: boolean, message: string} | null>(null);

    const handleSend = async () => {
        if (!message || isSending) return;
        setIsSending(true);
        setResult(null);

        try {
            const response = await apiService.broadcast(message, target, imageUrl);
            setResult({ success: true, message: response.message });
            setMessage('');
            setImageUrl('');
        } catch (err) {
            if (err instanceof Error) {
                setResult({ success: false, message: err.message });
            } else {
                setResult({ success: false, message: 'An unknown error occurred.'});
            }
        } finally {
            setIsSending(false);
        }
    };

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold text-black">Broadcast Center</h1>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="md:col-span-2">
                    <Card>
                        <CardHeader>
                            <CardTitle>Compose Message</CardTitle>
                            {/* THIS IS THE CORRECTED LINE */}
                            <CardDescription>Send a message to a group of customers. Use placeholders like `{'{'}{'{'}name{'}'}{'}'}`.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <Textarea
                                rows={6}
                                placeholder="Your message here..."
                                value={message}
                                onChange={(e) => setMessage(e.target.value)}
                            />
                            <Input
                                placeholder="Optional: Image URL (e.g., https://.../image.png)"
                                value={imageUrl}
                                onChange={(e) => setImageUrl(e.target.value)}
                            />
                             <div>
                                <label className="text-sm font-medium text-gray-900">Target Audience</label>
                                <div className="mt-2 space-y-2">
                                    {['all', 'active', 'inactive'].map((t) => (
                                        <label key={t} className="flex items-center">
                                            <input type="radio" value={t} checked={target === t} onChange={() => setTarget(t)} className="h-4 w-4 text-[#ff4d6d] focus:ring-[#ff4d6d] border-gray-300"/>
                                            <span className="ml-2 text-sm text-gray-900 capitalize">{t} Customers</span>
                                        </label>
                                    ))}
                                </div>
                            </div>
                            <Button onClick={handleSend} disabled={isSending || !message} className="w-full">
                                {isSending ? 'Sending...' : 'Send Broadcast'}
                            </Button>
                            {result && <div className={`p-3 rounded-md text-sm ${result.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>{result.message}</div>}
                        </CardContent>
                    </Card>
                </div>
                <div className="md:col-span-1">
                     <Card>
                        <CardHeader><CardTitle>Manage Custom Groups</CardTitle></CardHeader>
                        <CardContent className="space-y-4">
                            <Input placeholder="New group name..." />
                            <Textarea rows={4} placeholder="Paste phone numbers, one per line..."></Textarea>
                            <Button variant="secondary" className="w-full">Create Group</Button>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
};