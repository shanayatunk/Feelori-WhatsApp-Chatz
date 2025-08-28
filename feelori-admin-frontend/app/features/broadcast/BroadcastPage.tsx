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
        } catch (err: any) {
            setResult({ success: false, message: err.message });
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
                            <CardTitle>Compose Broadcast</CardTitle>
                            <CardDescription>Send a bulk message with optional media to a targeted group.</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div>
                                <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-1">Message</label>
                                <Textarea id="message" rows={5} placeholder="Type your caption or message here..." value={message} onChange={(e) => setMessage(e.target.value)} />
                                <p className="text-xs text-gray-700 mt-1">{message.length} / 1000 characters</p>
                            </div>
                            <div>
                                <label htmlFor="imageUrl" className="block text-sm font-medium text-gray-700 mb-1">Image URL (Optional)</label>
                                <Input id="imageUrl" type="text" placeholder="https://example.com/image.jpg" value={imageUrl} onChange={(e) => setImageUrl(e.target.value)} />
                                {imageUrl && (
                                    <div className="mt-2">
                                        <img src={imageUrl} alt="Preview" className="w-48 h-48 object-cover rounded-md border" />
                                    </div>
                                )}
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Target Audience</label>
                                <div className="space-y-2">
                                    {['active', 'recent', 'all'].map(t => (
                                        <label key={t} className="flex items-center"><input type="radio" name="target" value={t} checked={target === t} onChange={() => setTarget(t)} className="h-4 w-4 text-[#ff4d6d] focus:ring-[#ff4d6d] border-gray-300"/><span className="ml-2 text-sm text-gray-900 capitalize">{t} Customers</span></label>
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
