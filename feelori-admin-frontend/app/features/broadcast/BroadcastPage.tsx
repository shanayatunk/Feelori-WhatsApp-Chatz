import React from 'react';
import { apiService, BroadcastGroup } from '../../../lib/api';
import { Button } from '../../components/ui/Button';
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from '../../components/ui/Card';
import { Input } from '../../components/ui/Input';
import { Textarea } from '../../components/ui/Textarea';
import { BroadcastHistoryPage } from './BroadcastHistoryPage';
import { BroadcastReportPage } from './BroadcastReportPage';

// This is the main "Compose" view, extracted into its own component
const ComposeView = () => {
    const [message, setMessage] = React.useState('');
    const [imageUrl, setImageUrl] = React.useState('');
    const [target, setTarget] = React.useState('active');
    const [isSending, setIsSending] = React.useState(false);
    const [result, setResult] = React.useState<{success: boolean, message: string} | null>(null);
    const [groups, setGroups] = React.useState<BroadcastGroup[]>([]);
    const [selectedGroup, setSelectedGroup] = React.useState<string>('');
    const [newGroupName, setNewGroupName] = React.useState('');
    const [newGroupPhones, setNewGroupPhones] = React.useState('');

    const handleCreateGroup = async () => {
        if (!newGroupName || !newGroupPhones) return;
        const phone_numbers = newGroupPhones.split('\n').filter(p => p.trim() !== '');

        try {
            const newGroup = await apiService.createBroadcastGroup(newGroupName, phone_numbers);
            setGroups([...groups, newGroup]);
            setNewGroupName('');
            setNewGroupPhones('');
        } catch (error) {
            console.error("Failed to create group:", error);
        }
    };

    React.useEffect(() => {
        const fetchGroups = async () => {
            try {
                const fetchedGroups = await apiService.getBroadcastGroups();
                setGroups(fetchedGroups);
            } catch (error) {
                console.error("Failed to fetch broadcast groups:", error);
            }
        };
        fetchGroups();
    }, []);

    const handleSend = async () => {
        if (!message || isSending) return;
        setIsSending(true);
        setResult(null);

        try {
            const response = await apiService.broadcast(message, target, imageUrl, selectedGroup);
            setResult({ success: true, message: response.message });
            setMessage('');
            setImageUrl('');
        } catch (err) {
            setResult({ success: false, message: err instanceof Error ? err.message : 'An unknown error occurred.'});
        } finally {
            setIsSending(false);
        }
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2">
                <Card>
                    <CardHeader>
                        <CardTitle>Compose Message</CardTitle>
                        <CardDescription>Send a message to a group of customers. Use placeholders like {'{{name}}'}.</CardDescription>
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
                                <label className="flex items-center">
                                    <input type="radio" value="custom_group" checked={target === 'custom_group'} onChange={() => setTarget('custom_group')} className="h-4 w-4 text-[#ff4d6d] focus:ring-[#ff4d6d] border-gray-300"/>
                                    <span className="ml-2 text-sm text-gray-900">Custom Group</span>
                                </label>
                                {target === 'custom_group' && (
                                    <select
                                        value={selectedGroup}
                                        onChange={(e) => setSelectedGroup(e.target.value)}
                                        className="mt-2 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-[#ff4d6d] focus:border-[#ff4d6d] sm:text-sm rounded-md"
                                    >
                                        <option value="">Select a group</option>
                                        {groups.map((group) => (
                                            <option key={group._id} value={group._id}>{group.name}</option>
                                        ))}
                                    </select>
                                )}
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
                        <Input placeholder="New group name..." value={newGroupName} onChange={(e) => setNewGroupName(e.target.value)} />
                        <Textarea rows={4} placeholder="Paste phone numbers, one per line..." value={newGroupPhones} onChange={(e) => setNewGroupPhones(e.target.value)}></Textarea>
                        <Button variant="secondary" className="w-full" onClick={handleCreateGroup}>Create Group</Button>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};


// This is the main parent component that manages which view is shown
export const BroadcastPage = () => {
    // State to manage the current view: 'compose', 'history', or 'report'
    const [view, setView] = React.useState<'compose' | 'history' | 'report'>('compose');
    // State to store the ID of the broadcast job when viewing a report
    const [selectedJobId, setSelectedJobId] = React.useState<string | null>(null);

    const handleSelectBroadcast = (jobId: string) => {
        setSelectedJobId(jobId);
        setView('report');
    };

    const renderContent = () => {
        switch (view) {
            case 'history':
                return <BroadcastHistoryPage onSelectBroadcast={handleSelectBroadcast} />;
            case 'report':
                return <BroadcastReportPage jobId={selectedJobId!} onBack={() => setView('history')} />;
            case 'compose':
            default:
                return <ComposeView />;
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold text-black">Broadcast Center</h1>
                {view === 'compose' && (
                     <Button onClick={() => setView('history')} variant="secondary">View History</Button>
                )}
            </div>
            {renderContent()}
        </div>
    );
};